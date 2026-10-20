import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

class multiTimeAttention(nn.Module):
    def __init__(self, embed_dim, nhidden, num_heads):
        super(multiTimeAttention, self).__init__()
        self.embed_dim = embed_dim
        self.nhidden = nhidden
        self.num_heads = num_heads
        self.head_dim = nhidden // num_heads
        assert self.head_dim * num_heads == self.nhidden, "nhidden must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, nhidden)
        self.k_proj = nn.Linear(embed_dim, nhidden)
        self.v_proj = nn.Linear(embed_dim, nhidden)
        self.out_proj = nn.Linear(nhidden, embed_dim)
        
        self.k_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.v_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        
    def forward(self, q, k, v, mask=None):
        bsz, nq, embed_dim = q.size()
        _, nk, nv, embed_dim = k.size()
        if v.shape != k.shape:
            raise ValueError(f'key/value shapes differ: {k.shape} vs {v.shape}')
        
        # Project q, k, v
        q = self.q_proj(q)
        k = rearrange(k, 'b nk nv d -> (b nk) d nv 1')
        v = rearrange(v, 'b nk nv d -> (b nk) d nv 1')
        
        k = self.k_conv(k).squeeze(-1).transpose(1, 2).contiguous().view(bsz, nk, nv, embed_dim)
        v = self.v_conv(v).squeeze(-1).transpose(1, 2).contiguous().view(bsz, nk, nv, embed_dim)
        
        k = self.k_proj(k)
        v = self.v_proj(v)
        
        # Keep heads, pixels, source times and target times on separate axes.
        q = rearrange(q, 'b nq (h d) -> b h nq d', h=self.num_heads)
        k = rearrange(k, 'b nk nv (h d) -> b h nv nk d', h=self.num_heads)
        v = rearrange(v, 'b nk nv (h d) -> b h nv nk d', h=self.num_heads)
        
        # For each pixel and target time, attend over source times. Eq. (2)
        # scales by the number of spatial positions (H*W), not H*W*T.
        attn_weights = torch.einsum('bhqd,bhpkd->bhpqk', q, k) / math.sqrt(nv)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Attention output
        attn_output = torch.einsum('bhpqk,bhpkd->bhpqd', attn_weights, v)
        # A direct view here mixed head and target-time axes in the old code.
        attn_output = attn_output.permute(0, 3, 2, 1, 4).contiguous()
        attn_output = attn_output.view(bsz, nq, nv, self.nhidden)
        attn_output = self.out_proj(attn_output)
        
        return attn_output

class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x):
        return self.net(x)

class TimeCrossAttention(nn.Module):
    def __init__(self, nhidden=512, embed_time=16, num_heads=1, ff_hidden_dim=2048, dropout=0.1):
        super(TimeCrossAttention, self).__init__()
        self.embed_dim = embed_time
        self.nhidden = nhidden
        self.num_heads = num_heads

        self.att = multiTimeAttention(embed_time, nhidden, num_heads)
        self.norm1 = nn.LayerNorm(embed_time)
        self.norm2 = nn.LayerNorm(embed_time)
        self.ff = FeedForward(embed_time, ff_hidden_dim, dropout)
    
    def time_embedding(self, pos):
        embed_dim = self.embed_dim
        # 初始化位置编码张量：(B, T, embed_dim)，后续用 sin/cos 填充
        pe = torch.zeros(
            pos.shape[0], pos.shape[1], embed_dim,
            device=pos.device, dtype=pos.dtype
        )
        # 将时间戳缩放到合适量级并扩维为 (B, T, 1)，供正弦/余弦编码使用
        position = 48. * pos.unsqueeze(2)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, device=pos.device, dtype=pos.dtype)
            * -(np.log(10.0) / embed_dim)
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
    
    def forward(self, x, query_time):
        b, t, c, h, w = x.shape
        if query_time.shape != (b, t):
            raise ValueError(f'query_time must have shape {(b, t)}, got {tuple(query_time.shape)}')
        source = rearrange(x, 'b t c h w -> b t (h w) c')
        
        query = self.time_embedding(query_time)

        attn_output = self.att(query, source, source)
        # Cross-attention produces target-time tokens. Its residual must come
        # from the target query, not source[i] paired with target[i].
        x = query.unsqueeze(2) + attn_output
        x = self.norm1(x)
        
        ff_output = self.ff(x)
        x = x + ff_output  
        x = self.norm2(x)
        x = x.permute(0,1,3,2).reshape(b, t, c, h, w)
        
        return x

