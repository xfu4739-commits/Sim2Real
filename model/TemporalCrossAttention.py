import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None):
        bsz, tgt_len, embed_dim = q.size()
        src_len = k.size(1)

        q = self.q_proj(q).view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if attn_mask is not None:
            attn_weights = attn_weights.masked_fill(attn_mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output

class WxEncoder(nn.Module):
    def __init__(self, input_dim, num_heads=4, out_dim=768, hidden_dim=128):
        super(WxEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.node_feature = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1),
            nn.ELU()
        )
        self.node_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=0.1, batch_first=True)
        self.sublayer = nn.Sequential(nn.Linear(hidden_dim, out_dim), nn.ELU())

    def forward(self, inputs, mask):
        mask = mask.to(torch.float32).squeeze(-1)
        nodes = self.node_feature(inputs.permute(0, 2, 1)).permute(0, 2, 1)
        attn_output, _ = self.node_attention(nodes, nodes, nodes, key_padding_mask=~mask.bool())
        feature = self.sublayer(attn_output)
        
        return feature

class TemporalCrossAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, ff_hidden_dim, encoder_hidden_dim):
        super(TemporalCrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.wx_encoder = WxEncoder(input_dim=6, num_heads=num_heads, out_dim=embed_dim, hidden_dim=encoder_hidden_dim)

        self.q_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        # K/V come from concat([SwinEnc(I), MHA(W)]), so the fused token dim is 2C
        self.k_proj = nn.Linear(embed_dim * 2, embed_dim)
        self.v_proj = nn.Linear(embed_dim * 2, embed_dim)

        self.mha = MultiheadAttention(embed_dim, num_heads)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_hidden_dim)

    def _flatten_tokens(self, x):
        b, t, c, h, w = x.shape
        return x.reshape(b * t, c, h * w).transpose(1, 2).reshape(b, t * h * w, c)

    def forward(self, fire_features, satellite_features, weather_data):
        """
        fire_features:      (B, T, C, H, W)  Query = SwinEnc(S)
        satellite_features: (B, T, C, H, W)  SwinEnc(I)
        weather_data:       (B, T, N)        last 6 columns are meteorological features
        """
        b, t, c, h, w = fire_features.shape
        if satellite_features.shape != fire_features.shape:
            raise ValueError(
                "fire_features and satellite_features must have the same shape, "
                f"got {tuple(fire_features.shape)} and {tuple(satellite_features.shape)}"
            )
        if weather_data.shape[:2] != (b, t) or weather_data.shape[-1] < 6:
            raise ValueError(
                f"weather_data must have shape (B, T, N>=6), got {tuple(weather_data.shape)}"
            )
        mask = torch.ones(b, t, device=fire_features.device)
        weather_features = self.wx_encoder(weather_data[..., -6:], mask)  # (B, T, C)

        q = self.q_proj(fire_features.reshape(b * t, c, h, w))
        q = q.reshape(b * t, c, h * w).transpose(1, 2).reshape(b, t * h * w, c)

        weather_map = weather_features.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, h, w)
        kv_features = torch.cat([satellite_features, weather_map], dim=2)  # (B, T, 2C, H, W)
        kv_tokens = self._flatten_tokens(kv_features)  # (B, T*H*W, 2C)
        k = self.k_proj(kv_tokens)
        v = self.v_proj(kv_tokens)

        attn_output = self.mha(q, k, v)
        attn_output = attn_output.view(b, t, h, w, c).permute(0, 1, 4, 2, 3)
        attn_output = attn_output + fire_features

        attn_output = self.norm1(attn_output.view(b * t, c, h * w).transpose(1, 2)).transpose(1, 2).view(b, t, c, h, w)

        ff_output = self.ff(attn_output.view(b * t, c, h * w).transpose(1, 2)).transpose(1, 2).view(b, t, c, h, w)

        output = self.norm2(ff_output.view(b * t, c, h * w).transpose(1, 2)).transpose(1, 2).view(b, t, c, h, w)
        output = output + attn_output

        return output
