from torch import nn
from einops import rearrange
from .SwinTransformer import StageModule, StageModule_up, StageModule_up_final

class Encoder(nn.Module):
    def __init__(self, input_channel, hidden_dim, downscaling_factors, layers, heads, head_dim, window_size, relative_pos_embedding):
        super(Encoder, self).__init__()        
        # 
        # self.stage1 这里初始化了编码器的第一个stage（StageModule），主要用于对输入images进行低层次特征的提取。
        # 具体参数说明如下：
        #   in_channels=input_channel      # 输入通道数，通常为原始图像的通道数，如1（灰度）或3（RGB）
        #   hidden_dimension=hidden_dim    # 该stage输出特征的维度，也是当前stage block内部的通道数
        #   layers=layers[0]               # 当前stage包含的Swin Block数量（必须为偶数，因为有normal和shifted block交替堆叠）
        #   downscaling_factor=downscaling_factors[0]
        #                                   # 下采样倍率（Patch Merging缩小空间分辨率时的步长/比例）
        #   num_heads=heads[0]             # 多头自注意力中的头数
        #   head_dim=head_dim              # 每个注意力头的特征维度
        #   window_size=window_size        # SwinTransformer block中window self-attention的窗口大小
        #   relative_pos_embedding=relative_pos_embedding
        #                                   # 是否使用相对位置编码
        #   h_w=[64, 64]                   # 输入图片的空间尺寸[height, width]，用于某些模块计算
        # 该模块的主要作用：对原始输入图片逐层提取多尺度特征，为下一stage做准备。
        self.stage1 = StageModule(
            in_channels=input_channel,
            hidden_dimension=hidden_dim,
            layers=layers[0],
            downscaling_factor=downscaling_factors[0],
            num_heads=heads[0],
            head_dim=head_dim,
            window_size=window_size,
            relative_pos_embedding=relative_pos_embedding,
            h_w=[64, 64]
        )
              

        self.stage2 = StageModule(in_channels=hidden_dim, hidden_dimension=hidden_dim * 2, layers=layers[1],
                                  downscaling_factor=downscaling_factors[1], num_heads=heads[1], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding, h_w=[32, 32])

        self.stage3 = StageModule(in_channels=hidden_dim * 2, hidden_dimension=hidden_dim * 4, layers=layers[2],
                                  downscaling_factor=downscaling_factors[2], num_heads=heads[2], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding, h_w=[16, 16])


    def forward(self, images): 
        x1 = self.stage1(images)
        x2 = self.stage2(x1)    
        x3 = self.stage3(x2) 
        return x3, [x1, x2, x3]
    
