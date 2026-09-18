import torch
import torch.nn as nn
from einops import rearrange, repeat

from model.SwinTransformerEncoder import Encoder
from model.SpatialCrossAttention import SpatialCrossAttention
from model.TemporalCrossAttention import TemporalCrossAttention
from model.TimeCrossAttention import TimeCrossAttention
from model.SwinTransformerDecoder import Decoder

class FireTr(nn.Module):
    def __init__(self, cfg):
        super(FireTr, self).__init__()
        self.input_length = cfg['input_length']
        self.H, self.W = cfg['img_size']
        self.hidden_dim = cfg['Encoder']['hidden_dim']
        self.downscaling_factors = cfg['Encoder']['downscaling_factors']
        self.layers = cfg['Encoder']['layers']
        self.heads = cfg['Encoder']['heads']
        self.head_dim = cfg['Encoder']['head_dim']
        self.window_size = cfg['Encoder']['window_size']
        self.relative_pos_embedding = cfg['Encoder']['relative_pos_embedding']
        self.modal_channel = cfg['Encoder']['modal_channel']

        self.spatial_embed_dim = cfg['Spatial']['embed_dim']
        self.spatial_num_heads = cfg['Spatial']['num_heads']
        self.spatial_hidden_dim = cfg['Spatial']['hidden_dim']

        self.temporal_embed_dim = cfg['Temporal']['embed_dim']
        self.temporal_num_heads = cfg['Temporal']['num_heads']
        self.temporal_hidden_dim = cfg['Temporal']['hidden_dim']

        self.time_embed_dim = cfg['Time']['embed_dim']
        self.time_num_heads = cfg['Time']['num_heads']
        self.time_hidden_dim = cfg['Time']['hidden_dim']


        self.sequence_encoder = Encoder(input_channel = self.input_length, hidden_dim = self.hidden_dim, downscaling_factors = self.downscaling_factors, layers = self.layers, heads = self.heads, head_dim = self.head_dim, window_size = self.window_size, relative_pos_embedding = self.relative_pos_embedding)

        self.spatial_encoder = Encoder(input_channel = self.modal_channel, hidden_dim = self.hidden_dim, downscaling_factors = self.downscaling_factors, layers = self.layers, heads = self.heads, head_dim = self.head_dim, window_size = self.window_size, relative_pos_embedding = self.relative_pos_embedding)

        self.satellite_encoder = Encoder(input_channel = self.input_length, hidden_dim = self.hidden_dim, downscaling_factors = self.downscaling_factors, layers = self.layers, heads = self.heads, head_dim = self.head_dim, window_size = self.window_size, relative_pos_embedding = self.relative_pos_embedding)

        self.spatial_crossattention = SpatialCrossAttention(embed_dim = self.spatial_embed_dim, num_heads = self.spatial_num_heads, ff_hidden_dim = self.spatial_hidden_dim)
        self.temporal_crossattention = TemporalCrossAttention(input_dim = self.input_length, embed_dim=self.temporal_embed_dim, num_heads=self.temporal_num_heads, ff_hidden_dim=self.temporal_hidden_dim, encoder_hidden_dim=self.temporal_hidden_dim)

        self.time_crossattention = TimeCrossAttention(nhidden=self.time_hidden_dim, embed_time=self.time_embed_dim, num_heads=self.time_num_heads)

        self.decoder = Decoder(input_channel = self.input_length, hidden_dim = self.hidden_dim, downscaling_factors = self.downscaling_factors, layers = self.layers, heads = self.heads, head_dim = self.head_dim, window_size = self.window_size, relative_pos_embedding = self.relative_pos_embedding)

    def forward(self, input_sequence, fuel, vegetation, topography, satellite_images, weather_data, timestamps):
        '''
        input_sequence: (B, 3, 256, 256) 输入火场序列各时刻的图像
        fuel: (B, 10, 256, 256) 燃料特征
        vegetation: (B, 6, 256, 256) 植被特征
        topography: (B, 3, 256, 256) 地形特征
        satellite_images: (B, 3, 3, 256, 256) 卫星图像
        weather_data: (B, 3, 10) 年/月/日/时刻 + 6 维气象特征
        timestamps: (B, 3) 时间戳
        '''
        B, T, H, W = input_sequence.shape  # (B, 3, 256, 256)
        if satellite_images.shape != (B, T, 3, H, W):
            raise ValueError(
                "satellite_images must have shape "
                f"{(B, T, 3, H, W)}, got {tuple(satellite_images.shape)}"
            )
        # 1. 对输入火场序列进行编码，两路交叉注意力共用这个 Query
        input_sequence_encoder, input_sequence_encoder_list = self.sequence_encoder(input_sequence)
        fire_query = repeat(input_sequence_encoder, 'b c h w -> b t c h w', t=T)

        # 2. 论文 Eq. (1) 的空间 K/V 使用 [P, G, F, I]。数据加载器输出
        #    fuel=10, vegetation=6, topography=3, satellite=T*3=9，共 28 通道。
        spatial_satellite_images = satellite_images.reshape(B, T * 3, H, W)
        spatial_information = torch.cat(
            [fuel, vegetation, topography, spatial_satellite_images], dim=1
        )
        if spatial_information.shape[1] != self.modal_channel:
            raise ValueError(
                "spatial modality channel mismatch: "
                f"config expects {self.modal_channel}, dataset produced "
                f"{spatial_information.shape[1]} "
                f"(fuel={fuel.shape[1]}, vegetation={vegetation.shape[1]}, "
                f"topography={topography.shape[1]}, satellite={T * 3})"
            )
        spatial_information_encoder, _ = self.spatial_encoder(spatial_information)
        # 3. 空间交叉注意力：Q=火区，K/V=静态环境
        spatial = self.spatial_crossattention(input_sequence_encoder, spatial_information_encoder)
        spatial = repeat(spatial, 'b c h w -> b t c h w', t=T)

        temporal_satellite_images = satellite_images.reshape(-1, 3, self.H, self.W)
        # 4. 对卫星图像按时刻编码
        satellite_images_encoder, _ = self.satellite_encoder(temporal_satellite_images)
        satellite_images_encoder = rearrange(satellite_images_encoder, '(b t) c h w -> b t c h w', b=B, t=T)

        # 5. 时序交叉注意力：Q=火区，K/V=[卫星编码, 气象编码]
        temporal = self.temporal_crossattention(fire_query, satellite_images_encoder, weather_data)
        # 6. 将空间和时间特征融合
        area_representation = spatial + temporal

        # 7. 对区域表示和时间戳进行交叉注意力
        target_representation = self.time_crossattention(area_representation, timestamps)

        # 8. 解码器
        target = self.decoder(target_representation, input_sequence_encoder_list)

        return target


