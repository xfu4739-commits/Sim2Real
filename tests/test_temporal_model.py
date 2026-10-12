import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from model.S2RFireTr import FireTr
from model.SwinTransformerDecoder import boundary_source_skip
from model.TimeCrossAttention import TimeCrossAttention


class TinyEncoder(nn.Module):
    """Keep the FireTr wiring test independent of its large Swin backbone."""

    def __init__(self, input_channel, hidden_dim, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(input_channel, hidden_dim * 4, 1)

    def forward(self, images):
        encoded = F.avg_pool2d(self.conv(images), 2)
        return encoded, [encoded, encoded, encoded]


class TinyDecoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x, res):
        batch, steps, channels, height, width = x.shape
        assert res[0].shape[0] == batch * steps
        return x[:, :, 0]


class TemporalModelTest(unittest.TestCase):
    def test_time_attention_preserves_head_and_target_axes(self):
        torch.manual_seed(7)
        module = TimeCrossAttention(
            nhidden=8, embed_time=8, num_heads=2, ff_hidden_dim=16
        ).eval()
        repeated = torch.randn(1, 1, 8, 2, 2).expand(1, 3, 8, 2, 2).clone()
        times = torch.tensor([[0.1, 0.4, 0.9]])
        with torch.no_grad():
            source = repeated.permute(0, 1, 3, 4, 2).reshape(1, 3, 4, 8)
            attention = module.att(module.time_embedding(times), source, source)
            result = module(repeated, times)
            changed_times = module(repeated, torch.tensor([[0.2, 0.5, 0.8]]))
        # Equal source values give equal raw attention across target times;
        # the target-query residual still makes the target times distinct.
        self.assertTrue(torch.allclose(attention[:, 0], attention[:, 2], atol=1e-6))
        self.assertGreater((result[:, 0] - result[:, 2]).abs().max().item(), 1e-5)
        self.assertGreater((result - changed_times).abs().max().item(), 1e-5)

        varying = torch.randn(1, 3, 8, 2, 2)
        with torch.no_grad():
            earlier = module(varying, times)
            later = module(varying, torch.tensor([[0.2, 0.5, 0.8]]))
        self.assertGreater((earlier - later).abs().max().item(), 1e-5)

    def test_decoder_skips_use_last_observed_frame_for_every_target(self):
        # Forecast source [1,2,3] and backtrack source [6,5,4] both end at
        # the observation directly adjacent to their first target frame.
        skip = torch.tensor([1.0, 2.0, 3.0, 6.0, 5.0, 4.0]).view(6, 1, 1, 1)
        aligned = boundary_source_skip(skip, batch_size=2, source_steps=3, target_steps=3)
        self.assertEqual(aligned.flatten().tolist(), [3.0, 3.0, 3.0, 4.0, 4.0, 4.0])

    def test_firetr_keeps_source_time_axis_through_decoder(self):
        cfg = {
            'input_length': 3,
            'img_size': [8, 8],
            'Encoder': {
                'hidden_dim': 2,
                'downscaling_factors': [2, 2, 2],
                'layers': [2, 2, 2],
                'heads': [2, 2, 2],
                'head_dim': 4,
                'window_size': 2,
                'relative_pos_embedding': False,
                'modal_channel': 23,
            },
            'Spatial': {'embed_dim': 8, 'num_heads': 2, 'hidden_dim': 16},
            'Temporal': {'embed_dim': 8, 'num_heads': 2, 'hidden_dim': 16},
            'Time': {'embed_dim': 8, 'num_heads': 2, 'hidden_dim': 16},
        }
        torch.manual_seed(11)
        with patch('model.S2RFireTr.Encoder', TinyEncoder), patch(
            'model.S2RFireTr.Decoder', TinyDecoder
        ):
            model = FireTr(cfg).eval()

        fire = torch.zeros(1, 3, 8, 8)
        fire[:, 0, 2:4, 2:4] = 1
        fire[:, 1, 1:5, 1:5] = 1
        fire[:, 2, 1:6, 1:6] = 1
        fuel = torch.zeros(1, 10, 8, 8)
        vegetation = torch.zeros(1, 6, 8, 8)
        topography = torch.zeros(1, 4, 8, 8)
        satellite = torch.randn(1, 3, 3, 8, 8)
        weather = torch.randn(1, 3, 10)
        times = torch.tensor([[0.1, 0.4, 0.9]])
        with torch.no_grad():
            output = model(fire, fuel, vegetation, topography, satellite, weather, times)
            changed = model(
                fire, fuel, vegetation, topography, satellite, weather,
                torch.tensor([[0.2, 0.5, 0.8]]),
            )
        self.assertEqual(output.shape, (1, 3, 4, 4))
        self.assertGreater((output[:, 0] - output[:, 2]).abs().max().item(), 1e-5)
        self.assertGreater((output - changed).abs().max().item(), 1e-5)


if __name__ == '__main__':
    unittest.main()
