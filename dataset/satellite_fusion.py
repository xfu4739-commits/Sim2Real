"""On-the-fly satellite fusion for SimFireDataset."""

from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch

_FUSE_PATH = Path(__file__).resolve().parents[1] / "test" / "fusion" / "fuse_satellite.py"
_spec = spec_from_file_location("fuse_satellite", _FUSE_PATH)
_fuse = module_from_spec(_spec)
_spec.loader.exec_module(_fuse)


@dataclass
class SatelliteFusionConfig:
    enabled: bool = True
    style: str = "burn-scar"
    match_radiometry: bool = True
    hotspots: bool = True
    alpha: float = 0.85
    fire_color_bgr: tuple = (0, 80, 255)


def parse_satellite_fusion_cfg(cfg):
    """Read dataset.satellite_fusion from config.yaml."""
    raw = cfg.get("satellite_fusion") or {}
    color = raw.get("fire_color", [255, 80, 0])
    if isinstance(color, str):
        r, g, b = [int(x.strip()) for x in color.split(",")]
    else:
        r, g, b = [int(c) for c in color]
    return SatelliteFusionConfig(
        enabled=bool(raw.get("enabled", True)),
        style=str(raw.get("style", "burn-scar")),
        match_radiometry=bool(raw.get("match_radiometry", True)),
        hotspots=bool(raw.get("hotspots", True)),
        alpha=float(raw.get("alpha", 0.85)),
        fire_color_bgr=(b, g, r),
    )


def load_scene_base_bgr(satellite_path, img_size, fusion_cfg):
    """Load and optionally radiometrically normalise the per-scene base map."""
    base = _fuse.load_rgb(satellite_path, tuple(img_size))
    if fusion_cfg.match_radiometry:
        base = _fuse.match_radiometry(base)
    return base


def fuse_satellite_sequence(base_bgr, mask_paths, img_size, fusion_cfg, sample_seed=0):
    """
    Fuse one base satellite image with a list of fire masks.

    mask_paths must follow the same order as the input fire frames. Returns a tensor
    shaped (T, 3, H, W) in [0, 1], matching process_image(gray=False).
    """
    prev_mask = None
    frames = []
    for step, mask_path in enumerate(mask_paths):
        mask = _fuse.load_fire_mask(mask_path, tuple(img_size))
        fused = _fuse.apply_style(
            base_bgr,
            mask,
            style=fusion_cfg.style,
            alpha=fusion_cfg.alpha,
            fire_color_bgr=fusion_cfg.fire_color_bgr,
            hotspots=fusion_cfg.hotspots,
            prev_mask=prev_mask,
            seed=int(sample_seed) + step,
        )
        prev_mask = mask
        frame = torch.from_numpy(fused).permute(2, 0, 1).float() / 255.0
        frames.append(frame)
    return torch.stack(frames, dim=0)
