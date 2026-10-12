"""Fire overlay aligned with the paper supplement Figure 2.

The base satellite image stays unchanged except for a one-pixel red stroke
just outside the mask. Pixels inside the mask become one golden-orange colour
measured from the supplement figure, which is lighter than config fire_color
(255, 80, 0). There is no radiometric matching, alpha blend, texture
recolouring, or hotspot stamp.
"""

import cv2
import numpy as np

# RGB. Interior median of the flat fire fills in supplement Figure 2.
FIRE_COLOR_RGB = (189, 111, 10)
# RGB. One-pixel stroke sitting just outside that fill.
EDGE_COLOR_RGB = (191, 18, 10)


def load_base_bgr(path, size=(256, 256)):
    """Read a satellite base image and optionally resize it. No colour change."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"failed to read image: {path}")
    if size is not None:
        image = cv2.resize(image, tuple(size), interpolation=cv2.INTER_LINEAR)
    return image


def load_fire_mask(path, size=(256, 256)):
    """Read a JPEG fire mask as uint8 values 0 and 1."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"failed to read mask: {path}")
    if size is not None:
        mask = cv2.resize(mask, tuple(size), interpolation=cv2.INTER_NEAREST)
    threshold = float(mask.max()) / 2.0
    return (mask > threshold).astype(np.uint8)


def _bgr_uint8(color_rgb):
    if len(color_rgb) != 3:
        raise ValueError(f"colour must have 3 values, got {color_rgb}")
    red, green, blue = (int(channel) for channel in color_rgb)
    for channel in (red, green, blue):
        if channel < 0 or channel > 255:
            raise ValueError(f"colour channel out of range: {color_rgb}")
    return blue, green, red


def fuse_opaque_overlay(
    base_bgr,
    fire_mask,
    fire_color_rgb=FIRE_COLOR_RGB,
    edge_color_rgb=EDGE_COLOR_RGB,
):
    """Paste the Figure 2 fire colour onto a copy of the base image.

    Mask pixels become ``fire_color_rgb``. A one-pixel ring just outside the
    mask becomes ``edge_color_rgb``. Every other pixel stays equal to the base.
    """
    if base_bgr.ndim != 3 or base_bgr.shape[2] != 3:
        raise ValueError(f"base must be HxWx3, got {base_bgr.shape}")
    if fire_mask.shape != base_bgr.shape[:2]:
        raise ValueError(
            f"shape mismatch: base {base_bgr.shape[:2]} vs mask {fire_mask.shape}"
        )

    inside = fire_mask.astype(bool)
    dilated = cv2.dilate(inside.astype(np.uint8), np.ones((3, 3), np.uint8))
    ring = dilated.astype(bool) & ~inside

    fused = np.array(base_bgr, copy=True)
    fused[inside] = _bgr_uint8(fire_color_rgb)
    if edge_color_rgb is not None:
        fused[ring] = _bgr_uint8(edge_color_rgb)
    return fused


def fuse_mask_sequence(
    base_bgr,
    mask_paths,
    size=(256, 256),
    fire_color_rgb=FIRE_COLOR_RGB,
    edge_color_rgb=EDGE_COLOR_RGB,
):
    """Fuse one unchanged base image with each fire mask, in the given order."""
    frames = []
    masks = []
    for mask_path in mask_paths:
        mask = load_fire_mask(mask_path, size)
        frames.append(fuse_opaque_overlay(base_bgr, mask, fire_color_rgb, edge_color_rgb))
        masks.append(mask)
    return frames, masks
