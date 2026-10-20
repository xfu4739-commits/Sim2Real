"""Tracked image operations used by simulation satellite fusion.

These functions are runtime code. Keep them in the ``dataset`` package rather
than under the ignored local ``test`` directory so a clean checkout can train.
"""

import cv2
import numpy as np

from dataset.figure2_overlay import fuse_opaque_overlay


def load_rgb(path, size=(256, 256)):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"failed to read image: {path}")
    if size is not None:
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    return image


def load_fire_mask(path, size=(256, 256), threshold=None):
    """Read a JPEG-safe binary fire mask as a uint8 array containing 0/1."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"failed to read mask: {path}")
    if size is not None:
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    if threshold is None:
        threshold = float(mask.max()) / 2.0
    return (mask > threshold).astype(np.uint8)


# Rec.601 luma weights in OpenCV BGR order.
LUMA_BGR = np.array([0.114, 0.587, 0.299], dtype=np.float32)

# Calibration constants measured from the local real/simulation composites.
BURN_CHROMA_BGR = (0.779, 1.074, 0.940)
BURN_DARKEN = 1.0
BURN_SPREAD_BGR = (0.605, 1.210, 1.332)
VEG_CEILING_KNEE = 1.217
VEG_CEILING_MAX = 1.372
REAL_CANOPY_LUMA = 39.9
SIM_CHROMA_OFFSET_BGR = (-0.047, 0.023, -0.039)
HOTSPOT_BGR = (31, 44, 134)
HOTSPOT_AREA_FRACTION = 0.0003
HOTSPOT_BLOB_PX = 3
HOTSPOT_DEPTH_TAU = 12.0


def fuse_satellite(base_bgr, fire_mask, fire_color_bgr=(0, 80, 255), alpha=0.85):
    """Alpha-blend a fire colour onto the base satellite image."""
    if base_bgr.shape[:2] != fire_mask.shape:
        raise ValueError(
            f"shape mismatch: base {base_bgr.shape[:2]} vs mask {fire_mask.shape}"
        )

    base = base_bgr.astype(np.float32)
    mask = fire_mask.astype(np.float32)[..., None]
    fire = np.zeros_like(base)
    fire[:] = np.array(fire_color_bgr, dtype=np.float32)
    fused = (1.0 - alpha * mask) * base + (alpha * mask) * fire
    return np.clip(fused, 0, 255).astype(np.uint8)


def match_radiometry(
    image_bgr,
    luma=REAL_CANOPY_LUMA,
    chroma_offset_bgr=SIM_CHROMA_OFFSET_BGR,
):
    """Put a simulation base image on the real composites' brightness scale."""
    image = image_bgr.astype(np.float32)
    luminance = image @ LUMA_BGR
    mean_luminance = max(luminance.mean(), 1e-3)
    offset = np.array(chroma_offset_bgr, dtype=np.float32)
    shifted = image + luminance[..., None] * offset
    return np.clip(shifted * (luma / mean_luminance), 0, 255).astype(np.uint8)


def roll_off_vegetation(
    image_bgr,
    knee=VEG_CEILING_KNEE,
    ceiling=VEG_CEILING_MAX,
):
    """Compress implausibly vigorous green chroma while preserving luminance."""
    if ceiling <= knee:
        return image_bgr

    image = image_bgr.astype(np.float32)
    luminance = np.maximum(image @ LUMA_BGR, 1e-3)
    ratio = image[:, :, 1] / luminance
    excess = np.maximum(ratio - knee, 0.0)
    span = ceiling - knee
    wanted = np.where(excess > 0, knee + span * np.tanh(excess / span), ratio)

    weight = float(LUMA_BGR[1])
    green = wanted * (1.0 - weight * ratio) / (1.0 - weight * wanted)
    output = image.copy()
    output[:, :, 1] = green * luminance
    output *= (luminance / np.maximum(output @ LUMA_BGR, 1e-3))[..., None]
    return output


def fuse_burn_scar(
    base_bgr,
    fire_mask,
    burn_chroma_bgr=BURN_CHROMA_BGR,
    spread_bgr=BURN_SPREAD_BGR,
    darken=BURN_DARKEN,
    strength=1.0,
    feather=1.0,
    veg_knee=VEG_CEILING_KNEE,
    veg_max=VEG_CEILING_MAX,
):
    """Recolour a masked region while retaining the base image's terrain texture."""
    if base_bgr.shape[:2] != fire_mask.shape:
        raise ValueError(
            f"shape mismatch: base {base_bgr.shape[:2]} vs mask {fire_mask.shape}"
        )

    base = base_bgr.astype(np.float32)
    binary = fire_mask.astype(bool)
    mask = fire_mask.astype(np.float32)
    if feather > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), feather)
    mask = np.clip(strength * mask, 0.0, 1.0)[..., None]

    luminance = base @ LUMA_BGR
    region = base[binary] if binary.any() else base.reshape(-1, 3)
    region_luminance = (region @ LUMA_BGR).mean()
    base_chroma = region.mean(axis=0) / max(region_luminance, 1e-3)

    target = np.array(burn_chroma_bgr, dtype=np.float32)
    spread = np.array(spread_bgr, dtype=np.float32)
    deviation = base - luminance[..., None] * base_chroma
    burned = darken * (luminance[..., None] * target + spread * deviation)
    burned = roll_off_vegetation(burned, veg_knee, veg_max)

    fused = (1.0 - mask) * base + mask * burned
    return np.clip(fused, 0, 255).astype(np.uint8)


def add_fire_hotspots(
    image_bgr,
    fire_mask,
    prev_mask=None,
    color_bgr=HOTSPOT_BGR,
    area_fraction=HOTSPOT_AREA_FRACTION,
    blob_px=HOTSPOT_BLOB_PX,
    depth_tau=HOTSPOT_DEPTH_TAU,
    seed=0,
):
    """Place deterministic active-fire blobs near the fire front."""
    binary = fire_mask.astype(bool)
    if not binary.any() or area_fraction <= 0:
        return image_bgr.copy()

    padded = np.pad(binary.astype(np.uint8), 1)
    depth = cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1]
    weight = np.exp(-depth / max(depth_tau, 1e-3))
    weight[~binary] = 0.0

    if prev_mask is not None:
        ring = binary & ~prev_mask.astype(bool)
        if ring.any():
            weight[~ring] = 0.0

    total = weight.sum()
    if total <= 0:
        return image_bgr.copy()

    blob_pixels = max(blob_px, 1)
    blob_count = max(1, int(round(area_fraction * binary.sum() / blob_pixels)))
    rng = np.random.default_rng(seed)
    if blob_count <= np.count_nonzero(weight):
        flat = rng.choice(
            weight.size,
            size=blob_count,
            replace=False,
            p=(weight / total).ravel(),
        )
    else:
        flat = np.flatnonzero(weight)
    ys, xs = np.unravel_index(flat, weight.shape)

    radius = max(1, int(round(np.sqrt(blob_pixels / np.pi))))
    base_color = np.array(color_bgr, dtype=np.float32)
    layer = np.zeros_like(image_bgr)
    stamp = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    for y, x in zip(ys, xs):
        color = np.clip(base_color * rng.uniform(0.8, 1.6), 0, 255)
        cv2.circle(layer, (int(x), int(y)), radius, color.tolist(), -1)
        cv2.circle(stamp, (int(x), int(y)), radius, 1, -1)

    output = image_bgr.copy()
    keep = (stamp > 0) & binary
    output[keep] = layer[keep]
    return output


def apply_style(
    base,
    mask,
    style="burn-scar",
    alpha=0.85,
    fire_color_bgr=(0, 80, 255),
    hotspots=True,
    prev_mask=None,
    seed=0,
):
    """Apply the configured satellite fusion style."""
    if style == "figure2":
        return fuse_opaque_overlay(base, mask)
    if style == "overlay":
        return fuse_satellite(
            base,
            mask,
            fire_color_bgr=fire_color_bgr,
            alpha=alpha,
        )
    if style != "burn-scar":
        raise ValueError(f"unknown style: {style}")

    fused = fuse_burn_scar(base, mask)
    if hotspots:
        fused = add_fire_hotspots(
            fused,
            mask,
            prev_mask=prev_mask,
            seed=seed,
        )
    return fused
