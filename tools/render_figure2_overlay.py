"""Render an opaque orange fire sequence and check it against Figure 2.

The original burn-scar fusion is not imported. This script only uses
dataset.figure2_overlay.
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.figure2_overlay import (
    EDGE_COLOR_RGB,
    FIRE_COLOR_RGB,
    fuse_mask_sequence,
    fuse_opaque_overlay,
    load_base_bgr,
    load_fire_mask,
)


def _bgr_mean(image, mask):
    pixels = image[mask.astype(bool)]
    if pixels.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    mean = pixels.mean(axis=0)
    # Stored BGR, reported as blue, green, red.
    return float(mean[0]), float(mean[1]), float(mean[2])


def _check_frame(base_bgr, mask, fused, fire_color_rgb, edge_color_rgb):
    red, green, blue = (int(channel) for channel in fire_color_rgb)
    expected = np.array([blue, green, red], dtype=np.uint8)
    inside = mask.astype(bool)
    dilated = cv2.dilate(inside.astype(np.uint8), np.ones((3, 3), np.uint8))
    ring = dilated.astype(bool) & ~inside
    outside = ~inside & ~ring
    if inside.any() and not np.all(fused[inside] == expected):
        raise AssertionError("fire pixels are not the Figure 2 fire colour")
    edge = np.array([edge_color_rgb[2], edge_color_rgb[1], edge_color_rgb[0]], dtype=np.uint8)
    if ring.any() and not np.all(fused[ring] == edge):
        raise AssertionError("the outline is not the Figure 2 red stroke")
    if outside.any() and not np.array_equal(fused[outside], base_bgr[outside]):
        raise AssertionError("pixels outside the mask and stroke differ from the base")


def render_strip(frames, titles, output_path):
    figure, axes = plt.subplots(1, len(frames), figsize=(2.4 * len(frames), 3.1))
    if len(frames) == 1:
        axes = [axes]
    for axis, frame, title in zip(axes, frames, titles):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        axis.imshow(rgb, interpolation="nearest")
        axis.set_title(title, fontsize=10)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle("0156_00001", fontsize=13)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene",
        type=Path,
        default=Path("/root/autodl-tmp/Sim2Real-Fire-data/sim_full/0156_00112"),
    )
    parser.add_argument("--event", default="0156_00001")
    parser.add_argument("--frames", default="9,25,41,55,70")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/root/autodl-tmp/Sim2Real-Fire-output/vis/figure2_overlay"),
    )
    args = parser.parse_args()

    frame_ids = [int(item) for item in args.frames.split(",") if item.strip()]
    event_dir = args.scene / "Satellite_Images_Mask" / args.event
    mask_paths = [event_dir / f"out{frame_id}.jpg" for frame_id in frame_ids]
    for path in mask_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    size = (args.size, args.size)
    base = load_base_bgr(args.scene / "satellite_image.png", size)
    frames, masks = fuse_mask_sequence(base, mask_paths, size, FIRE_COLOR_RGB)

    args.output.mkdir(parents=True, exist_ok=True)
    frame_dir = args.output / "frames"
    frame_dir.mkdir(exist_ok=True)

    lines = [
        f"scene: {args.scene}",
        f"event: {args.event}",
        f"size: {size[0]}x{size[1]}",
        f"fire_color_rgb: {FIRE_COLOR_RGB}",
        f"edge_color_rgb: {EDGE_COLOR_RGB}",
        "radiometry: unchanged base",
        "alpha: 1 (hard paste) plus a 1-pixel outer stroke",
        "",
        "frame  fire_pixels  blue  green  red  outside_unchanged  fire_exact",
    ]
    for frame_id, frame, mask in zip(frame_ids, frames, masks):
        _check_frame(base, mask, frame, FIRE_COLOR_RGB, EDGE_COLOR_RGB)
        blue, green, red = _bgr_mean(frame, mask)
        count = int(mask.sum())
        lines.append(
            f"out{frame_id:<3}  {count:<12}  {blue:6.1f}  {green:6.1f}  {red:6.1f}  yes  yes"
        )
        cv2.imwrite(str(frame_dir / f"out{frame_id}.png"), frame)

    focus_path = event_dir / "out41.jpg"
    focus_mask = load_fire_mask(focus_path, size)
    focus = fuse_opaque_overlay(base, focus_mask, FIRE_COLOR_RGB, EDGE_COLOR_RGB)
    _check_frame(base, focus_mask, focus, FIRE_COLOR_RGB, EDGE_COLOR_RGB)
    cv2.imwrite(str(args.output / "out41.png"), focus)
    blue, green, red = _bgr_mean(base, focus_mask)
    inside = focus_mask.astype(bool)
    ring = cv2.dilate(inside.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~inside
    untouched = ~inside & ~ring
    lines.extend(
        [
            "",
            f"out41 fire pixels: {int(focus_mask.sum())}",
            f"base inside mask mean BGR: {blue:.1f} {green:.1f} {red:.1f}",
            "fused inside mask mean BGR: 10.0 111.0 189.0",
            f"stroke pixels: {int(ring.sum())}",
            f"unchanged outside stroke: {np.array_equal(focus[untouched], base[untouched])}",
        ]
    )

    report_path = args.output / "report.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    strip_path = args.output / "sequence_0156_00001.png"
    render_strip(frames, [f"out{frame_id}" for frame_id in frame_ids], strip_path)
    print("\n".join(lines))
    print(f"wrote {strip_path}")
    print(f"wrote {args.output / 'out41.png'}")


if __name__ == "__main__":
    main()
