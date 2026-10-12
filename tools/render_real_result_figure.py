"""Fuse one real-test sequence onto that scene's own satellite frames.

Masks come from the saved real-test outputs. Each date uses the satellite
image stored beside that test event. No colour remapping is applied.
"""

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.figure2_overlay import fuse_opaque_overlay, load_base_bgr


CASES = (
    {
        "scene": "0274_00001",
        "sample_index": 53,
        "task": "Backtrack",
        "target_name": "Backtrack",
        "dates": [
            "20201003",
            "20201001",
            "20200928",
            "20200920",
            "20200918",
            "20200916",
        ],
        "vis_dir": Path(
            "/root/autodl-tmp/Sim2Real-Fire-output/vis/sim_day_20261002_232142_064274_best_iou"
        ),
        "output": Path(
            "/root/autodl-tmp/Sim2Real-Fire-output/vis/figure2_overlay/test_0274_00001_backtrack.png"
        ),
    },
    {
        "scene": "0333_00001",
        "sample_index": 74,
        "task": "Forecast",
        "target_name": "Forecast",
        "dates": [
            "20180801",
            "20180803",
            "20180806",
            "20180808",
            "20180811",
            "20180813",
        ],
        "vis_dir": Path("/root/autodl-tmp/Sim2Real-Fire-output/vis/sim_day_forecast_best_iou"),
        "output": Path(
            "/root/autodl-tmp/Sim2Real-Fire-output/vis/figure2_overlay/test_0333_00001_forecast.png"
        ),
    },
)


def load_binary(path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return (mask > 0).astype(np.uint8)


def frame_iou(prediction, target):
    intersection = np.logical_and(prediction, target).sum()
    union = np.logical_or(prediction, target).sum()
    if union == 0:
        return 1.0
    return float(intersection) / float(union)


def render_case(case):
    scene = case["scene"]
    sample_index = case["sample_index"]
    dates = case["dates"]
    vis_dir = case["vis_dir"]
    sat_dir = Path(
        f"/root/autodl-tmp/Sim2Real-Fire-data/real_full/{scene}/Satellite_Images/{scene}"
    )
    gt_frames = []
    ours_frames = []
    ious = []
    for step, date in enumerate(dates):
        base = load_base_bgr(sat_dir / f"{date}.png", (256, 256))
        gt = load_binary(vis_dir / f"{sample_index}_gt_{step + 1}.png")
        gt_frames.append(fuse_opaque_overlay(base, gt))
        if step < 3:
            ours_frames.append(fuse_opaque_overlay(base, gt))
        else:
            prediction = load_binary(vis_dir / f"{sample_index}_pd_{step + 1}.png")
            ours_frames.append(fuse_opaque_overlay(base, prediction))
            ious.append(frame_iou(prediction, gt))

    figure = plt.figure(figsize=(13.6, 5.8), facecolor="white")
    grid = figure.add_gridspec(
        2,
        7,
        width_ratios=[1, 1, 1, 0.16, 1, 1, 1],
        left=0.02,
        right=0.93,
        top=0.76,
        bottom=0.10,
        wspace=0.06,
        hspace=0.18,
    )
    columns = [0, 1, 2, 4, 5, 6]
    date_labels = [f"{date[:4]}/{date[4:6]}/{date[6:]}" for date in dates]
    captions = ["Input", "Input", "Input"] + [f"IOU:{value:.4f}" for value in ious]

    for row, frames in enumerate((gt_frames, ours_frames)):
        for column, frame, caption in zip(columns, frames, captions):
            axis = figure.add_subplot(grid[row, column])
            axis.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_color("#222222")
                spine.set_linewidth(0.6)
            if row == 1:
                axis.set_xlabel(caption, fontsize=8, labelpad=3)

    figure.text(0.955, 0.62, "GT", fontsize=12, ha="left", va="center")
    figure.text(0.955, 0.30, "Ours", fontsize=12, ha="left", va="center")
    figure.text(0.28, 0.955, "Input", fontsize=11, ha="center", va="center")
    figure.text(0.70, 0.955, case["target_name"], fontsize=11, ha="center", va="center")
    figure.text(
        0.50,
        0.995,
        f"Test  {scene}  {case['task']}",
        fontsize=12,
        ha="center",
        va="top",
    )

    timeline = figure.add_axes([0.055, 0.80, 0.84, 0.10])
    positions = [0, 1, 2, 3.35, 4.35, 5.35]
    timeline.plot([positions[0], positions[-1]], [0, 0], color="black", lw=1.1, zorder=1)
    timeline.scatter(positions, [0] * 6, s=26, c="black", zorder=2)
    for position, label in zip(positions, date_labels):
        timeline.text(position, -0.55, label, ha="center", va="top", fontsize=8)
    timeline.set_xlim(-0.45, 5.8)
    timeline.set_ylim(-1.6, 0.8)
    timeline.axis("off")

    case["output"].parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(case["output"], dpi=160, facecolor="white")
    plt.close(figure)
    print(case["task"], scene, "sample", sample_index)
    print("dates", date_labels)
    print("target_iou", [round(value, 4) for value in ious])
    print("wrote", case["output"])


def main():
    for case in CASES:
        render_case(case)


if __name__ == "__main__":
    main()
