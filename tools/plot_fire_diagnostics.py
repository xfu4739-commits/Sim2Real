#!/usr/bin/env python3
"""Evaluate a checkpoint and plot fire-mask diagnostics without changing training.

Run from the repository root, for example:
python tools/plot_fire_diagnostics.py --config config/config_forecast.yaml \
    --checkpoint /path/to/best_iou.pth --output-dir /path/to/diagnostics

The evaluation split defaults to validation. Train and validation prevalence
are measured separately from the masks used by each dataset split.
"""

import argparse
import csv
import math
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.sim_dataset import SimFireDataset
from model.S2RFireTr import FireTr
from lib.plotting import plot_results, plot_checkpoint_trend
from utils import load_config


def ratio(numerator, denominator):
    """Undefined empty-mask metrics are NaN, rather than a perfect score."""
    return numerator / denominator if denominator else math.nan


def mask_counts(predicted, truth):
    """Return binary-mask counts for one sample and one target time step."""
    predicted = np.asarray(predicted, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    if predicted.shape != truth.shape:
        raise ValueError(f'mask shapes differ: {predicted.shape} != {truth.shape}')
    tp = int(np.count_nonzero(predicted & truth))
    fp = int(np.count_nonzero(predicted & ~truth))
    fn = int(np.count_nonzero(~predicted & truth))
    return dict(tp=tp, fp=fp, fn=fn, gt_pixels=tp + fn,
                pred_pixels=tp + fp, union=tp + fp + fn)


def add_scores(counts, prefix=''):
    """Scores use foreground pixels; empty ground truth has undefined recall."""
    tp, fp, fn = (counts[prefix + key] for key in ('tp', 'fp', 'fn'))
    counts[prefix + 'iou'] = ratio(tp, tp + fp + fn)
    counts[prefix + 'precision'] = ratio(tp, tp + fp)
    counts[prefix + 'recall'] = ratio(tp, tp + fn)
    return counts


def growth_masks(reference, truth, prediction, reverse):
    """New fire in forward time, for both forecast and backtracking.

    Forecast: pixels added after the last observed input fire mask.
    Backtrack: pixels added between the predicted earlier mask and the known
    later input mask. This measures recovered growth, not reverse-time growth.
    """
    if reverse:
        return reference & ~truth, reference & ~prediction
    return truth & ~reference, prediction & ~reference


def event_id(dataset, sample_index):
    path = Path(dataset.target_path[sample_index][0])
    return f'{path.parents[2].name}/{path.parent.name}'


def summarize(rows, group_field):
    groups = defaultdict(list)
    for row in rows:
        groups[row[group_field]].append(row)
    output = []
    count_fields = ('tp', 'fp', 'fn', 'gt_pixels', 'pred_pixels', 'union',
                    'growth_tp', 'growth_fp', 'growth_fn', 'growth_gt_pixels',
                    'growth_pred_pixels', 'growth_union')
    for name, group in sorted(groups.items()):
        item = {group_field: name, 'sample_steps': len(group)}
        for key in count_fields:
            item[key] = sum(row[key] for row in group)
        add_scores(item)
        add_scores(item, 'growth_')
        item['area_ratio'] = ratio(item['pred_pixels'], item['gt_pixels'])
        item['mean_fp_pixels'] = item['fp'] / len(group)
        item['mean_fn_pixels'] = item['fn'] / len(group)
        output.append(item)
    return output


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'no rows to write: {path}')
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@lru_cache(maxsize=2048)
def mask_pixel_count(path, width, height):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
    return int(np.count_nonzero(resized))


def prevalence(dataset):
    """Sample-weighted target prevalence, matching training's repeated frames."""
    width, height = dataset.img_size
    positive = 0
    frames = 0
    for paths in dataset.target_path:
        for path in paths:
            positive += mask_pixel_count(str(path), width, height)
            frames += 1
    pixels = frames * width * height
    return dict(split='train' if dataset.training else 'val',
                samples=len(dataset), target_frames=frames,
                fire_pixels=positive, total_pixels=pixels,
                fire_fraction=ratio(positive, pixels))


def evaluate(dataset, model, device, batch_size, threshold):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, drop_last=False)
    rows = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            (indices, source, target, fuel, vegetation, topography,
             satellite, weather, timestamps) = batch
            logits = model(source.to(device), fuel.to(device),
                           vegetation.to(device), topography.to(device),
                           satellite.to(device), weather.to(device),
                           timestamps.to(device))
            if logits.shape != target.shape:
                raise ValueError(f'model/target shapes differ: {logits.shape} != {target.shape}')
            predicted = (torch.sigmoid(logits) > threshold).cpu().numpy()
            target = (target > 0).numpy()
            source = (source > 0).numpy()
            for b, index in enumerate(indices.tolist()):
                # The last observed input frame is adjacent to the first target
                # in both chronological forecast and reverse-time backtracking.
                reference = source[b, -1]
                for step in range(target.shape[1]):
                    truth, prediction = target[b, step], predicted[b, step]
                    counts = add_scores(mask_counts(prediction, truth))
                    growth_gt, growth_pred = growth_masks(
                        reference, truth, prediction, dataset.reverse)
                    growth = mask_counts(growth_pred, growth_gt)
                    row = dict(sample_index=index, event=event_id(dataset, index),
                               step=step + 1, **counts)
                    row.update({'growth_' + key: value for key, value in growth.items()})
                    add_scores(row, 'growth_')
                    row['area_ratio'] = ratio(row['pred_pixels'], row['gt_pixels'])
                    rows.append(row)
    if not rows:
        raise ValueError('evaluation split contains no target frames')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True, action='append',
                        help='checkpoint path; repeat to compare several epochs')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--split', choices=('train', 'val'), default='val')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if not 0 < args.threshold < 1 or args.batch_size < 1:
        parser.error('threshold must be in (0,1) and batch-size must be positive')
    try:
        import matplotlib  # noqa: F401; fail before expensive dataset/model work
    except ImportError as exc:
        raise RuntimeError('matplotlib is required for the PNG plots; install requirements.txt') from exc
    cfg = load_config(args.config)
    if cfg is None:
        raise ValueError(f'cannot load config: {args.config}')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = {name: SimFireDataset(cfg['dataset'], training=(name == 'train'))
                for name in ('train', 'val')}
    prevalence_rows = [prevalence(datasets[name]) for name in ('train', 'val')]
    write_csv(output_dir / 'fire_prevalence.csv', prevalence_rows)

    device = torch.device(args.device)
    model = FireTr(cfg['FireTr']).to(device)
    summaries = []
    labels = [Path(path).stem for path in args.checkpoint]
    if len(labels) != len(set(labels)):
        parser.error('checkpoint filenames must have distinct stems')
    for checkpoint, label in zip(args.checkpoint, labels):
        current_dir = output_dir if len(labels) == 1 else output_dir / label
        current_dir.mkdir(parents=True, exist_ok=True)
        weights = torch.load(checkpoint, map_location=device, weights_only=True)
        if isinstance(weights, dict) and 'model_state_dict' in weights:
            weights = weights['model_state_dict']
        model.load_state_dict(weights, strict=True)
        rows = evaluate(datasets[args.split], model, device, args.batch_size, args.threshold)
        horizons, events = summarize(rows, 'step'), summarize(rows, 'event')
        write_csv(current_dir / 'sample_steps.csv', rows)
        write_csv(current_dir / 'horizon_metrics.csv', horizons)
        write_csv(current_dir / 'event_metrics.csv', events)
        plot_results(current_dir, horizons, events, prevalence_rows,
                     datasets[args.split].reverse)
        valid_event_iou = [item['iou'] for item in events if math.isfinite(item['iou'])]
        macro = float(np.mean(valid_event_iou)) if valid_event_iou else math.nan
        micro = ratio(sum(item['tp'] for item in horizons),
                      sum(item['union'] for item in horizons))
        summary = dict(split=args.split, checkpoint_label=label,
                       checkpoint=checkpoint, threshold=args.threshold,
                       events=len(events), sample_steps=len(rows),
                       micro_iou=micro, event_macro_iou=macro)
        write_csv(current_dir / 'summary.csv', [summary])
        summaries.append(summary)
        print(f'{label}: global IoU={micro:.4f}, event-mean IoU={macro:.4f}')
    if len(summaries) > 1:
        write_csv(output_dir / 'checkpoint_summary.csv', summaries)
        plot_checkpoint_trend(output_dir, summaries)
    print(f'Wrote diagnostics to {output_dir}')


if __name__ == '__main__':
    main()
