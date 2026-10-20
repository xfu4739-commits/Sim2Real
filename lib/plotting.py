"""Plot epoch histories and offline fire-mask diagnostics."""

import csv
import math
import os

import numpy as np


def _read_numeric_csv(path):
    rows = []
    with open(path, newline='', encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            if not row.get('epoch'):
                continue
            parsed = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = value
            rows.append(parsed)
    return rows


def plot_training_history(metrics_csv, output_dir=None):
    """Render epoch loss and quality curves from an existing metrics.csv."""
    if not os.path.isfile(metrics_csv):
        raise FileNotFoundError(f'metrics CSV not found: {metrics_csv}')
    rows = _read_numeric_csv(metrics_csv)
    if not rows:
        raise ValueError(f'metrics CSV contains no epoch rows: {metrics_csv}')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_dir = output_dir or os.path.dirname(os.path.abspath(metrics_csv))
    os.makedirs(output_dir, exist_ok=True)
    epochs = [row['epoch'] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(epochs, [row['train_loss'] for row in rows], marker='o', label='Train total')
    axes[0].plot(epochs, [row['val_loss'] for row in rows], marker='o', label='Val total')
    axes[0].set_title('Total loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    for key, label, style in (
        ('train_dice_loss', 'Train Dice', '-'),
        ('val_dice_loss', 'Val Dice', '--'),
        ('train_focal_loss', 'Train Focal', '-'),
        ('val_focal_loss', 'Val Focal', '--'),
    ):
        axes[1].plot(epochs, [row[key] for row in rows], style, marker='o', label=label)
    axes[1].set_title('Loss components')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    loss_path = os.path.join(output_dir, 'loss_curves.png')
    fig.savefig(loss_path, dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    for key, label in (
        ('iou_fg', 'Foreground IoU'),
        ('miou', 'mIoU'),
        ('f1', 'F1'),
        ('auprc', 'AUPRC'),
    ):
        axis.plot(epochs, [row[key] for row in rows], marker='o', label=label)
    axis.set_title('Validation metrics')
    axis.set_xlabel('Epoch')
    axis.set_ylabel('Score')
    axis.set_ylim(0, 1)
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    metrics_path = os.path.join(output_dir, 'metrics_curves.png')
    fig.savefig(metrics_path, dpi=160)
    plt.close(fig)
    return loss_path, metrics_path


def plot_fire_horizon_history(csv_path, output_dir, reverse=False):
    """Plot per-epoch validation diagnostics for each target step."""
    rows = _read_numeric_csv(csv_path)
    if not rows:
        return None
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fields = (
        ('iou', 'Foreground IoU'), ('precision', 'Precision'),
        ('recall', 'Recall'), ('area_ratio', 'Predicted / true area'),
        ('growth_iou', 'Recovered growth IoU' if reverse else 'New fire IoU'),
        ('growth_recall', 'Recovered growth recall' if reverse else 'New fire recall'),
        ('mean_fp_pixels', 'Mean false-positive pixels'),
        ('mean_fn_pixels', 'Mean false-negative pixels'),
    )
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for axis, (key, title) in zip(axes.flat, fields):
        for step in sorted({int(row['step']) for row in rows}):
            group = [row for row in rows if int(row['step']) == step]
            axis.plot([row['epoch'] for row in group], [row[key] for row in group],
                      marker='o', label=f'Step {step}')
        axis.set(title=title, xlabel='Epoch')
        if key == 'area_ratio':
            axis.axhline(1, color='gray', linestyle='--', linewidth=1)
        axis.grid(alpha=0.3)
        axis.legend()
    fig.tight_layout()
    output_path = os.path.join(output_dir, 'fire_horizon_curves.png')
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_results(output_dir, horizons, events, prevalence_rows, reverse):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    steps = [item['step'] for item in horizons]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for key, label in (('iou', 'IoU'), ('precision', 'Precision'), ('recall', 'Recall')):
        axes[0].plot(steps, [item[key] for item in horizons], marker='o', label=label)
    axes[0].set_title('Foreground by target step')
    for key, label in (('growth_iou', 'IoU'), ('growth_recall', 'Recall')):
        axes[1].plot(steps, [item[key] for item in horizons], marker='o', label=label)
    axes[1].set_title('Forward-time new fire (recovered)' if reverse else 'New fire after input')
    for axis in axes:
        axis.set_xlabel('Target step')
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / 'horizon_scores.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    values = [item['iou'] for item in events if math.isfinite(item['iou'])]
    axes[0].hist(values, bins=np.linspace(0, 1, 21))
    if values:
        axes[0].axvline(np.mean(values), color='red', linestyle='--',
                        label=f'Event mean {np.mean(values):.3f}')
        axes[0].legend()
    axes[0].set(title=f'Per-event IoU (n={len(values)})', xlabel='IoU', ylabel='Events')
    axes[1].plot(steps, [item['area_ratio'] for item in horizons], marker='o', label='Predicted / GT area')
    axes[1].axhline(1, color='gray', linestyle='--')
    axes[1].set(xlabel='Target step', ylabel='Area ratio')
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / 'event_and_area.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(steps, [item['mean_fp_pixels'] for item in horizons], marker='o', label='FP')
    axes[0].plot(steps, [item['mean_fn_pixels'] for item in horizons], marker='o', label='FN')
    axes[0].set(xlabel='Target step', ylabel='Pixels per sample/frame', title='Fire-mask errors')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].bar([item['split'] for item in prevalence_rows],
                [item['fire_fraction'] for item in prevalence_rows])
    axes[1].set(ylabel='Fire pixels / target pixels', title='Target fire prevalence')
    fig.tight_layout()
    fig.savefig(output_dir / 'errors_and_prevalence.png', dpi=160)
    plt.close(fig)


def plot_checkpoint_trend(output_dir, summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    positions = list(range(len(summaries)))
    fig, axis = plt.subplots(figsize=(max(8, len(summaries) * 1.1), 4.5))
    for key, label in (('micro_iou', 'Global IoU'),
                       ('event_macro_iou', 'Event mean IoU')):
        axis.plot(positions, [row[key] for row in summaries], marker='o', label=label)
    axis.set_xticks(positions, [row['checkpoint_label'] for row in summaries],
                    rotation=35, ha='right')
    axis.set(ylabel='Validation IoU', ylim=(0, 1), title='Checkpoint comparison')
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / 'checkpoint_trend.png', dpi=160)
    plt.close(fig)


def plot_horizon_trends(output_dir, rows):
    """Compare each target horizon across saved epochs, including missing epochs."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    steps = sorted({row['step'] for row in rows})
    labels = list(dict.fromkeys(row['checkpoint_label'] for row in rows))
    use_epochs = all(row['epoch'] is not None for row in rows)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fields = (
        ('iou', 'Foreground IoU'), ('precision', 'Precision'),
        ('recall', 'Recall'), ('area_ratio', 'Predicted / true area'),
        ('growth_iou', 'Forward-time growth IoU'),
        ('growth_recall', 'Forward-time growth recall'),
        ('mean_fp_pixels', 'Mean FP pixels'), ('mean_fn_pixels', 'Mean FN pixels'),
    )
    for axis, (key, title) in zip(axes.flat, fields):
        for step in steps:
            group = [row for row in rows if row['step'] == step]
            if use_epochs:
                group.sort(key=lambda row: row['epoch'])
            x = [row['epoch'] if use_epochs else labels.index(row['checkpoint_label'])
                 for row in group]
            axis.plot(x, [row[key] for row in group], marker='o', label=f'Step {step}')
        axis.set(title=title, xlabel='Epoch' if use_epochs else 'Checkpoint')
        if key == 'area_ratio':
            axis.axhline(1, color='gray', linestyle='--', linewidth=1)
        if not use_epochs:
            axis.set_xticks(range(len(labels)), labels, rotation=35, ha='right')
        axis.grid(alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / 'epoch_horizon_trends.png', dpi=160)
    plt.close(fig)

