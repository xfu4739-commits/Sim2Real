import csv
import glob
import os
import shutil
from datetime import datetime

import torch
import yaml


CSV_COLUMNS = [
    'epoch',
    'lr',
    'train_loss',
    'train_dice_loss',
    'train_focal_loss',
    'val_loss',
    'val_dice_loss',
    'val_focal_loss',
    'iou_fg',
    'iou_bg',
    'miou',
    'f1',
    'precision',
    'recall',
    'auprc',
    'pixel_acc',
    'train_time_sec',
    'val_time_sec',
    'epoch_time_sec',
    'samples_per_sec',
    'max_mem_mb',
    'best_iou',
    'best_map',
    'timestamp',
]

BATCH_CSV_COLUMNS = [
    'epoch', 'batch', 'total_batches', 'global_step', 'loss', 'dice_loss',
    'focal_loss', 'learning_rate', 'batch_time_sec', 'samples_per_sec',
    'timestamp',
]


def create_run_id():
    return datetime.now().strftime('%Y%m%d_%H%M%S_%f')


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


class TrainingLogger:
    def __init__(self, log_dir, config_path='config/config.yaml', run_id=None,
                 plot_curves=True, save_batch_metrics=False, tensorboard=False,
                 batch_log_interval=1):
        self.run_id = run_id or create_run_id()
        self.base_log_dir = log_dir
        self.log_dir = os.path.join(log_dir, self.run_id)
        os.makedirs(self.log_dir, exist_ok=False)

        self.plot_curves = bool(plot_curves)
        self.save_batch_metrics = bool(save_batch_metrics)
        self.tensorboard_enabled = bool(tensorboard)
        self.batch_log_interval = max(1, int(batch_log_interval))
        self.csv_path = os.path.join(self.log_dir, 'metrics.csv')
        self.batch_csv_path = os.path.join(self.log_dir, 'batch_metrics.csv')
        self.log_path = os.path.join(self.log_dir, 'training.log')
        self.start_time = datetime.now()
        self.writer = None

        if os.path.exists(config_path):
            shutil.copy(config_path, os.path.join(self.log_dir, 'config_snapshot.yaml'))

        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(CSV_COLUMNS)
        if self.save_batch_metrics:
            with open(self.batch_csv_path, 'w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(BATCH_CSV_COLUMNS)
        if self.tensorboard_enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as exc:
                raise RuntimeError(
                    'TensorBoard logging is enabled but tensorboard is not installed; '
                    'run: python -m pip install -r requirements.txt'
                ) from exc
            self.writer = SummaryWriter(log_dir=os.path.join(self.log_dir, 'tensorboard'))

    def log(self, message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        line = f'[{timestamp}] {message}'
        print(line)
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')

    def log_run_header(self, info):
        self.log('=' * 60)
        self.log(f'Training run started | run_id={self.run_id}')
        for key, value in info.items():
            self.log(f'  {key}: {value}')
        self.log('=' * 60)

    def log_epoch(self, metrics):
        row = [metrics.get(col, '') for col in CSV_COLUMNS]
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(row)

        if self.writer is not None:
            epoch = metrics['epoch']
            for key in CSV_COLUMNS:
                value = metrics.get(key)
                if key not in ('epoch', 'timestamp') and isinstance(value, (int, float)):
                    self.writer.add_scalar(f'epoch/{key}', value, epoch)
            self.writer.flush()
        if self.plot_curves:
            plot_training_history(self.csv_path, self.log_dir)

        self.log(
            f'Epoch {metrics["epoch"]} summary | '
            f'train_loss={metrics["train_loss"]:.4f} '
            f'(dice={metrics["train_dice_loss"]:.4f}, focal={metrics["train_focal_loss"]:.4f}) | '
            f'val_loss={metrics["val_loss"]:.4f} '
            f'(dice={metrics["val_dice_loss"]:.4f}, focal={metrics["val_focal_loss"]:.4f}) | '
            f'IOU_fg={metrics["iou_fg"]:.4f} mIoU={metrics["miou"]:.4f} '
            f'F1={metrics["f1"]:.4f} P={metrics["precision"]:.4f} R={metrics["recall"]:.4f} '
            f'AUPRC={metrics["auprc"]:.4f} | '
            f'lr={metrics["lr"]:.6f} | '
            f'time={metrics["epoch_time_sec"]:.1f}s '
            f'(train={metrics["train_time_sec"]:.1f}s, val={metrics["val_time_sec"]:.1f}s) | '
            f'{metrics["samples_per_sec"]:.1f} samples/s | '
            f'mem={metrics["max_mem_mb"]:.0f}MB | '
            f'best_iou={metrics["best_iou"]:.4f} best_map={metrics["best_map"]:.4f}'
        )

    def should_log_batch(self, batch, total_batches):
        return batch == 1 or batch == total_batches or batch % self.batch_log_interval == 0

    def log_batch(self, metrics):
        if not self.should_log_batch(metrics['batch'], metrics['total_batches']):
            return
        if self.save_batch_metrics:
            row = [metrics.get(col, '') for col in BATCH_CSV_COLUMNS]
            with open(self.batch_csv_path, 'a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(row)
        if self.writer is not None:
            step = metrics['global_step']
            for key in ('loss', 'dice_loss', 'focal_loss', 'learning_rate', 'samples_per_sec'):
                self.writer.add_scalar(f'batch/{key}', metrics[key], step)

    def close(self):
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None

    def log_finish(self, total_epochs):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.log(f'Training finished | epochs={total_epochs} | total_time={elapsed / 3600:.2f}h')
        self.close()


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def compute_segmentation_stats(metric):
    iou = metric.IntersectionOverUnion()
    miou = metric.meanIntersectionOverUnion()
    f1 = metric.f1()
    pixel_acc = metric.pixelAccuracy()

    tp = metric.confusionMatrix[1, 1]
    fp = metric.confusionMatrix[0, 1]
    fn = metric.confusionMatrix[1, 0]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        'iou_fg': float(iou[1]),
        'iou_bg': float(iou[0]),
        'miou': float(miou),
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'pixel_acc': float(pixel_acc),
    }


def build_checkpoint(epoch, model, optimizer, scheduler, best_metrics, cfg):
    model = model.module if hasattr(model, 'module') else model
    return {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_iou': best_metrics[0],
        'best_map': best_metrics[1],
        'config': cfg,
    }


def save_checkpoint(path, checkpoint):
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_metrics = (checkpoint.get('best_iou', 0.0), checkpoint.get('best_map', 0.0))
    return start_epoch, best_metrics


def save_weights_only(path, model):
    model = model.module if hasattr(model, 'module') else model
    torch.save(model.state_dict(), path)


def prune_old_checkpoints(model_save_path, keep_last_n):
    if keep_last_n <= 0:
        return
    pattern = os.path.join(model_save_path, 'epoch_*.pth')
    files = sorted(glob.glob(pattern), key=lambda p: int(os.path.basename(p).split('_')[1].split('.')[0]))
    for old_path in files[:-keep_last_n]:
        os.remove(old_path)


def ensure_dirs(*paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)


def _to_yaml_safe(value):
    """Coerce torch/packaging version objects and other metadata to plain scalars."""
    if isinstance(value, dict):
        return {str(key): _to_yaml_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_yaml_safe(item) for item in value]
    if value is None:
        return None
    # PyYAML SafeDumper only accepts built-in str, not str subclasses (e.g. torch.__version__).
    if isinstance(value, str):
        return str(value)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    return str(value)


def dump_run_info(log_dir, info):
    info_path = os.path.join(log_dir, 'run_info.yaml')
    with open(info_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(
            _to_yaml_safe(info),
            f,
            allow_unicode=True,
            sort_keys=False,
        )
