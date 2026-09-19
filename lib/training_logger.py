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


class TrainingLogger:
    def __init__(self, log_dir, config_path='config/config.yaml'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_path = os.path.join(log_dir, 'metrics.csv')
        self.log_path = os.path.join(log_dir, 'training.log')
        self.start_time = datetime.now()

        if os.path.exists(config_path):
            shutil.copy(config_path, os.path.join(log_dir, 'config_snapshot.yaml'))

        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(CSV_COLUMNS)

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

    def log_finish(self, total_epochs):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.log(f'Training finished | epochs={total_epochs} | total_time={elapsed / 3600:.2f}h')


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
    return {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_iou': best_metrics[0],
        'best_map': best_metrics[1],
        'config': cfg,
    }


def save_checkpoint(path, checkpoint):
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_metrics = (checkpoint.get('best_iou', 0.0), checkpoint.get('best_map', 0.0))
    return start_epoch, best_metrics


def save_weights_only(path, model):
    torch.save(model.module.state_dict(), path)


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


def dump_run_info(log_dir, info):
    info_path = os.path.join(log_dir, 'run_info.yaml')
    with open(info_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(info, f, allow_unicode=True, sort_keys=False)
