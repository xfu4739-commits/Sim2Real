import os
import sys
import time
from datetime import datetime

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import StepLR
from torchmetrics import AveragePrecision
from tqdm import tqdm

from model.S2RFireTr import FireTr
from dataset.sim_dataset import SimFireDataset
from lib.dice_loss import BinaryDiceLoss
from lib.focal_loss import BinaryFocalLoss
from lib.training_logger import (
    TrainingLogger,
    build_checkpoint,
    compute_segmentation_stats,
    count_parameters,
    dump_run_info,
    ensure_dirs,
    load_checkpoint,
    prune_old_checkpoints,
    save_checkpoint,
    save_weights_only,
)
from metrics.metrics import SegmentationMetric
from utils import setup_seed, load_config
from ddp.ddp_setup import setup, cleanup

WORLD_SIZE = 4


def init_distributed(rank=None, world_size=None):
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            backend = 'nccl' if torch.cuda.is_available() else 'gloo'
            dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f'cuda:{local_rank}')
    else:
        setup(rank, world_size)
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
    return rank, world_size, device


def is_main_process(rank):
    return rank == 0


def reduce_values(values, device):
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.cpu().tolist()


def load_data(cfg, rank, world_size):
    train_dataset = SimFireDataset(cfg['dataset'])
    val_dataset = SimFireDataset(cfg['dataset'], training=False)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg['training']['batch_size'],
        sampler=train_sampler,
        num_workers=cfg['dataset']['num_workers'],
        pin_memory=True,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg['training']['batch_size'],
        sampler=val_sampler,
        num_workers=cfg['dataset']['num_workers'],
        pin_memory=True,
        drop_last=False,
    )
    if is_main_process(rank):
        print(f'len of train_dataloader (per GPU): {len(train_dataloader)}')
        print(f'len of val_dataloader (per GPU): {len(val_dataloader)}')
        print(f'global batch size: {cfg["training"]["batch_size"] * world_size}')
    return train_dataloader, val_dataloader, train_sampler, val_sampler, len(train_dataset), len(val_dataset)


def initialize_model(cfg, device, rank):
    model = FireTr(cfg['FireTr']).to(device)
    if cfg['checkpoint']['checkpoint'] and not cfg['checkpoint'].get('resume', False):
        checkpoint_path = cfg['checkpoint']['checkpoint_path']
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint, strict=False)
        if is_main_process(rank):
            print(f'Loaded pretrained weights from {checkpoint_path}')
    model = DDP(model, device_ids=[device.index])
    return model


def initialize_optimizer_scheduler(model, cfg):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'])
    scheduler = StepLR(optimizer, step_size=5, gamma=0.95)
    return optimizer, scheduler


def compute_losses(pred, target, dice, focal):
    dice_loss = dice(pred, target, use_sigmoid=True)
    focal_loss = focal(pred, target)
    total_loss = dice_loss + focal_loss
    return total_loss, dice_loss.item(), focal_loss.item()


def train_loop(dataloader, model, optimizer, device, loss_dice, loss_focal, epoch, total_epochs, rank, batch_size):
    model.train()
    train_loss, train_dice_loss, train_focal_loss = [], [], []
    total_batches = len(dataloader)
    epoch_start = time.time()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    progress = tqdm(
        dataloader,
        total=total_batches,
        desc=f'Epoch {epoch}/{total_epochs} train',
        unit='batch',
        leave=True,
        dynamic_ncols=True,
        file=sys.stdout,
        disable=not is_main_process(rank),
    )
    for batch_idx, (index, input_squence, output_squence, fuel, vegetation, topography, satellite_images, weather_data, timestamps) in enumerate(progress, start=1):
        batch_start = time.time()
        input_squence, output_squence, fuel, vegetation, topography, satellite_images, weather_data, timestamps = (
            input_squence.to(device), output_squence.to(device), fuel.to(device), vegetation.to(device),
            topography.to(device), satellite_images.to(device), weather_data.to(device), timestamps.to(device),
        )
        optimizer.zero_grad()
        pred = model(input_squence, fuel, vegetation, topography, satellite_images, weather_data, timestamps)
        loss, dice_val, focal_val = compute_losses(pred, output_squence, loss_dice, loss_focal)
        train_loss.append(loss.item())
        train_dice_loss.append(dice_val)
        train_focal_loss.append(focal_val)
        loss.backward()
        optimizer.step()

        if is_main_process(rank):
            batch_time = max(time.time() - batch_start, 1e-6)
            samples_per_sec = batch_size / batch_time
            progress.set_postfix(
                batch=f'{batch_idx}/{total_batches}',
                loss=f'{loss.item():.4f}',
                dice=f'{dice_val:.4f}',
                focal=f'{focal_val:.4f}',
                sps=f'{samples_per_sec:.1f}',
                refresh=False,
            )

    loss_sum, batch_count = reduce_values([sum(train_loss), len(train_loss)], device)
    dice_sum, _ = reduce_values([sum(train_dice_loss), len(train_dice_loss)], device)
    focal_sum, _ = reduce_values([sum(train_focal_loss), len(train_focal_loss)], device)

    max_mem_mb = 0.0
    if device.type == 'cuda':
        max_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    train_time_sec = time.time() - epoch_start
    global_samples = batch_count * batch_size
    samples_per_sec = global_samples / train_time_sec if train_time_sec > 0 else 0.0

    return {
        'train_loss': loss_sum / batch_count if batch_count > 0 else 0.0,
        'train_dice_loss': dice_sum / batch_count if batch_count > 0 else 0.0,
        'train_focal_loss': focal_sum / batch_count if batch_count > 0 else 0.0,
        'train_time_sec': train_time_sec,
        'samples_per_sec': samples_per_sec,
        'max_mem_mb': max_mem_mb,
    }


def val_loop(dataloader, model, device, loss_dice, loss_focal, epoch, total_epochs, rank):
    model.eval()
    val_loss, val_dice_loss, val_focal_loss = [], [], []
    metric = SegmentationMetric(2)
    auprc = AveragePrecision('binary', average='macro').to(device)
    auprc_list = []
    val_start = time.time()

    with torch.no_grad():
        total_batches = len(dataloader)
        progress = tqdm(
            dataloader,
            total=total_batches,
            desc=f'Epoch {epoch}/{total_epochs} val',
            unit='batch',
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            disable=not is_main_process(rank),
        )
        for batch_idx, (index, input_squence, output_squence, fuel, vegetation, topography, satellite_images, weather_data, timestamps) in enumerate(progress, start=1):
            input_squence, output_squence, fuel, vegetation, topography, satellite_images, weather_data, timestamps = (
                input_squence.to(device), output_squence.to(device), fuel.to(device), vegetation.to(device),
                topography.to(device), satellite_images.to(device), weather_data.to(device), timestamps.to(device),
            )
            pred = model(input_squence, fuel, vegetation, topography, satellite_images, weather_data, timestamps)
            loss, dice_val, focal_val = compute_losses(pred, output_squence, loss_dice, loss_focal)
            val_loss.append(loss.item())
            val_dice_loss.append(dice_val)
            val_focal_loss.append(focal_val)

            pred = torch.sigmoid(pred)
            auprc.update(pred, output_squence.long())
            auprc_value = auprc.compute()
            if not torch.isnan(auprc_value):
                auprc_list.append(auprc_value.item())
            auprc.reset()

            pred = torch.where(pred > 0.5, torch.ones_like(pred), torch.zeros_like(pred))
            pred, output_squence = pred.cpu().numpy(), output_squence.cpu().numpy()
            pred, output_squence = pred.astype(np.int32), output_squence.astype(np.int32)
            metric.addBatch(pred, output_squence)
            if is_main_process(rank):
                progress.set_postfix(
                    batch=f'{batch_idx}/{total_batches}',
                    loss=f'{loss.item():.4f}',
                    avg=f'{np.mean(val_loss):.4f}',
                    refresh=False,
                )

    loss_sum, batch_count = reduce_values([sum(val_loss), len(val_loss)], device)
    dice_sum, _ = reduce_values([sum(val_dice_loss), len(val_dice_loss)], device)
    focal_sum, _ = reduce_values([sum(val_focal_loss), len(val_focal_loss)], device)

    confusion = torch.tensor(metric.confusionMatrix, dtype=torch.float64, device=device)
    dist.all_reduce(confusion, op=dist.ReduceOp.SUM)
    metric.confusionMatrix = confusion.cpu().numpy()

    auprc_sum, auprc_count = reduce_values([sum(auprc_list), len(auprc_list)], device)
    mAp = auprc_sum / auprc_count if auprc_count > 0 else 0.0

    seg_stats = compute_segmentation_stats(metric)
    metric.reset()

    return {
        'val_loss': loss_sum / batch_count if batch_count > 0 else 0.0,
        'val_dice_loss': dice_sum / batch_count if batch_count > 0 else 0.0,
        'val_focal_loss': focal_sum / batch_count if batch_count > 0 else 0.0,
        'auprc': mAp,
        'val_time_sec': time.time() - val_start,
        **seg_stats,
    }


def save_best_weights(model, iou, mAp, best_metrics, model_save_path, logger):
    best_iou, best_map = best_metrics
    if iou > best_iou:
        logger.log(f'New best IOU: {iou:.4f} (prev {best_iou:.4f})')
        best_iou = iou
        save_weights_only(os.path.join(model_save_path, 'best_iou.pth'), model)
    if mAp > best_map:
        logger.log(f'New best mAP: {mAp:.4f} (prev {best_map:.4f})')
        best_map = mAp
        save_weights_only(os.path.join(model_save_path, 'best_map.pth'), model)
    return best_iou, best_map


def save_epoch_checkpoints(model, optimizer, scheduler, epoch, best_metrics, cfg, logger):
    ckpt_cfg = cfg['checkpoint']
    model_save_path = ckpt_cfg['model_save_path']
    checkpoint = build_checkpoint(epoch, model, optimizer, scheduler, best_metrics, cfg)

    last_path = os.path.join(model_save_path, 'last.pth')
    save_checkpoint(last_path, checkpoint)
    logger.log(f'Saved checkpoint: {last_path}')

    save_interval = ckpt_cfg.get('save_interval', 0)
    if save_interval > 0 and epoch % save_interval == 0:
        epoch_path = os.path.join(model_save_path, f'epoch_{epoch}.pth')
        save_checkpoint(epoch_path, checkpoint)
        logger.log(f'Saved periodic checkpoint: {epoch_path}')
        prune_old_checkpoints(model_save_path, ckpt_cfg.get('keep_last_n', 3))


def main_worker(rank, world_size):
    setup_seed(12345 + rank)
    rank, world_size, device = init_distributed(rank, world_size)
    cfg = load_config('config/config.yaml')

    ckpt_cfg = cfg['checkpoint']
    log_cfg = cfg.get('logging', {})
    model_save_path = ckpt_cfg['model_save_path']
    log_dir = log_cfg.get('log_dir', os.path.join('output/logs', os.path.basename(model_save_path)))

    if is_main_process(rank):
        ensure_dirs(model_save_path, log_dir)
    if dist.is_initialized():
        dist.barrier()

    logger = None
    if is_main_process(rank):
        logger = TrainingLogger(log_dir)
        logger.log(f'DDP training on {world_size} GPUs, rank={rank}, device={device}')

    train_dataloader, val_dataloader, train_sampler, val_sampler, train_size, val_size = load_data(
        cfg, rank, world_size
    )

    model = initialize_model(cfg, device, rank)
    optimizer, scheduler = initialize_optimizer_scheduler(model, cfg)

    start_epoch = 1
    best_metrics = (0.0, 0.0)
    if ckpt_cfg.get('resume', False):
        resume_path = ckpt_cfg['resume_path']
        start_epoch, best_metrics = load_checkpoint(
            resume_path, model.module, optimizer, scheduler, device
        )
        if is_main_process(rank):
            logger.log(f'Resumed from {resume_path} at epoch {start_epoch - 1}')
            logger.log(f'Restored best metrics: IOU={best_metrics[0]:.4f}, mAP={best_metrics[1]:.4f}')

    loss_dice = BinaryDiceLoss()
    loss_focal = BinaryFocalLoss()
    total_epochs = cfg['training']['epochs']
    per_gpu_batch = cfg['training']['batch_size']
    global_batch = per_gpu_batch * world_size

    if is_main_process(rank):
        total_params, trainable_params = count_parameters(model.module)
        run_info = {
            'run_id': logger.run_id,
            'start_time': logger.start_time.isoformat(),
            'world_size': world_size,
            'per_gpu_batch_size': per_gpu_batch,
            'global_batch_size': global_batch,
            'train_samples': train_size,
            'val_samples': val_size,
            'total_epochs': total_epochs,
            'start_epoch': start_epoch,
            'learning_rate': cfg['training']['learning_rate'],
            'model_save_path': model_save_path,
            'log_dir': log_dir,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'pytorch_version': torch.__version__,
            'cuda_version': torch.version.cuda,
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
        }
        logger.log_run_header(run_info)
        dump_run_info(log_dir, run_info)

    for epoch in range(start_epoch, total_epochs + 1):
        epoch_wall_start = time.time()
        train_sampler.set_epoch(epoch)

        train_metrics = train_loop(
            train_dataloader, model, optimizer, device, loss_dice, loss_focal,
            epoch, total_epochs, rank, per_gpu_batch,
        )
        val_metrics = val_loop(
            val_dataloader, model, device, loss_dice, loss_focal, epoch, total_epochs, rank
        )
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        if is_main_process(rank):
            best_metrics = save_best_weights(
                model, val_metrics['iou_fg'], val_metrics['auprc'], best_metrics, model_save_path, logger
            )
            save_epoch_checkpoints(model, optimizer, scheduler, epoch, best_metrics, cfg, logger)

            epoch_time_sec = time.time() - epoch_wall_start
            epoch_log = {
                'epoch': epoch,
                'lr': current_lr,
                **train_metrics,
                **val_metrics,
                'epoch_time_sec': epoch_time_sec,
                'best_iou': best_metrics[0],
                'best_map': best_metrics[1],
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }
            logger.log_epoch(epoch_log)

    if is_main_process(rank):
        logger.log_finish(total_epochs)

    if dist.is_initialized():
        cleanup()


def main():
    if 'LOCAL_RANK' in os.environ:
        world_size = int(os.environ.get('WORLD_SIZE', WORLD_SIZE))
        rank = int(os.environ['LOCAL_RANK'])
        main_worker(rank, world_size)
    else:
        mp.spawn(main_worker, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)


if __name__ == '__main__':
    main()
