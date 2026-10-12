import os
import sys
import time
import argparse
import math
import tempfile
import uuid
from contextlib import nullcontext
from datetime import datetime
from itertools import islice
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from model.S2RFireTr import FireTr
from dataset.sim_dataset import SimFireDataset
from lib.dice_loss import BinaryDiceLoss
from lib.focal_loss import BinaryFocalLoss
from lib.losses import combined_losses
from lib.training_logger import (
    TrainingLogger,
    build_checkpoint,
    compute_segmentation_stats,
    count_parameters,
    create_run_id,
    dump_run_info,
    ensure_dirs,
    load_checkpoint,
    prune_old_checkpoints,
    save_checkpoint,
    save_weights_only,
)
from metrics.metrics import BinaryAUPRCHistogram, SegmentationMetric
from utils import setup_seed, load_config
from ddp.ddp_setup import setup, cleanup

WORLD_SIZE = 1


class DistributedEvalSampler(Sampler):
    """Partition evaluation data across ranks without padding or duplication."""

    def __init__(self, dataset, num_replicas, rank):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self):
        remaining = len(self.dataset) - self.rank
        return 0 if remaining <= 0 else math.ceil(remaining / self.num_replicas)


def _runtime_device(local_rank=0):
    if local_rank < 0:
        raise ValueError(f'local rank must be non-negative, got {local_rank}')
    if torch.cuda.is_available():
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f'local rank {local_rank} exceeds available CUDA devices '
                f'({torch.cuda.device_count()})'
            )
        torch.cuda.set_device(local_rank)
        return torch.device(f'cuda:{local_rank}')
    return torch.device('cpu')


def validate_training_runtime(cfg, device):
    """Fail before creating run directories if this experiment cannot train."""
    training = cfg['training']
    if training.get('require_cuda', False) and device.type != 'cuda':
        raise RuntimeError('This experiment requires CUDA, but no CUDA GPU is available')
    for key in ('batch_size', 'gradient_accumulation_steps'):
        value = training.get(key, 1)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f'training.{key} must be a positive integer, got {value!r}')

    roots = cfg['dataset']['sim_root_dir']
    configured = [roots] if isinstance(roots, str) else list(roots)
    missing = [root for root in configured if not Path(root).expanduser().is_dir()]
    if missing:
        raise FileNotFoundError(
            'Configured simulation data paths do not exist: '
            + ', '.join(map(str, missing))
        )


def init_distributed(rank=None, world_size=None):
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        device = _runtime_device(local_rank)
        setup(int(os.environ['RANK']), int(os.environ['WORLD_SIZE']))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0 if rank is None else rank
        world_size = 1 if world_size is None else world_size
        device = _runtime_device(rank if torch.cuda.is_available() else 0)
        if world_size > 1:
            setup(rank, world_size)
    return rank, world_size, device


def is_main_process(rank):
    return rank == 0


def reduce_values(values, device):
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.cpu().tolist()


def load_data(cfg, rank, world_size, device):
    train_dataset = SimFireDataset(cfg['dataset'])
    val_dataset = SimFireDataset(cfg['dataset'], training=False)
    # Drop the incomplete global batch instead of padding it with repeated scenes.
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    )
    val_sampler = DistributedEvalSampler(
        val_dataset, num_replicas=world_size, rank=rank
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg['training']['batch_size'],
        sampler=train_sampler,
        num_workers=cfg['dataset']['num_workers'],
        pin_memory=device.type == 'cuda',
        drop_last=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg['training']['batch_size'],
        sampler=val_sampler,
        num_workers=cfg['dataset']['num_workers'],
        pin_memory=device.type == 'cuda',
        drop_last=False,
    )
    if len(train_dataloader) == 0:
        raise ValueError(
            'No complete training batch: reduce training.batch_size or the number '
            'of processes, or add more training samples.'
        )
    if is_main_process(rank):
        print(f'len of train_dataloader (per process): {len(train_dataloader)}')
        print(f'len of val_dataloader (per process): {len(val_dataloader)}')
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
    if dist.is_initialized():
        device_ids = [device.index] if device.type == 'cuda' else None
        model = DDP(model, device_ids=device_ids)
    return model


def initialize_optimizer_scheduler(model, cfg):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'])
    scheduler = StepLR(optimizer, step_size=5, gamma=0.95)
    return optimizer, scheduler


def compute_losses(pred, target, dice, focal):
    total_loss, dice_loss, focal_loss = combined_losses(pred, target, dice, focal)
    return total_loss, dice_loss.item(), focal_loss.item()


def train_loop(dataloader, model, optimizer, device, loss_dice, loss_focal,
               epoch, total_epochs, rank, batch_size, logger=None,
               accumulation_steps=1):
    model.train()
    train_loss, train_dice_loss, train_focal_loss = [], [], []
    total_batches = len(dataloader)
    if accumulation_steps < 1:
        raise ValueError('accumulation_steps must be at least 1')
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
        window_start = ((batch_idx - 1) // accumulation_steps) * accumulation_steps
        window_size = min(accumulation_steps, total_batches - window_start)
        update_due = batch_idx == window_start + window_size
        if batch_idx == window_start + 1:
            optimizer.zero_grad(set_to_none=True)
        # DDP must see the forward pass inside no_sync as well as backward.
        sync_context = (model.no_sync() if isinstance(model, DDP) and not update_due
                        else nullcontext())
        with sync_context:
            pred = model(input_squence, fuel, vegetation, topography, satellite_images, weather_data, timestamps)
            if pred.shape != output_squence.shape:
                raise ValueError(
                    f'model output shape {tuple(pred.shape)} does not match '
                    f'target shape {tuple(output_squence.shape)}'
                )
            loss, dice_val, focal_val = compute_losses(pred, output_squence, loss_dice, loss_focal)
            if not torch.isfinite(loss).item():
                raise RuntimeError(f'non-finite training loss at epoch {epoch}, batch {batch_idx}')
            (loss / window_size).backward()
        train_loss.append(loss.item())
        train_dice_loss.append(dice_val)
        train_focal_loss.append(focal_val)
        if update_due:
            optimizer.step()

        if is_main_process(rank):
            batch_time = max(time.time() - batch_start, 1e-6)
            samples_per_sec = batch_size / batch_time
            if logger is not None:
                logger.log_batch({
                    'epoch': epoch,
                    'batch': batch_idx,
                    'total_batches': total_batches,
                    'global_step': (epoch - 1) * total_batches + batch_idx,
                    'loss': loss.item(),
                    'dice_loss': dice_val,
                    'focal_loss': focal_val,
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'batch_time_sec': batch_time,
                    'samples_per_sec': samples_per_sec,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                })
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
    eval_model = model.module if isinstance(model, DDP) else model
    loss_sum_local = 0.0
    dice_sum_local = 0.0
    focal_sum_local = 0.0
    sample_count_local = 0
    metric = SegmentationMetric(2)
    auprc = BinaryAUPRCHistogram(device)
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
            pred = eval_model(input_squence, fuel, vegetation, topography, satellite_images, weather_data, timestamps)
            loss, dice_val, focal_val = compute_losses(pred, output_squence, loss_dice, loss_focal)
            current_batch = input_squence.shape[0]
            loss_sum_local += loss.item() * current_batch
            dice_sum_local += dice_val * current_batch
            focal_sum_local += focal_val * current_batch
            sample_count_local += current_batch

            pred = torch.sigmoid(pred)
            auprc.update(pred, output_squence)

            pred = torch.where(pred > 0.5, torch.ones_like(pred), torch.zeros_like(pred))
            pred, output_squence = pred.cpu().numpy(), output_squence.cpu().numpy()
            pred, output_squence = pred.astype(np.int32), output_squence.astype(np.int32)
            metric.addBatch(pred, output_squence)
            if is_main_process(rank):
                progress.set_postfix(
                    batch=f'{batch_idx}/{total_batches}',
                    loss=f'{loss.item():.4f}',
                    avg=f'{loss_sum_local / max(sample_count_local, 1):.4f}',
                    refresh=False,
                )

    loss_sum, dice_sum, focal_sum, sample_count = reduce_values(
        [loss_sum_local, dice_sum_local, focal_sum_local, sample_count_local],
        device,
    )

    confusion = torch.tensor(metric.confusionMatrix, dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(confusion, op=dist.ReduceOp.SUM)
    metric.confusionMatrix = confusion.cpu().numpy()

    mAp = float(auprc.compute(sync_dist=True).item())

    seg_stats = compute_segmentation_stats(metric)
    metric.reset()

    return {
        'val_loss': loss_sum / sample_count if sample_count > 0 else 0.0,
        'val_dice_loss': dice_sum / sample_count if sample_count > 0 else 0.0,
        'val_focal_loss': focal_sum / sample_count if sample_count > 0 else 0.0,
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


def main_worker(rank, world_size, config_path):
    setup_seed(12345 + rank)
    rank, world_size, device = init_distributed(rank, world_size)
    try:
        _run_training(rank, world_size, device, config_path)
    finally:
        cleanup()


def _run_training(rank, world_size, device, config_path):
    cfg = load_config(config_path)
    validate_training_runtime(cfg, device)

    # Check the actual data before creating an output run directory.
    train_dataloader, val_dataloader, train_sampler, val_sampler, train_size, val_size = load_data(
        cfg, rank, world_size, device
    )

    ckpt_cfg = cfg['checkpoint']
    log_cfg = cfg.get('logging', {})
    base_model_save_path = ckpt_cfg['model_save_path']
    base_log_dir = log_cfg.get(
        'log_dir', os.path.join('output/logs', os.path.basename(base_model_save_path))
    )

    run_id_holder = [create_run_id() if is_main_process(rank) else None]
    if dist.is_initialized():
        dist.broadcast_object_list(run_id_holder, src=0)
    run_id = run_id_holder[0]
    model_save_path = os.path.join(base_model_save_path, run_id)
    ckpt_cfg['model_save_path'] = model_save_path

    if is_main_process(rank):
        ensure_dirs(model_save_path)
    if dist.is_initialized():
        dist.barrier()

    logger = None
    if is_main_process(rank):
        logger = TrainingLogger(
            base_log_dir,
            config_path=config_path,
            run_id=run_id,
            plot_curves=log_cfg.get('plot_curves', True),
            save_batch_metrics=log_cfg.get('save_batch_metrics', False),
            tensorboard=log_cfg.get('tensorboard', True),
            batch_log_interval=log_cfg.get('batch_log_interval', 10),
        )
        mode = 'DDP' if dist.is_initialized() else 'single process'
        logger.log(f'{mode} training, world_size={world_size}, rank={rank}, device={device}')
        logger.log(f'Run outputs: logs={logger.log_dir}, models={model_save_path}')

    model = initialize_model(cfg, device, rank)
    optimizer, scheduler = initialize_optimizer_scheduler(model, cfg)

    start_epoch = 1
    best_metrics = (0.0, 0.0)
    if ckpt_cfg.get('resume', False):
        resume_path = ckpt_cfg['resume_path']
        start_epoch, best_metrics = load_checkpoint(
            resume_path,
            model.module if isinstance(model, DDP) else model,
            optimizer,
            scheduler,
            device,
        )
        if is_main_process(rank):
            logger.log(f'Resumed from {resume_path} at epoch {start_epoch - 1}')
            logger.log(f'Restored best metrics: IOU={best_metrics[0]:.4f}, mAP={best_metrics[1]:.4f}')

    loss_dice = BinaryDiceLoss()
    loss_focal = BinaryFocalLoss()
    total_epochs = cfg['training']['epochs']
    per_gpu_batch = cfg['training']['batch_size']
    global_batch = per_gpu_batch * world_size
    accumulation_steps = cfg['training'].get('gradient_accumulation_steps', 1)

    if is_main_process(rank):
        total_params, trainable_params = count_parameters(
            model.module if isinstance(model, DDP) else model
        )
        run_info = {
            'run_id': logger.run_id,
            'start_time': logger.start_time.isoformat(),
            'world_size': world_size,
            'per_gpu_batch_size': per_gpu_batch,
            'global_batch_size': global_batch,
            'gradient_accumulation_steps': accumulation_steps,
            'effective_batch_size': global_batch * accumulation_steps,
            'train_samples': train_size,
            'val_samples': val_size,
            'total_epochs': total_epochs,
            'start_epoch': start_epoch,
            'learning_rate': cfg['training']['learning_rate'],
            'model_save_path': model_save_path,
            'log_dir': logger.log_dir,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'pytorch_version': torch.__version__,
            'cuda_version': torch.version.cuda,
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
        }
        logger.log_run_header(run_info)
        dump_run_info(logger.log_dir, run_info)

    for epoch in range(start_epoch, total_epochs + 1):
        epoch_wall_start = time.time()
        train_sampler.set_epoch(epoch)

        train_metrics = train_loop(
            train_dataloader, model, optimizer, device, loss_dice, loss_focal,
            epoch, total_epochs, rank, per_gpu_batch, logger,
            accumulation_steps=accumulation_steps,
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

        # Keep every rank on the same epoch while rank 0 writes checkpoints.
        if dist.is_initialized():
            dist.barrier()

    if is_main_process(rank):
        logger.log_finish(total_epochs)

def run_smoke_test(config_path):
    """Run one configured optimizer update without writing checkpoints."""
    setup_seed(12345)
    device = _runtime_device(0)
    cfg = load_config(config_path)
    validate_training_runtime(cfg, device)
    train_loader, _, _, _, train_size, _ = load_data(cfg, 0, 1, device)
    accumulation_steps = cfg['training'].get('gradient_accumulation_steps', 1)
    batches = list(islice(train_loader, accumulation_steps))
    model = FireTr(cfg['FireTr']).to(device)
    optimizer, _ = initialize_optimizer_scheduler(model, cfg)
    metrics = train_loop(
        batches, model, optimizer, device, BinaryDiceLoss(), BinaryFocalLoss(),
        1, 1, 0, cfg['training']['batch_size'],
        accumulation_steps=accumulation_steps,
    )
    if not math.isfinite(metrics['train_loss']):
        raise RuntimeError(f"smoke-test loss is not finite: {metrics['train_loss']}")
    print(
        'Smoke test passed: '
        f'device={device}, samples={train_size}, '
        f'microbatch={batches[0][1].shape[0]}, accumulated={len(batches)}, '
        f'output={tuple(batches[0][2].shape)}, '
        f'loss={metrics["train_loss"]:.6f}, '
        f'dice={metrics["train_dice_loss"]:.6f}, '
        f'focal={metrics["train_focal_loss"]:.6f}, '
        f'peak_allocated_mb={metrics["max_mem_mb"]:.0f}'
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/config.yaml')
    parser.add_argument(
        '--world-size',
        type=int,
        default=None,
        help='number of local training processes; defaults to WORLD_SIZE or 1',
    )
    parser.add_argument('--local-rank', '--local_rank', type=int, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument(
        '--smoke-test',
        action='store_true',
        help='run one configured optimizer update, then exit without writing outputs',
    )
    args = parser.parse_args()
    if args.smoke_test:
        if 'LOCAL_RANK' in os.environ:
            raise ValueError('--smoke-test runs on one process; use python train.py --smoke-test')
        run_smoke_test(args.config)
        return
    if 'LOCAL_RANK' in os.environ:
        world_size = int(os.environ.get('WORLD_SIZE', WORLD_SIZE))
        rank = int(os.environ['RANK'])
        if world_size > 1 and not torch.cuda.is_available():
            raise RuntimeError('Multi-GPU training requires CUDA, but no CUDA device is available')
        if args.world_size is not None and args.world_size != world_size:
            raise ValueError('--world-size conflicts with torchrun WORLD_SIZE')
        main_worker(rank, world_size, args.config)
    else:
        world_size = args.world_size
        if world_size is None:
            world_size = int(os.environ.get('WORLD_SIZE', WORLD_SIZE))
        if world_size < 1:
            raise ValueError('--world-size must be at least 1')
        if world_size > 1 and not torch.cuda.is_available():
            raise RuntimeError('Multi-GPU training requires CUDA, but no CUDA device is available')
        if torch.cuda.is_available() and world_size > torch.cuda.device_count():
            raise ValueError(
                f'world_size={world_size}, but only {torch.cuda.device_count()} CUDA devices are available'
            )
        if world_size == 1:
            main_worker(0, 1, args.config)
            return
        init_file = os.path.join(tempfile.gettempdir(), f"sim2real_fire_ddp_{uuid.uuid4().hex}.init")
        os.environ["DDP_INIT_FILE"] = init_file
        try:
            mp.spawn(
                main_worker,
                args=(world_size, args.config),
                nprocs=world_size,
                join=True,
            )
        finally:
            os.environ.pop("DDP_INIT_FILE", None)
            if os.path.isfile(init_file):
                os.remove(init_file)


if __name__ == '__main__':
    main()
