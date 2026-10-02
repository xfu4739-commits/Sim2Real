import tempfile
import unittest
from pathlib import Path

import torch

from lib.training_logger import (
    TrainingLogger,
    build_checkpoint,
    load_checkpoint,
    plot_training_history,
    save_checkpoint,
)
from train import _runtime_device, reduce_values


class TrainingRuntimeTest(unittest.TestCase):
    def test_logger_uses_run_directory_and_generates_curves(self):
        metrics = {
            'epoch': 1, 'lr': 0.0001,
            'train_loss': 0.8, 'train_dice_loss': 0.6, 'train_focal_loss': 0.2,
            'val_loss': 0.9, 'val_dice_loss': 0.7, 'val_focal_loss': 0.2,
            'iou_fg': 0.4, 'iou_bg': 0.8, 'miou': 0.6, 'f1': 0.5,
            'precision': 0.55, 'recall': 0.45, 'auprc': 0.52,
            'pixel_acc': 0.75, 'train_time_sec': 1.0, 'val_time_sec': 0.5,
            'epoch_time_sec': 1.5, 'samples_per_sec': 2.0, 'max_mem_mb': 100.0,
            'best_iou': 0.4, 'best_map': 0.52, 'timestamp': '2026-01-01 00:00:00',
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = TrainingLogger(
                temp_dir, config_path='does-not-exist.yaml', run_id='autodl-test',
                save_batch_metrics=True, tensorboard=True,
            )
            logger.log_batch({
                'epoch': 1, 'batch': 1, 'total_batches': 1, 'global_step': 1,
                'loss': 0.8, 'dice_loss': 0.6, 'focal_loss': 0.2,
                'learning_rate': 0.0001, 'batch_time_sec': 0.1,
                'samples_per_sec': 10.0, 'timestamp': '2026-01-01 00:00:00',
            })
            logger.log_epoch(metrics)
            logger.close()

            run_dir = Path(temp_dir) / 'autodl-test'
            self.assertTrue((run_dir / 'metrics.csv').is_file())
            self.assertTrue((run_dir / 'batch_metrics.csv').is_file())
            self.assertTrue((run_dir / 'loss_curves.png').is_file())
            self.assertTrue((run_dir / 'metrics_curves.png').is_file())
            self.assertTrue(any((run_dir / 'tensorboard').glob('events.out.tfevents.*')))
            paths = plot_training_history(run_dir / 'metrics.csv')
            self.assertTrue(all(Path(path).is_file() for path in paths))

    def test_runtime_device_falls_back_to_cpu_without_cuda(self):
        if torch.cuda.is_available():
            self.skipTest('CUDA is available in this environment')
        self.assertEqual(_runtime_device().type, 'cpu')

    def test_reduce_values_works_without_process_group(self):
        self.assertEqual(reduce_values([1.5, 2], torch.device('cpu')), [1.5, 2.0])

    def test_plain_model_checkpoint_round_trip_with_safe_loading(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        checkpoint = build_checkpoint(
            3, model, optimizer, scheduler, (0.4, 0.5), {'name': 'test'}
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'checkpoint.pth'
            save_checkpoint(path, checkpoint)
            restored = torch.nn.Linear(2, 1)
            restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.01)
            restored_scheduler = torch.optim.lr_scheduler.StepLR(
                restored_optimizer, step_size=1
            )
            start_epoch, best_metrics = load_checkpoint(
                path,
                restored,
                restored_optimizer,
                restored_scheduler,
                torch.device('cpu'),
            )

        self.assertEqual(start_epoch, 4)
        self.assertEqual(best_metrics, (0.4, 0.5))
        for expected, actual in zip(model.parameters(), restored.parameters()):
            self.assertTrue(torch.equal(expected, actual))


if __name__ == '__main__':
    unittest.main()
