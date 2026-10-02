import tempfile
import unittest
from pathlib import Path

import torch

from lib.training_logger import build_checkpoint, load_checkpoint, save_checkpoint
from train import _runtime_device, reduce_values


class TrainingRuntimeTest(unittest.TestCase):
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
