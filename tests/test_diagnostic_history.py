import tempfile
import unittest
from pathlib import Path

from lib.plotting import plot_horizon_trends
from tools.plot_fire_diagnostics import epoch_checkpoints, summarize


class DiagnosticHistoryTest(unittest.TestCase):
    def test_directory_selects_only_numbered_epochs_in_numeric_order(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ('epoch_10.pth', 'epoch_2.pth', 'epoch_1.pth',
                         'best_iou.pth', 'last.pth', 'epoch_bad.pth'):
                (Path(directory) / name).touch()
            selected = epoch_checkpoints(directory)
            self.assertEqual([Path(path).name for path in selected],
                             ['epoch_1.pth', 'epoch_2.pth', 'epoch_10.pth'])

    def test_empty_checkpoint_directory_does_not_evaluate_random_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                epoch_checkpoints(directory)

    def test_epoch_curves_render_actual_horizon_summaries(self):
        sample = dict(step=1, tp=8, fp=2, fn=4, gt_pixels=12,
                      pred_pixels=10, union=14, growth_tp=2, growth_fp=1,
                      growth_fn=2, growth_gt_pixels=4, growth_pred_pixels=3,
                      growth_union=5)
        horizon = summarize([sample], 'step')[0]
        rows = [dict(epoch=epoch, checkpoint_label=f'epoch_{epoch}', **horizon)
                for epoch in (1, 2, 10)]
        with tempfile.TemporaryDirectory() as directory:
            plot_horizon_trends(Path(directory), rows)
            self.assertTrue((Path(directory) / 'epoch_horizon_trends.png').is_file())


if __name__ == '__main__':
    unittest.main()
