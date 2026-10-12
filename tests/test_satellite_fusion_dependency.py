import importlib
from pathlib import Path
import unittest

import numpy as np


class SatelliteFusionDependencyTest(unittest.TestCase):
    def test_runtime_module_imports_only_tracked_dataset_code(self):
        module = importlib.import_module("dataset.satellite_fusion")
        helper_path = Path(module._fuse.__file__).resolve()

        self.assertEqual(helper_path.name, "satellite_fusion_ops.py")
        self.assertEqual(helper_path.parent.name, "dataset")
        self.assertNotIn("test", helper_path.parts)

    def test_overlay_fusion_runs_without_local_test_helpers(self):
        module = importlib.import_module("dataset.satellite_fusion")
        base = np.zeros((8, 8, 3), dtype=np.uint8)
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:6, 2:6] = 1

        fused = module._fuse.apply_style(base, mask, style="overlay")

        self.assertEqual(fused.shape, base.shape)
        self.assertGreater(int(fused[3, 3, 2]), 0)


if __name__ == "__main__":
    unittest.main()
