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

    def test_figure2_style_ignores_burn_scar_and_config_orange(self):
        module = importlib.import_module("dataset.satellite_fusion")
        base = np.full((8, 8, 3), 40, dtype=np.uint8)
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:6, 2:6] = 1

        fused = module._fuse.apply_style(
            base,
            mask,
            style="figure2",
            hotspots=True,
            fire_color_bgr=(0, 80, 255),
        )

        self.assertTrue(np.array_equal(fused[3, 3], np.array([10, 111, 189], dtype=np.uint8)))
        self.assertTrue(np.array_equal(fused[0, 0], base[0, 0]))
        self.assertTrue(np.array_equal(fused[1, 3], np.array([10, 18, 191], dtype=np.uint8)))


if __name__ == "__main__":
    unittest.main()
