import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import tifffile as tif

from dataset.common import FUEL_FILES, TOPOGRAPHY_FILES, VEGETATION_FILES
from dataset.sim_dataset import SimFireDataset


def _write_minimal_scene(scene_root: Path) -> None:
    cv2.imwrite(
        str(scene_root / "satellite_image.png"),
        np.zeros((8, 8, 3), dtype=np.uint8),
    )
    for directory, filenames in (
        ("Topography_Map", TOPOGRAPHY_FILES),
        ("Vegetation_Map", VEGETATION_FILES),
        ("Fuel_Map", FUEL_FILES),
    ):
        target = scene_root / directory
        target.mkdir(parents=True, exist_ok=True)
        for name in filenames:
            tif.imwrite(str(target / name), np.ones((4, 4), dtype=np.float32))

    event_dir = scene_root / "Satellite_Images_Mask" / "evt1"
    event_dir.mkdir(parents=True, exist_ok=True)
    for index in range(1, 7):
        cv2.imwrite(
            str(event_dir / f"out{index}.jpg"),
            np.zeros((8, 8, 3), dtype=np.uint8),
        )

    wxs_dir = scene_root / "Weather_Data"
    wxs_dir.mkdir(parents=True, exist_ok=True)


def _minimal_cfg(scene_root: Path) -> dict:
    root = str(scene_root)
    return {
        "sim_root_dir": root,
        "train_dir": "Satellite_Images_Mask",
        "val_dir": "Satellite_Images_Mask",
        "trainwxs_dir": "Weather_Data",
        "valwxs_dir": "Weather_Data",
        "topo_dir": "Topography_Map",
        "vege_dir": "Vegetation_Map",
        "fuel_dir": "Fuel_Map",
        "sim_satellite_file": "satellite_image.png",
        "satellite_fusion": {"enabled": False},
        "input_length": 3,
        "img_size": [8, 8],
        "reverse": False,
        "num_workers": 0,
        "environment_cache_size": 2,
        "sim_split": {"enabled": False},
    }


class SimWeatherKeyTest(unittest.TestCase):
    def test_missing_weather_row_skips_sequence_without_key_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_root = Path(tmp)
            _write_minimal_scene(scene_root)
            # Frames 1-2 only; sequence needs input frames 1,2,3 -> key for 3 is missing.
            (scene_root / "Weather_Data" / "evt1.txt").write_text(
                "1 2 3 4 5 6 7 8 9 10\n2 2 3 4 5 6 7 8 9 10\n",
                encoding="utf-8",
            )
            cfg = _minimal_cfg(scene_root)
            with patch.object(
                SimFireDataset,
                "extract_sequences",
                return_value=[[1, 2, 3, 4, 5, 6]],
            ):
                dataset = SimFireDataset(cfg, training=True)

            self.assertEqual(len(dataset), 0)

    def test_complete_weather_rows_keep_sequence(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_root = Path(tmp)
            _write_minimal_scene(scene_root)
            lines = [" ".join(str(v) for v in range(1, 11)) for _ in range(6)]
            (scene_root / "Weather_Data" / "evt1.txt").write_text(
                "\n".join(lines) + "\n",
                encoding="utf-8",
            )
            cfg = _minimal_cfg(scene_root)
            with patch.object(
                SimFireDataset,
                "extract_sequences",
                return_value=[[1, 2, 3, 4, 5, 6]],
            ):
                dataset = SimFireDataset(cfg, training=True)

            self.assertGreater(len(dataset), 0)


if __name__ == "__main__":
    unittest.main()
