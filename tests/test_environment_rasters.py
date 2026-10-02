import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import tifffile

from dataset.common import (
    CATEGORY_CARDINALITY,
    EnvironmentRasterCache,
    FUEL_FILES,
    TOPOGRAPHY_FILES,
    VEGETATION_FILES,
    _encode_categorical,
    _normalize_continuous,
    _read_raster,
    _scale_percent,
    filter_complete_environment_scenes,
    missing_environment_files,
)
from model.S2RFireTr import EnvironmentCategoryEncoder


class EnvironmentRasterTest(unittest.TestCase):
    def test_raster_masks_negative_sentinels_without_matching_nodata_tag(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'category.tif'
            tifffile.imwrite(path, np.array([[1, -1111], [-32768, 2]], dtype=np.int16))
            array, valid = _read_raster(path)

        self.assertEqual(array.tolist(), [[1, -1111], [-32768, 2]])
        self.assertEqual(valid.tolist(), [[True, False], [False, True]])

    def test_continuous_normalization_ignores_nodata(self):
        array = np.array([[1, 2], [9999, 3]], dtype=np.float32)
        valid = np.array([[True, True], [False, True]])

        output = _normalize_continuous(array, valid, Path("continuous.tif"))

        self.assertAlmostEqual(output[valid].mean().item(), 0.0, places=6)
        self.assertEqual(output[1, 0].item(), 0.0)

    def test_categorical_encoding_reserves_zero_for_nodata(self):
        array = np.array([[0, 99], [-32768, 7011]], dtype=np.int32)
        valid = np.array([[True, True], [False, True]])

        output = _encode_categorical(array, valid, Path("category.tif"))

        self.assertEqual(output.tolist(), [[1.0, 100.0], [0.0, 7012.0]])

    def test_percent_scaling_excludes_land_cover_codes(self):
        array = np.array([[0, 50], [100, 180]], dtype=np.int16)
        valid = np.ones((2, 2), dtype=bool)

        output = _scale_percent(array, valid)

        self.assertEqual(output.tolist(), [[0.0, 0.5], [1.0, 0.0]])

    def test_missing_environment_files_are_not_silently_imputed(self):
        missing = missing_environment_files(
            "does-not-exist", "Topography_Map", "Vegetation_Map", "Fuel_Map"
        )

        self.assertEqual(
            len(missing),
            len(TOPOGRAPHY_FILES) + len(VEGETATION_FILES) + len(FUEL_FILES),
        )
        with self.assertRaisesRegex(FileNotFoundError, "no scene has"):
            filter_complete_environment_scenes(
                ["does-not-exist"],
                "Topography_Map",
                "Vegetation_Map",
                "Fuel_Map",
            )

    def test_category_encoder_keeps_nodata_zero_and_learns_valid_codes(self):
        encoder = EnvironmentCategoryEncoder(
            channel_count=1,
            categorical_indices=(0,),
            cardinality=CATEGORY_CARDINALITY,
        )
        rasters = torch.tensor([[[[0.0, 2.0], [2.0, 7012.0]]]])

        output = encoder(rasters, "vegetation")
        output.sum().backward()

        self.assertEqual(output[0, 0, 0, 0].item(), 0.0)
        self.assertEqual(output[0, 0, 0, 1].item(), output[0, 0, 1, 0].item())
        self.assertEqual(
            encoder.embeddings[0].weight.grad[0].abs().sum().item(), 0.0
        )
        self.assertGreater(
            encoder.embeddings[0].weight.grad[2].abs().sum().item(), 0.0
        )

    def test_environment_cache_is_bounded_and_reuses_recent_scene(self):
        cache = EnvironmentRasterCache(
            "topography", "vegetation", "fuel", (8, 8), max_scenes=2
        )
        with patch(
            "dataset.common.load_environment",
            side_effect=lambda scene, *_: (scene, scene, scene),
        ) as load:
            first = cache.get("scene-a")
            self.assertIs(first, cache.get("scene-a"))
            cache.get("scene-b")
            cache.get("scene-c")
            cache.get("scene-a")

        self.assertEqual(load.call_count, 4)

    def test_category_encoder_rejects_interpolated_ids(self):
        encoder = EnvironmentCategoryEncoder(1, (0,), CATEGORY_CARDINALITY)

        with self.assertRaisesRegex(ValueError, "non-integer"):
            encoder(torch.tensor([[[[1.5]]]]), "vegetation")


if __name__ == "__main__":
    unittest.main()
