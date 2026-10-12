from collections import OrderedDict
from dataclasses import dataclass
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import tifffile as tif


# Some released int16 GeoTIFFs store the textual NoData value 32767/32768 in a
# way that tifffile logs as a cast warning even though pixels decode correctly.
# Decoding failures still raise exceptions; only noisy metadata warnings hide.
# Use the standard logger API across old and current tifffile releases.
logging.getLogger("tifffile").setLevel(logging.ERROR)


@dataclass(frozen=True)
class RasterSpec:
    filename: str
    kind: str


# The paper describes three topography rasters. Aspect is circular, so it is
# represented by sine and cosine features (four model channels in total).
TOPOGRAPHY_SPECS = (
    RasterSpec("Aspect.tif", "circular"),
    RasterSpec("Elevation.tif", "continuous"),
    RasterSpec("Slope.tif", "continuous"),
)

# LANDFIRE vegetation products are coded classes/bins. Treating their numeric
# IDs as continuous quantities invents an ordering that the data do not have.
VEGETATION_SPECS = (
    RasterSpec("Existing_Vegetation_Cover.tif", "categorical"),
    RasterSpec("Existing_Vegetation_Height.tif", "categorical"),
    RasterSpec("Existing_Vegetation_Type.tif", "categorical"),
    RasterSpec("Succession_Classes.tif", "categorical"),
    RasterSpec("Vegetation_Condition_Class.tif", "categorical"),
    # Codes 0-100 are physical percent departure; 101+ are non-applicable
    # land-cover classes and are mapped to zero in this continuous feature.
    RasterSpec("Vegetation_Departure.tif", "percent"),
)

# Canopy measurements are physical continuous quantities; surface-fuel,
# disturbance and fuel-vegetation products are coded classes/bins.
FUEL_SPECS = (
    RasterSpec("Canopy_Base_Height.tif", "continuous"),
    RasterSpec("Canopy_Bulk_Density.tif", "continuous"),
    RasterSpec("Canopy_Cover.tif", "continuous"),
    RasterSpec("Canopy_Height.tif", "continuous"),
    RasterSpec("FBFM13.tif", "categorical"),
    RasterSpec("FBFM40.tif", "categorical"),
    RasterSpec("Fuel_Disturbance.tif", "categorical"),
    RasterSpec("Fuel_Vegetation_Cover.tif", "categorical"),
    RasterSpec("Fuel_Vegetation_Height.tif", "categorical"),
    RasterSpec("Fuel_Vegetation_Type.tif", "categorical"),
)

TOPOGRAPHY_FILES = tuple(spec.filename for spec in TOPOGRAPHY_SPECS)
VEGETATION_FILES = tuple(spec.filename for spec in VEGETATION_SPECS)
FUEL_FILES = tuple(spec.filename for spec in FUEL_SPECS)

TOPOGRAPHY_OUTPUT_CHANNELS = 4
VEGETATION_OUTPUT_CHANNELS = len(VEGETATION_SPECS)
FUEL_OUTPUT_CHANNELS = len(FUEL_SPECS)
CATEGORY_CARDINALITY = 65537  # 0=NoData, raw uint16-compatible code n -> n+1


def resolve_scene_roots(root_config, required_dirs):
    """Resolve either explicit scene roots or a parent containing scene folders."""
    configured = [root_config] if isinstance(root_config, str) else list(root_config)
    roots = []
    for value in configured:
        path = Path(value).expanduser()
        if all((path / name).is_dir() for name in required_dirs):
            roots.append(path)
            continue
        if path.is_dir():
            roots.extend(
                child for child in sorted(path.iterdir())
                if child.is_dir() and all((child / name).is_dir() for name in required_dirs)
            )
    if not roots:
        raise FileNotFoundError(
            f"no dataset scene roots found under {configured}; "
            f"each scene must contain {list(required_dirs)}"
        )
    return [str(path) for path in roots]


def missing_environment_files(scene_root, topo_dir, vege_dir, fuel_dir):
    groups = (
        (topo_dir, TOPOGRAPHY_SPECS),
        (vege_dir, VEGETATION_SPECS),
        (fuel_dir, FUEL_SPECS),
    )
    scene = Path(scene_root)
    return [
        str(Path(directory) / spec.filename)
        for directory, specs in groups
        for spec in specs
        if not (scene / directory / spec.filename).is_file()
    ]


def filter_complete_environment_scenes(
    scene_roots, topo_dir, vege_dir, fuel_dir, warning_limit=20
):
    """Keep only scenes with every paper/model environmental input.

    Missing rasters used to be silently replaced with zero tensors. That made a
    missing categorical layer indistinguishable from a real category/value and
    violated the paper's complete multi-modal input assumption.
    """
    complete = []
    skipped = []
    for scene_root in scene_roots:
        missing = missing_environment_files(
            scene_root, topo_dir, vege_dir, fuel_dir
        )
        if missing:
            skipped.append((scene_root, missing))
        else:
            complete.append(scene_root)
    for scene_root, missing in skipped[:warning_limit]:
        print(
            f"Warning: skipping scene with {len(missing)} missing environment "
            f"rasters: {scene_root} ({', '.join(missing)})"
        )
    if len(skipped) > warning_limit:
        print(
            f"Warning: {len(skipped) - warning_limit} additional incomplete "
            "scenes omitted; run tools/validate_dataset.py for the full report"
        )
    if not complete:
        raise FileNotFoundError(
            "no scene has the complete required environment raster set; "
            "run tools/validate_dataset.py for a detailed report"
        )
    return complete


def _read_raster(path):
    nodata = None
    try:
        with tif.TiffFile(path) as image:
            page = image.pages[0]
            tag = page.tags.get(42113)
            if tag is not None:
                try:
                    value = tag.value
                    if isinstance(value, bytes):
                        value = value.decode("ascii", errors="strict")
                    nodata = float(str(value).strip("\x00"))
                except (TypeError, ValueError, UnicodeError):
                    nodata = None
            array = page.asarray()
    except ValueError as error:
        if "imagecodecs" not in str(error):
            raise
        array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if array is None:
            raise RuntimeError(f"failed to decode compressed TIFF: {path}") from error
    if array is None or array.ndim != 2:
        shape = None if array is None else array.shape
        raise ValueError(f"environment raster must be 2-D: {path}, shape={shape}")

    if np.issubdtype(array.dtype, np.floating):
        valid = np.isfinite(array)
    else:
        valid = np.ones(array.shape, dtype=bool)
    if nodata is not None:
        valid &= array != nodata
    # Some released rasters have a GDAL_NODATA tag that disagrees with the
    # signed pixels (for example tag=32768 but pixels=-32768), while others use
    # -1111 without a tag. These negative values cannot be valid category IDs
    # or physical inputs, so mask them even when a metadata tag is present.
    for sentinel in (-32768, -9999, -1111):
        valid &= array != sentinel
    if nodata is None:
        valid &= array != 32767
    return array, valid


def _resize(array, valid, img_size):
    height, width = (int(img_size[0]), int(img_size[1]))
    cv_size = (width, height)
    resized = cv2.resize(array, cv_size, interpolation=cv2.INTER_NEAREST)
    resized_valid = cv2.resize(
        valid.astype(np.uint8), cv_size, interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    return resized, resized_valid


def _normalize_continuous(array, valid, path):
    values = torch.from_numpy(np.asarray(array)).float()
    mask = torch.from_numpy(np.asarray(valid)).bool()
    if not torch.any(mask):
        raise ValueError(f"environment raster contains no valid pixels: {path}")
    valid_values = values[mask]
    mean = valid_values.mean()
    std = valid_values.std(unbiased=False)
    output = torch.zeros_like(values)
    if torch.isfinite(std) and std > 0:
        output[mask] = (valid_values - mean) / std
    else:
        output[mask] = valid_values - mean
    return output


def _encode_categorical(array, valid, path):
    values = torch.from_numpy(np.asarray(array)).float()
    mask = torch.from_numpy(np.asarray(valid)).bool()
    if not torch.any(mask):
        raise ValueError(f"categorical raster contains no valid pixels: {path}")
    valid_values = values[mask]
    if not torch.allclose(valid_values, valid_values.round()):
        raise ValueError(f"categorical raster contains non-integer codes: {path}")
    if torch.any(valid_values < 0) or torch.any(valid_values > 65535):
        minimum = float(valid_values.min())
        maximum = float(valid_values.max())
        raise ValueError(
            f"categorical code outside supported [0, 65535] range: {path}, "
            f"min={minimum}, max={maximum}"
        )
    output = torch.zeros_like(values)
    output[mask] = valid_values.round() + 1
    return output


def _scale_percent(array, valid):
    values = torch.from_numpy(np.asarray(array)).float()
    mask = torch.from_numpy(np.asarray(valid)).bool()
    mask &= values >= 0
    mask &= values <= 100
    output = torch.zeros_like(values)
    output[mask] = values[mask] / 100.0
    return output


def _load_map_directory(scene_root, directory, specs, img_size):
    root = Path(scene_root) / directory
    channels = []
    for spec in specs:
        path = root / spec.filename
        if not path.is_file():
            raise FileNotFoundError(
                f"missing required environment raster: {path}; "
                "run tools/validate_dataset.py for a detailed report"
            )
        array, valid = _read_raster(path)
        array, valid = _resize(array, valid, img_size)
        if spec.kind == "continuous":
            channels.append(_normalize_continuous(array, valid, path).unsqueeze(0))
        elif spec.kind == "categorical":
            channels.append(_encode_categorical(array, valid, path).unsqueeze(0))
        elif spec.kind == "percent":
            channels.append(_scale_percent(array, valid).unsqueeze(0))
        elif spec.kind == "circular":
            radians = np.deg2rad(array.astype(np.float32))
            channels.append(
                _normalize_continuous(np.sin(radians), valid, path).unsqueeze(0)
            )
            channels.append(
                _normalize_continuous(np.cos(radians), valid, path).unsqueeze(0)
            )
        else:
            raise ValueError(f"unsupported raster kind {spec.kind!r}: {path}")
    return torch.cat(channels, dim=0)


def load_environment(scene_root, topo_dir, vege_dir, fuel_dir, img_size):
    """Load, mask and type-aware encode the required environment rasters."""
    topography = _load_map_directory(
        scene_root, topo_dir, TOPOGRAPHY_SPECS, img_size
    )
    vegetation = _load_map_directory(
        scene_root, vege_dir, VEGETATION_SPECS, img_size
    )
    fuel = _load_map_directory(scene_root, fuel_dir, FUEL_SPECS, img_size)
    return topography, vegetation, fuel


class EnvironmentRasterCache:
    """Small per-process LRU cache for scene-level environment tensors.

    Preloading hundreds of real scenes consumes several GB before evaluation
    starts, and Windows DataLoader workers may duplicate that memory. A bounded
    cache keeps memory predictable while retaining reuse within each scene.
    """

    def __init__(
        self, topo_dir, vege_dir, fuel_dir, img_size, max_scenes=8
    ):
        if max_scenes < 1:
            raise ValueError("environment_cache_size must be at least 1")
        self.topo_dir = topo_dir
        self.vege_dir = vege_dir
        self.fuel_dir = fuel_dir
        self.img_size = tuple(img_size)
        self.max_scenes = int(max_scenes)
        self._cache = OrderedDict()

    def get(self, scene_root):
        key = str(scene_root)
        if key in self._cache:
            value = self._cache.pop(key)
            self._cache[key] = value
            return value
        value = load_environment(
            key,
            self.topo_dir,
            self.vege_dir,
            self.fuel_dir,
            self.img_size,
        )
        self._cache[key] = value
        if len(self._cache) > self.max_scenes:
            self._cache.popitem(last=False)
        return value
