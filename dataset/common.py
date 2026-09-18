from pathlib import Path

import cv2
import torch
import tifffile as tif


TOPOGRAPHY_FILES = ("Aspect.tif", "Elevation.tif", "Slope.tif")
VEGETATION_FILES = (
    "Existing_Vegetation_Cover.tif",
    "Existing_Vegetation_Height.tif",
    "Existing_Vegetation_Type.tif",
    "Succession_Classes.tif",
    "Vegetation_Condition_Class.tif",
    "Vegetation_Departure.tif",
)
FUEL_FILES = (
    "Canopy_Base_Height.tif",
    "Canopy_Bulk_Density.tif",
    "Canopy_Cover.tif",
    "Canopy_Height.tif",
    "FBFM13.tif",
    "FBFM40.tif",
    "Fuel_Disturbance.tif",
    "Fuel_Vegetation_Cover.tif",
    "Fuel_Vegetation_Height.tif",
    "Fuel_Vegetation_Type.tif",
)


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


def _load_map_directory(scene_root, directory, filenames, img_size):
    root = Path(scene_root) / directory
    channels = []
    for filename in filenames:
        path = root / filename
        if not path.is_file():
            channels.append(torch.zeros(1, *img_size, dtype=torch.float32))
            continue
        try:
            array = tif.imread(path)
        except ValueError as error:
            if "imagecodecs" not in str(error):
                raise
            array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if array is None:
                raise RuntimeError(f"failed to decode compressed TIFF: {path}") from error
        array = cv2.resize(array, tuple(img_size), interpolation=cv2.INTER_NEAREST)
        channel = torch.from_numpy(array).float()
        if path.stem.lower() == "aspect":
            channel = torch.sin(torch.deg2rad(channel))
        mean = channel.mean()
        std = channel.std()
        if std > 0:
            channel = (channel - mean) / std
        else:
            channel = channel - mean
        channels.append(channel.unsqueeze(0))
    return torch.cat(channels, dim=0)


def load_environment(scene_root, topo_dir, vege_dir, fuel_dir, img_size):
    """Load the official 3/6/10 environmental raster groups."""
    topography = _load_map_directory(
        scene_root, topo_dir, TOPOGRAPHY_FILES, img_size
    )
    vegetation = _load_map_directory(
        scene_root, vege_dir, VEGETATION_FILES, img_size
    )
    fuel = _load_map_directory(scene_root, fuel_dir, FUEL_FILES, img_size)
    return topography, vegetation, fuel
