#!/usr/bin/env python3
"""Validate Sim2Real-Fire data against this repository and the paper.

The validator is read-only.  It checks every scene and event structurally, and
can optionally decode a sample of files or every image/raster.  Exit codes:

* 0: no hard errors (warnings may still exist)
* 1: the dataset contains hard errors
* 2: the validator itself could not start (bad arguments/dependencies)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


if os.name == "nt":
    # Keep Chinese diagnostics readable in Windows Terminal/PowerShell and in
    # redirected CI logs, regardless of the active legacy code page.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


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
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
SCENE_RE = re.compile(r"^\d{4}_\d{5}$")
SIM_FRAME_RE = re.compile(r"^out([1-9]\d*)\.jpg$")


@dataclass(frozen=True)
class Rules:
    sim_mask_dir: str = "Satellite_Images_Mask"
    real_mask_dir: str = "Satellite_Images_Mask"
    real_satellite_dir: str = "Satellite_Images"
    weather_dir: str = "Weather_Data"
    topography_dir: str = "Topography_Map"
    vegetation_dir: str = "Vegetation_Map"
    fuel_dir: str = "Fuel_Map"
    sim_satellite_file: str = "satellite_image.png"
    input_length: int = 3
    image_size: tuple[int, int] = (256, 256)
    modal_channel: int = 29


@dataclass(frozen=True)
class Issue:
    severity: str
    code: str
    path: str
    message: str


class Results:
    def __init__(self, detail_limit: int) -> None:
        self.detail_limit = detail_limit
        self.issues: list[Issue] = []
        self.issue_counts: Counter[str] = Counter()
        self.stats: Counter[str] = Counter()

    def add(self, severity: str, code: str, path: Path, message: str) -> None:
        key = f"{severity}:{code}"
        self.issue_counts[key] += 1
        if len(self.issues) < self.detail_limit:
            self.issues.append(Issue(severity, code, str(path), message))

    def error(self, code: str, path: Path, message: str) -> None:
        self.add("ERROR", code, path, message)

    def warning(self, code: str, path: Path, message: str) -> None:
        self.add("WARNING", code, path, message)

    @property
    def error_count(self) -> int:
        return sum(v for k, v in self.issue_counts.items() if k.startswith("ERROR:"))

    @property
    def warning_count(self) -> int:
        return sum(v for k, v in self.issue_counts.items() if k.startswith("WARNING:"))


def load_rules(config_path: Path | None, results: Results) -> Rules:
    if config_path is None:
        return Rules()
    if not config_path.is_file():
        results.error("CONFIG_NOT_FOUND", config_path, "配置文件不存在")
        return Rules()
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("读取 --config 需要 PyYAML；请安装 requirements.txt") from exc

    with config_path.open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream) or {}
    dataset = cfg.get("dataset") or {}
    model = cfg.get("FireTr") or {}
    encoder = model.get("Encoder") or {}

    input_length = int(dataset.get("input_length", 3))
    model_input_length = int(model.get("input_length", input_length))
    if input_length != model_input_length:
        results.error(
            "CONFIG_SEQUENCE_MISMATCH",
            config_path,
            f"dataset.input_length={input_length}，但 FireTr.input_length={model_input_length}",
        )

    image_size = tuple(int(v) for v in dataset.get("img_size", (256, 256)))
    model_image_size = tuple(int(v) for v in model.get("img_size", image_size))
    if len(image_size) != 2 or any(v <= 0 for v in image_size):
        results.error("CONFIG_IMAGE_SIZE", config_path, f"无效 img_size: {image_size}")
        image_size = (256, 256)
    if image_size != model_image_size:
        results.error(
            "CONFIG_IMAGE_SIZE_MISMATCH",
            config_path,
            f"dataset.img_size={image_size}，但 FireTr.img_size={model_image_size}",
        )

    modal_channel = int(encoder.get("modal_channel", 29))
    expected_channels = 4 + 6 + 10 + input_length * 3
    if modal_channel != expected_channels:
        results.error(
            "CONFIG_MODAL_CHANNEL",
            config_path,
            f"modal_channel={modal_channel}，代码按 4+6+10+{input_length}*3 需要 {expected_channels}",
        )

    return Rules(
        sim_mask_dir=str(dataset.get("train_dir", "Satellite_Images_Mask")),
        real_mask_dir=str(dataset.get("test_dir", "Satellite_Images_Mask")),
        real_satellite_dir=str(dataset.get("real_satellite_dir", "Satellite_Images")),
        weather_dir=str(dataset.get("trainwxs_dir", dataset.get("testwxs_dir", "Weather_Data"))),
        topography_dir=str(dataset.get("topo_dir", "Topography_Map")),
        vegetation_dir=str(dataset.get("vege_dir", "Vegetation_Map")),
        fuel_dir=str(dataset.get("fuel_dir", "Fuel_Map")),
        sim_satellite_file=str(dataset.get("sim_satellite_file", "satellite_image.png")),
        input_length=input_length,
        image_size=image_size,
        modal_channel=modal_channel,
    )


def import_decoders():
    try:
        import cv2
        import numpy as np
        import tifffile
    except ImportError as exc:
        raise RuntimeError(
            "sample/full 校验需要 opencv-python、numpy 和 tifffile；请安装 requirements.txt"
        ) from exc
    return cv2, np, tifffile


def selected_for_decode(files: Sequence[Path], level: str) -> Iterable[Path]:
    if level == "structure" or not files:
        return ()
    if level == "full" or len(files) <= 2:
        return files
    return (files[0], files[-1])


def nonempty_file(path: Path, results: Results, code: str) -> bool:
    try:
        if path.stat().st_size > 0:
            return True
    except OSError as exc:
        results.error(code, path, f"无法读取文件状态: {exc}")
        return False
    results.error(code, path, "文件大小为 0")
    return False


def decode_image(path: Path, results: Results, decoders) -> tuple[int, int] | None:
    cv2, _, _ = decoders
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    results.stats["images_decoded"] += 1
    if image is None:
        results.error("IMAGE_UNREADABLE", path, "OpenCV 无法解码")
        return None
    if image.ndim != 3 or image.shape[2] != 3 or image.shape[0] <= 0 or image.shape[1] <= 0:
        results.error("IMAGE_FORMAT", path, f"代码需要可转灰度的三通道图像，实际 shape={image.shape}")
        return None
    return int(image.shape[0]), int(image.shape[1])


def decode_raster(path: Path, results: Results, decoders) -> tuple[int, int] | None:
    cv2, np, tifffile = decoders
    try:
        array = tifffile.imread(path)
    except ValueError as exc:
        if "imagecodecs" not in str(exc):
            results.error("TIFF_UNREADABLE", path, f"tifffile 解码失败: {exc}")
            return None
        array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    except Exception as exc:  # corrupt TIFFs can raise several backend exceptions
        results.error("TIFF_UNREADABLE", path, f"TIFF 解码失败: {exc}")
        return None

    results.stats["rasters_decoded"] += 1
    if array is None:
        results.error("TIFF_UNREADABLE", path, "tifffile 和 OpenCV 都无法解码")
        return None
    if array.ndim != 2 or array.shape[0] <= 0 or array.shape[1] <= 0:
        results.error("TIFF_FORMAT", path, f"代码要求单通道二维栅格，实际 shape={array.shape}")
        return None
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        results.error("TIFF_NONFINITE", path, "栅格包含 NaN 或 Inf")

    nodata = None
    try:
        with tifffile.TiffFile(path) as tif:
            tag = tif.pages[0].tags.get(42113)
            nodata = tag.value if tag is not None else None
    except Exception:
        pass
    if nodata is not None:
        try:
            value = float(str(nodata).strip("\x00"))
            count = int(np.count_nonzero(array == value))
            if count:
                results.warning(
                    "TIFF_NODATA_PRESENT",
                    path,
                    f"包含 {count} 个 NoData={nodata} 像素；加载器会将其映射到保留的 NoData 值",
                )
        except (TypeError, ValueError):
            results.warning("TIFF_NODATA_TAG", path, f"无法解析 GDAL_NODATA={nodata!r}")
    return int(array.shape[0]), int(array.shape[1])


def validate_static_maps(
    scene: Path, rules: Rules, results: Results, level: str, decoders
) -> bool:
    groups = (
        (rules.topography_dir, TOPOGRAPHY_FILES),
        (rules.vegetation_dir, VEGETATION_FILES),
        (rules.fuel_dir, FUEL_FILES),
    )
    valid = True
    shapes: dict[str, tuple[int, int]] = {}
    for directory, filenames in groups:
        root = scene / directory
        if not root.is_dir():
            results.error("MISSING_DIRECTORY", root, "缺少代码要求的环境模态目录")
            valid = False
            continue
        for filename in filenames:
            path = root / filename
            if not path.is_file():
                results.error(
                    "MISSING_RASTER",
                    path,
                    "缺少论文模态/代码通道；当前加载器会静默填零，不能视为完整数据",
                )
                valid = False
                continue
            if not nonempty_file(path, results, "EMPTY_RASTER"):
                valid = False
                continue
            results.stats["rasters_found"] += 1
            if level != "structure":
                shape = decode_raster(path, results, decoders)
                if shape is None:
                    valid = False
                else:
                    shapes[str(path.relative_to(scene))] = shape
    if shapes and len(set(shapes.values())) > 1:
        summary = Counter(shapes.values()).most_common(4)
        results.warning(
            "STATIC_SHAPE_MISMATCH",
            scene,
            f"静态模态原始尺寸不一致 {summary}；代码会缩放，但论文要求空间对齐",
        )
    return valid


def parse_weather_line(
    line: str, path: Path, line_number: int, results: Results, strict_single_space: bool
) -> list[str] | None:
    stripped = line.strip()
    if not stripped:
        if strict_single_space:
            results.error("WEATHER_EMPTY_LINE", path, f"第 {line_number} 行为空，会破坏模拟帧与天气行号对应")
        else:
            results.warning("WEATHER_EMPTY_LINE", path, f"第 {line_number} 行为空，真实数据加载器会忽略")
        return None
    if stripped.startswith("#"):
        results.warning("WEATHER_EXCLUDED", path, f"第 {line_number} 行以 # 开头，会使相关模拟样本被排除")
        return None
    values = stripped.split(" ") if strict_single_space else stripped.split()
    if "" in values:
        results.error(
            "WEATHER_SPACING",
            path,
            f"第 {line_number} 行含连续空格；模拟加载器 split(' ') 后会产生空字段",
        )
        return None
    if len(values) != 10:
        results.error(
            "WEATHER_COLUMNS",
            path,
            f"第 {line_number} 行应为日期/时刻 4 列 + 气象 6 列，共 10 列，实际 {len(values)} 列",
        )
        return None
    try:
        [float(value) for value in values]
        year, month, day = (int(values[i]) for i in range(3))
        datetime(year, month, day)
        hhmm = values[3]
        if not re.fullmatch(r"\d{4}", hhmm):
            raise ValueError("时刻不是四位 HHMM")
        hour, minute = int(hhmm[:2]), int(hhmm[2:])
        if hour > 23 or minute > 59:
            raise ValueError("HHMM 超出范围")
    except ValueError as exc:
        results.error("WEATHER_VALUE", path, f"第 {line_number} 行含无效数值/日期/时刻: {exc}")
        return None
    return values


def read_weather(
    path: Path, results: Results, strict_single_space: bool
) -> tuple[list[list[str] | None], dict[str, list[str]]]:
    if not path.is_file():
        results.error("MISSING_WEATHER", path, "缺少与事件同名的天气文件")
        return [], {}
    if not nonempty_file(path, results, "EMPTY_WEATHER"):
        return [], {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        results.error("WEATHER_UNREADABLE", path, f"UTF-8 读取失败: {exc}")
        return [], {}
    parsed: list[list[str] | None] = []
    by_date: dict[str, list[str]] = {}
    for number, line in enumerate(lines, 1):
        values = parse_weather_line(line, path, number, results, strict_single_space)
        parsed.append(values)
        if values is not None:
            token = f"{int(values[0]):04d}{int(values[1]):02d}{int(values[2]):02d}"
            if token in by_date:
                results.warning("WEATHER_DUPLICATE_DATE", path, f"日期 {token} 重复，代码会使用最后一行")
            by_date[token] = values
    results.stats["weather_files"] += 1
    results.stats["weather_rows"] += len(lines)
    return parsed, by_date


def simulation_can_form_sequence(frame_count: int, input_length: int) -> bool:
    required = input_length * 2
    if frame_count < required:
        return False
    max_interval = int(math.floor((frame_count - 1) * 0.167))
    return max_interval >= 1 and frame_count - 1 >= required - 1


def validate_sim_event(
    scene: Path,
    event: Path,
    rules: Rules,
    results: Results,
    level: str,
    decoders,
) -> bool:
    valid = True
    if not SCENE_RE.fullmatch(event.name):
        results.warning("NONCANONICAL_EVENT_NAME", event, "事件名通常应为 ####_#####")
    image_files = sorted(
        (path for path in event.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: path.name,
    )
    numbered: list[tuple[int, Path]] = []
    for path in image_files:
        if not nonempty_file(path, results, "EMPTY_SIM_FRAME"):
            valid = False
        match = SIM_FRAME_RE.fullmatch(path.name)
        if match is None:
            results.error("SIM_FRAME_NAME", path, "模拟帧必须严格命名为 out1.jpg、out2.jpg……")
            valid = False
        else:
            numbered.append((int(match.group(1)), path))
    numbered.sort(key=lambda item: item[0])
    indices = [item[0] for item in numbered]
    if not indices:
        results.error("NO_SIM_FRAMES", event, "没有可用的 outN.jpg 模拟火区帧")
        return False
    expected = list(range(1, indices[-1] + 1))
    if indices != expected:
        missing = sorted(set(expected) - set(indices))
        results.error("SIM_FRAME_GAP", event, f"outN.jpg 编号不连续，缺少前若干编号: {missing[:20]}")
        valid = False
    if not simulation_can_form_sequence(len(indices), rules.input_length):
        results.error(
            "SIM_SEQUENCE_TOO_SHORT",
            event,
            f"仅 {len(indices)} 帧，当前随机间隔算法无法稳定形成 {rules.input_length * 2} 帧序列",
        )
        valid = False
    if len(indices) < 80 or len(indices) > 120:
        results.warning(
            "SIM_PAPER_FRAME_COUNT",
            event,
            f"共有 {len(indices)} 帧；论文描述每个模拟序列约 100 帧",
        )

    weather_path = scene / rules.weather_dir / f"{event.name}.txt"
    weather_rows, _ = read_weather(weather_path, results, strict_single_space=True)
    if len(weather_rows) < indices[-1]:
        results.error(
            "SIM_WEATHER_TOO_SHORT",
            weather_path,
            f"天气仅 {len(weather_rows)} 行，但最大火区帧编号为 {indices[-1]}",
        )
        valid = False
    elif any(weather_rows[index - 1] is None for index in indices):
        results.error("SIM_WEATHER_INVALID_ROW", weather_path, "至少一个火区帧没有有效的对应天气行")
        valid = False

    paths = [path for _, path in numbered]
    for path in selected_for_decode(paths, level):
        if decode_image(path, results, decoders) is None:
            valid = False
    results.stats["sim_frames"] += len(paths)
    return valid


def validate_sim_scene(
    scene: Path, rules: Rules, results: Results, level: str, decoders
) -> None:
    valid = validate_static_maps(scene, rules, results, level, decoders)
    satellite = scene / rules.sim_satellite_file
    if not satellite.is_file():
        results.error("MISSING_SIM_SATELLITE", satellite, "缺少合成卫星底图，SimFireDataset 会跳过整个场景")
        valid = False
    elif not nonempty_file(satellite, results, "EMPTY_SIM_SATELLITE"):
        valid = False
    elif level != "structure" and decode_image(satellite, results, decoders) is None:
        valid = False

    mask_root = scene / rules.sim_mask_dir
    if not mask_root.is_dir():
        aliases = [
            name
            for name in ("Satellite_Image_Mask", "Satellite_lmages_Mask", "Satellite_Images_Mask")
            if (scene / name).is_dir()
        ]
        hint = f"；发现近似目录 {aliases}，但当前配置/代码不会读取" if aliases else ""
        results.error("MISSING_SIM_MASK_DIR", mask_root, "缺少配置指定的模拟火区目录" + hint)
        return
    events = sorted(path for path in mask_root.iterdir() if path.is_dir())
    if not events:
        results.error("NO_SIM_EVENTS", mask_root, "没有模拟事件子目录")
        return
    weather_stems = {
        path.stem
        for path in (scene / rules.weather_dir).glob("*.txt")
        if path.is_file()
    }
    event_names = {event.name for event in events}
    for orphan in sorted(weather_stems - event_names):
        results.warning(
            "ORPHAN_SIM_WEATHER",
            scene / rules.weather_dir / f"{orphan}.txt",
            "天气文件没有对应的模拟火区事件目录",
        )
    usable = 0
    for event in events:
        results.stats["sim_events"] += 1
        if validate_sim_event(scene, event, rules, results, level, decoders):
            usable += 1
    results.stats["usable_sim_events"] += usable
    if usable == 0:
        results.error("NO_USABLE_SIM_EVENTS", scene, "该场景没有可被当前代码使用的模拟事件")
    elif valid:
        results.stats["usable_sim_scenes"] += 1


def date_token(filename: str) -> str | None:
    stem = Path(filename).stem
    token = stem.rsplit("_", 1)[-1]
    try:
        datetime.strptime(token, "%Y%m%d")
    except ValueError:
        return None
    return token


def validate_real_event(
    scene: Path,
    mask_event: Path,
    rules: Rules,
    results: Results,
    level: str,
    decoders,
) -> bool:
    valid = True
    satellite_event = scene / rules.real_satellite_dir / mask_event.name
    weather_path = scene / rules.weather_dir / f"{mask_event.name}.txt"
    if not satellite_event.is_dir():
        results.error("MISSING_REAL_SATELLITE_EVENT", satellite_event, "缺少与掩膜事件同名的卫星图像目录")
        return False

    mask_files = sorted(
        path for path in mask_event.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not mask_files:
        results.error("NO_REAL_FRAMES", mask_event, "没有真实火区掩膜图像")
        return False
    token_to_mask: dict[str, Path] = {}
    for path in mask_files:
        if not nonempty_file(path, results, "EMPTY_REAL_MASK"):
            valid = False
        token = date_token(path.name)
        if token is None:
            results.error("REAL_FRAME_DATE", path, "文件名末尾必须含 YYYYMMDD 日期")
            valid = False
            continue
        if token in token_to_mask:
            results.error("REAL_DUPLICATE_DATE", path, f"事件中日期 {token} 重复")
            valid = False
        token_to_mask[token] = path

    frame_count = len(token_to_mask)
    if frame_count < rules.input_length * 2:
        results.error(
            "REAL_SEQUENCE_TOO_SHORT",
            mask_event,
            f"仅 {frame_count} 个有效日期；当前代码需要至少 {rules.input_length * 2} 帧",
        )
        valid = False
    if frame_count < 2 or frame_count > 10:
        results.warning("REAL_PAPER_FRAME_COUNT", mask_event, f"共有 {frame_count} 帧；论文描述真实序列为 2–10 帧")

    _, weather_by_date = read_weather(weather_path, results, strict_single_space=False)
    pairs: list[tuple[Path, Path]] = []
    for token, mask_path in sorted(token_to_mask.items()):
        satellite_path = satellite_event / mask_path.name
        if not satellite_path.is_file():
            results.error("MISSING_REAL_SATELLITE", satellite_path, "缺少与掩膜同名的真实卫星图像")
            valid = False
        else:
            if not nonempty_file(satellite_path, results, "EMPTY_REAL_SATELLITE"):
                valid = False
            pairs.append((mask_path, satellite_path))
        if token not in weather_by_date:
            results.error("MISSING_REAL_WEATHER_DATE", weather_path, f"缺少日期 {token} 的天气记录")
            valid = False

    chosen_masks = set(selected_for_decode([pair[0] for pair in pairs], level))
    for mask_path, satellite_path in pairs:
        if mask_path not in chosen_masks:
            continue
        mask_shape = decode_image(mask_path, results, decoders)
        satellite_shape = decode_image(satellite_path, results, decoders)
        if mask_shape is None or satellite_shape is None:
            valid = False
        elif mask_shape != satellite_shape:
            results.warning(
                "REAL_PAIR_SHAPE_MISMATCH",
                mask_path,
                f"掩膜尺寸 {mask_shape} 与卫星图像 {satellite_shape} 不同；代码会缩放，但论文要求空间对齐",
            )
    results.stats["real_frames"] += frame_count
    return valid


def validate_real_scene(
    scene: Path, rules: Rules, results: Results, level: str, decoders
) -> None:
    static_valid = validate_static_maps(scene, rules, results, level, decoders)
    required_dirs = (rules.real_mask_dir, rules.real_satellite_dir, rules.weather_dir)
    for directory in required_dirs:
        if not (scene / directory).is_dir():
            results.error("MISSING_DIRECTORY", scene / directory, "缺少真实数据必需目录")
            return
    mask_root = scene / rules.real_mask_dir
    events = sorted(path for path in mask_root.iterdir() if path.is_dir())
    if not events:
        results.error("NO_REAL_EVENTS", mask_root, "没有真实事件子目录")
        return
    event_names = {event.name for event in events}
    satellite_names = {
        path.name
        for path in (scene / rules.real_satellite_dir).iterdir()
        if path.is_dir()
    }
    weather_stems = {
        path.stem
        for path in (scene / rules.weather_dir).glob("*.txt")
        if path.is_file()
    }
    for orphan in sorted(satellite_names - event_names):
        results.warning(
            "ORPHAN_REAL_SATELLITE_EVENT",
            scene / rules.real_satellite_dir / orphan,
            "卫星图像事件目录没有对应的掩膜事件目录",
        )
    for orphan in sorted(weather_stems - event_names):
        results.warning(
            "ORPHAN_REAL_WEATHER",
            scene / rules.weather_dir / f"{orphan}.txt",
            "天气文件没有对应的真实掩膜事件目录",
        )
    usable = 0
    for event in events:
        results.stats["real_events"] += 1
        if validate_real_event(scene, event, rules, results, level, decoders):
            usable += 1
    results.stats["usable_real_events"] += usable
    if usable == 0:
        results.error("NO_USABLE_REAL_EVENTS", scene, "该场景没有可被当前代码使用的真实事件")
    elif static_valid:
        results.stats["usable_real_scenes"] += 1


def scene_directories(root: Path, kind: str, results: Results) -> list[Path]:
    if not root.is_dir():
        results.error("DATASET_ROOT_NOT_FOUND", root, f"{kind} 数据目录不存在")
        return []
    scenes = sorted(path for path in root.iterdir() if path.is_dir())
    if not scenes:
        results.error("NO_SCENES", root, f"{kind} 数据目录没有场景")
    for scene in scenes:
        if not SCENE_RE.fullmatch(scene.name):
            results.warning(
                "NONCANONICAL_SCENE_NAME",
                scene,
                "场景名不是 ####_#####；代码可能将副本/临时目录也当成数据",
            )
    return scenes


def validate_all(
    dataset_root: Path,
    rules: Rules,
    results: Results,
    level: str,
    progress_every: int,
) -> None:
    decoders = import_decoders() if level != "structure" else None
    sim_root = dataset_root / "sim_full"
    real_root = dataset_root / "real_full"
    sim_scenes = scene_directories(sim_root, "模拟", results)
    real_scenes = scene_directories(real_root, "真实", results)
    results.stats["sim_scenes"] = len(sim_scenes)
    results.stats["real_scenes"] = len(real_scenes)

    print(f"[INFO] 模拟场景: {len(sim_scenes)}，真实场景: {len(real_scenes)}，级别: {level}")
    for index, scene in enumerate(sim_scenes, 1):
        validate_sim_scene(scene, rules, results, level, decoders)
        if progress_every and (index % progress_every == 0 or index == len(sim_scenes)):
            print(f"[INFO] 模拟数据进度 {index}/{len(sim_scenes)}")
    for index, scene in enumerate(real_scenes, 1):
        validate_real_scene(scene, rules, results, level, decoders)
        if progress_every and (index % progress_every == 0 or index == len(real_scenes)):
            print(f"[INFO] 真实数据进度 {index}/{len(real_scenes)}")

    if results.stats["usable_sim_events"] == 0:
        results.error("NO_USABLE_SIM_DATASET", sim_root, "整个模拟数据集没有可用事件")
    if results.stats["usable_real_events"] == 0:
        results.error("NO_USABLE_REAL_DATASET", real_root, "整个真实数据集没有可用事件")


def write_report(
    report_path: Path,
    dataset_root: Path,
    config_path: Path | None,
    rules: Rules,
    level: str,
    results: Results,
) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "config": str(config_path) if config_path else None,
        "validation_level": level,
        "exit_status": "failed" if results.error_count else "passed_with_warnings" if results.warning_count else "passed",
        "requirements": {
            "paper": [
                "five modalities: topography, vegetation, fuel, weather, satellite/fire masks",
                "simulation sequences contain about 100 frames",
                "real-world sequences contain 2-10 frames",
                "S2R-FireTr uses six frames resized to 256x256",
            ],
            "repository": asdict(rules),
        },
        "summary": {
            "errors": results.error_count,
            "warnings": results.warning_count,
            "details_recorded": len(results.issues),
            "detail_limit": results.detail_limit,
        },
        "issue_counts": dict(sorted(results.issue_counts.items())),
        "stats": dict(sorted(results.stats.items())),
        "issues": [asdict(issue) for issue in results.issues],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def print_summary(results: Results, report_path: Path) -> None:
    print(f"[RESULT] errors={results.error_count}, warnings={results.warning_count}")
    if results.issue_counts:
        print("[RESULT] 问题类型统计:")
        for key, count in sorted(results.issue_counts.items(), key=lambda item: (-item[1], item[0])):
            print(f"  {count:8d}  {key}")
    if results.issues:
        print("[RESULT] 前 20 条问题明细:")
        for issue in results.issues[:20]:
            print(f"  [{issue.severity}] {issue.code}: {issue.path} - {issue.message}")
    print(f"[RESULT] JSON 报告: {report_path}")
    total_issues = results.error_count + results.warning_count
    if total_issues > len(results.issues):
        print(
            f"[WARNING] JSON 仅保存前 {len(results.issues)} 条明细；"
            "完整数量仍保存在 issue_counts 中"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="只读校验 Sim2Real-Fire 的目录、缺失文件、天气字段和可解码性。"
    )
    parser.add_argument("--dataset-root", required=True, help="同时包含 sim_full 和 real_full 的目录")
    parser.add_argument("--config", help="项目 YAML；用于按当前代码配置检查目录名与模型维度")
    parser.add_argument(
        "--level",
        choices=("structure", "sample", "full"),
        default="full",
        help="structure 只查结构；sample 抽样解码；full 解码全部文件（默认）",
    )
    parser.add_argument("--report", default="dataset_validation_report.json", help="JSON 报告路径")
    parser.add_argument("--detail-limit", type=int, default=5000, help="报告最多保存多少条问题明细")
    parser.add_argument("--progress-every", type=int, default=25, help="每多少个场景打印一次进度，0 表示关闭")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.detail_limit < 1 or args.progress_every < 0:
        print("[FATAL] --detail-limit 必须大于 0，--progress-every 不能小于 0", file=sys.stderr)
        return 2
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve() if args.config else None
    report_path = Path(args.report).expanduser().resolve()
    results = Results(args.detail_limit)
    try:
        rules = load_rules(config_path, results)
        validate_all(dataset_root, rules, results, args.level, args.progress_every)
        write_report(report_path, dataset_root, config_path, rules, args.level, results)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[FATAL] 校验器无法继续: {exc}", file=sys.stderr)
        return 2

    print_summary(results, report_path)
    return 1 if results.error_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
