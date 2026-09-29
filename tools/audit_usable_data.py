#!/usr/bin/env python3
"""Build an exact usable-data inventory for one sim scene and all real scenes.

This is a read-only companion to ``validate_dataset.py``.  It always performs
full decoding and writes both a machine-readable JSON report and CSV lists.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import warnings
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from validate_dataset import (
    IMAGE_SUFFIXES,
    Results,
    import_decoders,
    load_rules,
    nonempty_file,
    scene_directories,
    validate_real_event,
    validate_sim_event,
    validate_static_maps,
    decode_image,
)


if os.name == "nt":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def issue_codes(results: Results, severity: str) -> list[str]:
    prefix = f"{severity}:"
    return sorted(key[len(prefix):] for key in results.issue_counts if key.startswith(prefix))


def event_record(event: Path, results: Results, valid: bool) -> dict:
    return {
        "event": event.name,
        "usable": bool(valid and results.error_count == 0),
        "errors": results.error_count,
        "warnings": results.warning_count,
        "error_codes": issue_codes(results, "ERROR"),
        "warning_codes": issue_codes(results, "WARNING"),
        "issue_counts": dict(sorted(results.issue_counts.items())),
        "issues": [asdict(issue) for issue in results.issues],
    }


def audit_sim_scene(scene: Path, rules, decoders) -> dict:
    base = Results(10000)
    base_valid = validate_static_maps(scene, rules, base, "full", decoders)
    satellite = scene / rules.sim_satellite_file
    if not satellite.is_file():
        base.error("MISSING_SIM_SATELLITE", satellite, "缺少合成卫星底图")
        base_valid = False
    elif not nonempty_file(satellite, base, "EMPTY_SIM_SATELLITE"):
        base_valid = False
    elif decode_image(satellite, base, decoders) is None:
        base_valid = False

    mask_root = scene / rules.sim_mask_dir
    events: list[dict] = []
    if not mask_root.is_dir():
        base.error("MISSING_SIM_MASK_DIR", mask_root, "缺少模拟火区目录")
    else:
        event_dirs = sorted(path for path in mask_root.iterdir() if path.is_dir())
        if not event_dirs:
            base.error("NO_SIM_EVENTS", mask_root, "没有模拟事件子目录")
        for index, event in enumerate(event_dirs, 1):
            event_results = Results(1000)
            valid = validate_sim_event(scene, event, rules, event_results, "full", decoders)
            events.append(event_record(event, event_results, valid))
            if index % 25 == 0 or index == len(event_dirs):
                print(f"[INFO] 模拟事件进度 {index}/{len(event_dirs)}", flush=True)

    usable_count = sum(item["usable"] for item in events)
    complete = (
        base_valid
        and base.error_count == 0
        and bool(events)
        and usable_count == len(events)
    )
    return {
        "scene": scene.name,
        "path": str(scene),
        "complete_for_training": complete,
        "base_errors": base.error_count,
        "base_warnings": base.warning_count,
        "base_error_codes": issue_codes(base, "ERROR"),
        "base_warning_codes": issue_codes(base, "WARNING"),
        "base_issue_counts": dict(sorted(base.issue_counts.items())),
        "base_issues": [asdict(issue) for issue in base.issues],
        "events_total": len(events),
        "events_usable": usable_count,
        "events_unusable": len(events) - usable_count,
        "usable_events": [item["event"] for item in events if item["usable"]],
        "unusable_events": [item for item in events if not item["usable"]],
    }


def audit_real_scene(scene: Path, rules, decoders) -> dict:
    base = Results(1000)
    base_valid = validate_static_maps(scene, rules, base, "full", decoders)
    required_dirs = (rules.real_mask_dir, rules.real_satellite_dir, rules.weather_dir)
    for directory in required_dirs:
        path = scene / directory
        if not path.is_dir():
            base.error("MISSING_DIRECTORY", path, "缺少真实数据必需目录")
            base_valid = False

    events: list[dict] = []
    mask_root = scene / rules.real_mask_dir
    if mask_root.is_dir():
        event_dirs = sorted(path for path in mask_root.iterdir() if path.is_dir())
        if not event_dirs:
            base.error("NO_REAL_EVENTS", mask_root, "没有真实事件子目录")
        for event in event_dirs:
            event_results = Results(1000)
            valid = validate_real_event(scene, event, rules, event_results, "full", decoders)
            events.append(event_record(event, event_results, valid))

    environment_complete = base_valid and base.error_count == 0
    usable_events = [item for item in events if item["usable"] and environment_complete]
    if not environment_complete or not usable_events:
        status = "unusable"
    elif len(usable_events) == len(events):
        status = "complete"
    else:
        status = "partial"
    return {
        "scene": scene.name,
        "path": str(scene),
        "status": status,
        "environment_complete": environment_complete,
        "base_errors": base.error_count,
        "base_warnings": base.warning_count,
        "base_error_codes": issue_codes(base, "ERROR"),
        "base_warning_codes": issue_codes(base, "WARNING"),
        "base_issue_counts": dict(sorted(base.issue_counts.items())),
        "events_total": len(events),
        "events_usable": len(usable_events),
        "events_unusable": len(events) - len(usable_events),
        "usable_events": [item["event"] for item in usable_events],
        "unusable_events": [item for item in events if item not in usable_events],
    }


def accumulate_issue_counts(sim: dict, real: list[dict]) -> dict[str, int]:
    counts: Counter[str] = Counter(sim["base_issue_counts"])
    for event in sim["unusable_events"]:
        counts.update(event["issue_counts"])
    for scene in real:
        counts.update(scene["base_issue_counts"])
        for event in scene["unusable_events"]:
            counts.update(event["issue_counts"])
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def verify_real_loader(real: list[dict], config_path: Path) -> dict:
    """Materialise every file-complete sample through the current Dataset."""
    import torch
    import yaml

    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from dataset.real_dataset import RealFireDataset

    with config_path.open("r", encoding="utf-8") as stream:
        cfg = (yaml.safe_load(stream) or {})["dataset"]
    dataset = RealFireDataset(cfg)
    file_complete = {item["scene"] for item in real if item["status"] == "complete"}
    selected = [
        index for index, scene_path in enumerate(dataset.fire_name)
        if Path(scene_path).name in file_complete
    ]
    expected_shapes = [
        (3, 256, 256),
        (3, 256, 256),
        (10, 256, 256),
        (6, 256, 256),
        (4, 256, 256),
        (3, 3, 256, 256),
        (3, 10),
        (3,),
    ]
    failures: dict[str, list[dict]] = {}
    passed = 0
    for position, index in enumerate(selected, 1):
        scene_name = Path(dataset.fire_name[index]).name
        try:
            item = dataset[index]
            shapes = [tuple(value.shape) for value in item[1:]]
            finite = all(torch.isfinite(value).all().item() for value in item[1:])
            if shapes != expected_shapes or not finite:
                raise ValueError(f"unexpected shapes={shapes} or finite={finite}")
            passed += 1
        except Exception as exc:
            failures.setdefault(scene_name, []).append({
                "sample_index": index,
                "error_type": type(exc).__name__,
                "message": str(exc),
            })
        if position % 50 == 0 or position == len(selected):
            print(f"[INFO] 真实样本实际加载进度 {position}/{len(selected)}", flush=True)

    for scene in real:
        scene["file_status"] = scene["status"]
        if scene["scene"] in failures:
            scene["status"] = "unusable"
            scene["runtime_status"] = "failed"
            scene["runtime_failures"] = failures[scene["scene"]]
            formerly_usable = list(scene["usable_events"])
            scene["usable_events"] = []
            scene["events_usable"] = 0
            scene["events_unusable"] = scene["events_total"]
            for event_name in formerly_usable:
                scene["unusable_events"].append({
                    "event": event_name,
                    "usable": False,
                    "errors": len(failures[scene["scene"]]),
                    "warnings": 0,
                    "error_codes": ["RUNTIME_SAMPLE_FAILURE"],
                    "warning_codes": [],
                    "issue_counts": {
                        "ERROR:RUNTIME_SAMPLE_FAILURE": len(failures[scene["scene"]])
                    },
                    "issues": [],
                })
        elif scene["status"] == "complete":
            scene["runtime_status"] = "passed"
            scene["runtime_failures"] = []
        else:
            scene["runtime_status"] = "not_selected"
            scene["runtime_failures"] = []

    return {
        "dataset_materialized_samples": len(dataset),
        "file_complete_samples_selected": len(selected),
        "samples_passed": passed,
        "samples_failed": sum(len(items) for items in failures.values()),
        "scenes_failed": len(failures),
        "failed_scene_names": sorted(failures),
        "failures": failures,
    }


def write_csvs(output_dir: Path, sim: dict, real: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "real_scene_inventory.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "scene", "status", "environment_complete", "events_total",
            "events_usable", "events_unusable", "base_error_codes", "base_warning_codes",
        ])
        for scene in real:
            writer.writerow([
                scene["scene"], scene["status"], scene["environment_complete"],
                scene["events_total"], scene["events_usable"], scene["events_unusable"],
                ";".join(scene["base_error_codes"]), ";".join(scene["base_warning_codes"]),
            ])

    with (output_dir / "usable_real_events.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["scene", "event"])
        for scene in real:
            for event in scene["usable_events"]:
                writer.writerow([scene["scene"], event])

    with (output_dir / "unusable_real_events.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["scene", "event", "error_codes", "warning_codes"])
        for scene in real:
            for event in scene["unusable_events"]:
                writer.writerow([
                    scene["scene"], event["event"], ";".join(event["error_codes"]),
                    ";".join(event["warning_codes"]),
                ])

    with (output_dir / "sim_event_inventory.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["scene", "event", "usable", "error_codes", "warning_codes"])
        unusable = {item["event"]: item for item in sim["unusable_events"]}
        for event in sim["usable_events"]:
            writer.writerow([sim["scene"], event, True, "", ""])
        for name, item in unusable.items():
            writer.writerow([
                sim["scene"], name, False, ";".join(item["error_codes"]),
                ";".join(item["warning_codes"]),
            ])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="完整解码并列出当前代码可使用的数据。")
    parser.add_argument("--dataset-root", required=True, help="包含 sim_full 和 real_full 的目录")
    parser.add_argument("--sim-scene", required=True, help="唯一训练模拟场景名，例如 0156_00112")
    parser.add_argument("--config", required=True, help="当前项目 YAML 配置")
    parser.add_argument("--output-dir", default="output/data_audit", help="JSON/CSV 输出目录")
    parser.add_argument("--progress-every", type=int, default=25, help="真实场景进度间隔")
    parser.add_argument(
        "--reuse-full-report",
        action="store_true",
        help="复用 output-dir 中已有的全文件报告，仅重跑当前 Dataset 的逐样本验证",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    report_path = output_dir / "usable_data_report.json"
    setup = Results(100)
    rules = load_rules(config_path, setup)
    if setup.error_count:
        print("[FATAL] 配置校验失败", file=sys.stderr)
        return 2
    if args.reuse_full_report:
        if not report_path.is_file():
            print(f"[FATAL] 没有可复用的报告: {report_path}", file=sys.stderr)
            return 2
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        sim = previous["simulation"]
        real = previous["real_scenes"]
        discovery_issues = previous.get("discovery_issues", [])
        for scene in real:
            scene["status"] = scene.get("file_status", scene["status"])
            scene.pop("file_status", None)
            scene.pop("runtime_status", None)
            scene.pop("runtime_failures", None)
    else:
        decoders = import_decoders()
        cv2, _, _ = decoders
        # GeoTIFF georeferencing and out-of-range NoData tags are irrelevant to
        # pixel decoding but both backends otherwise print one message per file.
        # The validator still records actual decode failures in the JSON report.
        logging.getLogger("tifffile").setLevel(logging.CRITICAL)
        warnings.filterwarnings("ignore", module="tifffile")
        if hasattr(cv2, "setLogLevel"):
            cv2.setLogLevel(0)

        sim_scene = dataset_root / "sim_full" / args.sim_scene
        if not sim_scene.is_dir():
            print(f"[FATAL] 模拟场景不存在: {sim_scene}", file=sys.stderr)
            return 2
        real_root = dataset_root / "real_full"
        discovery = Results(1000)
        real_paths = scene_directories(real_root, "真实", discovery)
        discovery_issues = [asdict(issue) for issue in discovery.issues]

        print(f"[INFO] 完整校验模拟场景 {sim_scene.name}", flush=True)
        sim = audit_sim_scene(sim_scene, rules, decoders)
        print(f"[INFO] 完整校验全部真实场景，共 {len(real_paths)} 个", flush=True)
        real = []
        for index, scene in enumerate(real_paths, 1):
            try:
                real.append(audit_real_scene(scene, rules, decoders))
            except Exception as exc:
                real.append({
                    "scene": scene.name,
                    "path": str(scene),
                    "status": "unusable",
                    "environment_complete": False,
                    "base_errors": 1,
                    "base_warnings": 0,
                    "base_error_codes": ["VALIDATOR_EXCEPTION"],
                    "base_warning_codes": [],
                    "base_issue_counts": {"ERROR:VALIDATOR_EXCEPTION": 1},
                    "events_total": 0,
                    "events_usable": 0,
                    "events_unusable": 0,
                    "usable_events": [],
                    "unusable_events": [],
                    "exception": repr(exc),
                })
            if args.progress_every and (index % args.progress_every == 0 or index == len(real_paths)):
                print(f"[INFO] 真实场景进度 {index}/{len(real_paths)}", flush=True)

    runtime = verify_real_loader(real, config_path)

    summary = {
        "sim_scene_complete_for_training": sim["complete_for_training"],
        "sim_events_total": sim["events_total"],
        "sim_events_usable": sim["events_usable"],
        "sim_events_unusable": sim["events_unusable"],
        "real_scenes_total": len(real),
        "real_scenes_complete": sum(item["status"] == "complete" for item in real),
        "real_scenes_partial": sum(item["status"] == "partial" for item in real),
        "real_scenes_unusable": sum(item["status"] == "unusable" for item in real),
        "real_events_total": sum(item["events_total"] for item in real),
        "real_events_usable": sum(item["events_usable"] for item in real),
        "real_events_unusable": sum(item["events_unusable"] for item in real),
        "real_samples_materialized": runtime["dataset_materialized_samples"],
        "real_samples_selected": runtime["file_complete_samples_selected"],
        "real_samples_usable": runtime["samples_passed"],
        "real_samples_failed": runtime["samples_failed"],
    }
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "config": str(config_path),
        "validation_level": "full",
        "definition": {
            "sim_complete": "环境栅格、卫星底图及该场景全部事件均无硬错误且所有文件可解码",
            "real_complete": "环境栅格完整且场景中全部事件满足当前 6 帧代码要求、天气配对及图像解码",
            "real_partial": "环境栅格完整，但只有部分事件通过",
            "real_unusable": "环境栅格不完整或没有通过的事件",
        },
        "summary": summary,
        "issue_counts": accumulate_issue_counts(sim, real),
        "simulation": sim,
        "runtime_validation": runtime,
        "real_scenes": real,
        "discovery_issues": discovery_issues,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csvs(output_dir, sim, real)
    print(f"[RESULT] {json.dumps(summary, ensure_ascii=False)}", flush=True)
    print(f"[RESULT] 报告目录: {output_dir}", flush=True)
    return 0 if sim["complete_for_training"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
