#!/usr/bin/env python3
"""Inspect every simulation folder and list candidates for one-scene training."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from validate_dataset import (
    Results, decode_image, import_decoders, load_rules, nonempty_file,
    scene_directories, validate_sim_event, validate_static_maps,
)

if os.name == "nt":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="检查 sim_full 下全部模拟场景文件夹。")
    parser.add_argument("--dataset-root", required=True, help="包含 sim_full 的数据集目录")
    parser.add_argument("--config", required=True, help="当前项目 YAML 配置")
    parser.add_argument("--output-dir", default="output/data_audit", help="报告输出目录")
    return parser


def sample_three(paths: list[Path]) -> list[Path]:
    if len(paths) <= 3:
        return paths
    return [paths[0], paths[len(paths) // 2], paths[-1]]


def audit_folder(scene: Path, rules, decoders) -> dict:
    results = Results(1000)
    environment_files_ok = validate_static_maps(scene, rules, results, "full", decoders)
    satellite = scene / rules.sim_satellite_file
    satellite_ok = False
    if not satellite.is_file():
        results.error("MISSING_SIM_SATELLITE", satellite, "缺少合成卫星底图")
    elif nonempty_file(satellite, results, "EMPTY_SIM_SATELLITE"):
        satellite_ok = decode_image(satellite, results, decoders) is not None

    runtime_environment_ok = False
    runtime_environment_error = None
    if environment_files_ok:
        project_root = str(Path(__file__).resolve().parents[1])
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from dataset.common import load_environment

        try:
            load_environment(
                scene, rules.topography_dir, rules.vegetation_dir,
                rules.fuel_dir, rules.image_size,
            )
            runtime_environment_ok = True
        except Exception as exc:
            runtime_environment_error = f"{type(exc).__name__}: {exc}"
            results.error("RUNTIME_ENVIRONMENT_FAILURE", scene, runtime_environment_error)

    mask_root = scene / rules.sim_mask_dir
    weather_root = scene / rules.weather_dir
    if not mask_root.is_dir():
        results.error("MISSING_SIM_MASK_DIR", mask_root, "缺少模拟事件目录")
        event_dirs = []
    else:
        event_dirs = sorted(path for path in mask_root.iterdir() if path.is_dir())
        if not event_dirs:
            results.error("NO_SIM_EVENTS", mask_root, "没有模拟事件")
    if not weather_root.is_dir():
        results.error("MISSING_DIRECTORY", weather_root, "缺少模拟天气目录")
        weather_stems: set[str] = set()
    else:
        weather_stems = {
            path.stem for path in weather_root.iterdir()
            if path.is_file() and path.suffix.lower() == ".txt"
        }
    event_names = {path.name for path in event_dirs}
    missing_weather = sorted(event_names - weather_stems)
    orphan_weather = sorted(weather_stems - event_names)
    if missing_weather:
        results.error(
            "MISSING_WEATHER_EVENTS", weather_root,
            f"{len(missing_weather)} 个事件缺同名天气文件；示例 {missing_weather[:5]}",
        )
    if orphan_weather:
        results.warning(
            "ORPHAN_SIM_WEATHER", weather_root,
            f"{len(orphan_weather)} 个天气文件没有同名事件；示例 {orphan_weather[:5]}",
        )

    sampled = []
    for event in sample_three(event_dirs):
        event_results = Results(1000)
        valid = validate_sim_event(scene, event, rules, event_results, "sample", decoders)
        sampled.append({
            "event": event.name,
            "usable": valid and event_results.error_count == 0,
            "errors": event_results.error_count,
            "warnings": event_results.warning_count,
            "issue_counts": dict(sorted(event_results.issue_counts.items())),
        })

    folder_candidate = (
        results.error_count == 0 and environment_files_ok and runtime_environment_ok
        and satellite_ok and bool(event_dirs) and not missing_weather
        and all(item["usable"] for item in sampled)
    )
    return {
        "scene": scene.name, "path": str(scene), "folder_candidate": folder_candidate,
        "environment_files_ok": environment_files_ok,
        "runtime_environment_ok": runtime_environment_ok,
        "runtime_environment_error": runtime_environment_error,
        "satellite_ok": satellite_ok, "events_total": len(event_dirs),
        "weather_files_total": len(weather_stems),
        "event_weather_names_match": not missing_weather,
        "missing_weather_count": len(missing_weather),
        "orphan_weather_count": len(orphan_weather), "sampled_events": sampled,
        "errors": results.error_count + sum(item["errors"] for item in sampled),
        "warnings": results.warning_count + sum(item["warnings"] for item in sampled),
        "issue_counts": dict(sorted(results.issue_counts.items())),
    }


def main() -> int:
    args = build_parser().parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    setup = Results(100)
    rules = load_rules(config_path, setup)
    if setup.error_count:
        print("[FATAL] 配置校验失败", file=sys.stderr)
        return 2
    decoders = import_decoders()
    cv2, _, _ = decoders
    logging.getLogger("tifffile").setLevel(logging.CRITICAL)
    warnings.filterwarnings("ignore", module="tifffile")
    if hasattr(cv2, "setLogLevel"):
        cv2.setLogLevel(0)

    discovery = Results(1000)
    scenes = scene_directories(dataset_root / "sim_full", "模拟", discovery)
    print(f"[INFO] 检查全部 {len(scenes)} 个模拟场景文件夹", flush=True)
    records = []
    for index, scene in enumerate(scenes, 1):
        try:
            record = audit_folder(scene, rules, decoders)
        except Exception as exc:
            record = {
                "scene": scene.name, "path": str(scene), "folder_candidate": False,
                "environment_files_ok": False, "runtime_environment_ok": False,
                "runtime_environment_error": f"{type(exc).__name__}: {exc}",
                "satellite_ok": False, "events_total": 0, "weather_files_total": 0,
                "event_weather_names_match": False, "missing_weather_count": 0,
                "orphan_weather_count": 0, "sampled_events": [], "errors": 1,
                "warnings": 0, "issue_counts": {"ERROR:VALIDATOR_EXCEPTION": 1},
            }
        records.append(record)
        print(
            f"[INFO] {index}/{len(scenes)} {scene.name}: "
            f"{'候选可用' if record['folder_candidate'] else '不可直接选'}，"
            f"事件={record['events_total']}", flush=True,
        )

    candidates = [item for item in records if item["folder_candidate"]]
    issue_counts: Counter[str] = Counter()
    for item in records:
        issue_counts.update(item["issue_counts"])
        for event in item["sampled_events"]:
            issue_counts.update(event["issue_counts"])
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root), "config": str(config_path),
        "validation_level": "all-folders",
        "definition": {
            "folder_candidate": (
                "19 个环境栅格与卫星底图完整可解码，环境栅格能通过当前预处理，"
                "事件目录与天气文件名一一对应，且首/中/末事件通过帧连续性、"
                "逐行天气和首尾帧解码检查"
            ),
            "selected_scene_full_validation": (
                "最终用于训练的一个场景还需解码全部事件的全部帧；0156_00112 已完成"
            ),
        },
        "summary": {
            "scenes_total": len(records), "folder_candidates": len(candidates),
            "folders_rejected": len(records) - len(candidates),
            "events_in_candidates": sum(item["events_total"] for item in candidates),
        },
        "candidate_scene_names": [item["scene"] for item in candidates],
        "issue_counts": dict(sorted(issue_counts.items(), key=lambda item: (-item[1], item[0]))),
        "scenes": records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "sim_folder_report.json"
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "sim_folder_inventory.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "scene", "folder_candidate", "events_total", "weather_files_total",
            "environment_files_ok", "runtime_environment_ok", "satellite_ok",
            "event_weather_names_match", "errors", "runtime_environment_error",
        ])
        for item in records:
            writer.writerow([
                item["scene"], item["folder_candidate"], item["events_total"],
                item["weather_files_total"], item["environment_files_ok"],
                item["runtime_environment_ok"], item["satellite_ok"],
                item["event_weather_names_match"], item["errors"],
                item["runtime_environment_error"] or "",
            ])
    with (output_dir / "usable_sim_folders.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["scene", "events", "full_validation"])
        for item in candidates:
            writer.writerow([
                item["scene"], item["events_total"],
                "passed" if item["scene"] == "0156_00112" else "required_after_selection",
            ])
    print(f"[RESULT] {json.dumps(payload['summary'], ensure_ascii=False)}", flush=True)
    print(f"[RESULT] 候选文件夹: {', '.join(payload['candidate_scene_names'])}", flush=True)
    print(f"[RESULT] 报告: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
