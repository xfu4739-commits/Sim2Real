"""Copy remaining Sim2Real-Fire simulation archives to a separate local folder.

Use this when sim_full already has extracted scenes you are uploading elsewhere,
and you still need the missing .zip packages from Google Drive (G:) for later
upload/extract on a server.

Default is inventory only. Pass --execute to start copying.

Example:
  python tools/download_remaining_archives.py
  tools\\run_download_remaining.bat
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Local paths (no D: drive on this machine).
DATA_ROOT = Path(r"C:/Users/10115/Datasets/Sim2Real-Fire")
DEFAULT_SOURCE = Path(r"G:/我的云端硬盘")
DEFAULT_SIM_FULL = DATA_ROOT / "sim_full"
DEFAULT_DEST = DATA_ROOT / "sim_archives_remaining"

GIB = 1024**3
CHUNK_BYTES = 8 * 1024 * 1024
PATTERN = re.compile(r"\d{4}_\d{5}\.(zip|rar|7z)$", re.I)
PRINT_LOCK = threading.Lock()


def say(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def scene_names(sim_full: Path) -> set[str]:
    return {
        path.name
        for path in sim_full.iterdir()
        if path.is_dir() and re.fullmatch(r"\d{4}_\d{5}", path.name)
    }


def list_archives(source_root: Path) -> list[Path]:
    return sorted(
        path
        for path in source_root.iterdir()
        if path.is_file() and PATTERN.fullmatch(path.name)
    )


def remaining_archives(source_root: Path, sim_full: Path, dest_root: Path) -> list[Path]:
    done_scenes = scene_names(sim_full)
    pending = []
    for archive in list_archives(source_root):
        if archive.stem in done_scenes:
            continue
        dest = dest_root / archive.name
        if dest.is_file() and dest.stat().st_size == archive.stat().st_size:
            continue
        pending.append(archive)
    return pending


def bytes_needed(archives: list[Path], dest_root: Path) -> int:
    total = 0
    for archive in archives:
        dest = dest_root / archive.name
        if dest.is_file():
            remaining = archive.stat().st_size - dest.stat().st_size
            if remaining > 0:
                total += remaining
        else:
            total += archive.stat().st_size
    return total


def python_copy(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".copy-part")
    if part.exists():
        part.unlink()
    with source.open("rb") as src, part.open("wb") as out:
        shutil.copyfileobj(src, out, length=CHUNK_BYTES)
    os.replace(part, dest)


def robocopy_file(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "robocopy",
            str(source.parent),
            str(dest.parent),
            source.name,
            "/J",
            "/NFL",
            "/NDL",
            "/NJH",
            "/NJS",
            "/NC",
            "/NS",
            "/NP",
        ],
        capture_output=True,
        text=True,
        errors="replace",
    )
    if result.returncode >= 8:
        detail = (result.stdout or result.stderr or "").strip()[-1000:]
        raise RuntimeError(f"robocopy failed ({result.returncode}): {detail}")


def copy_one(source: Path, dest_root: Path, use_robocopy: bool) -> dict:
    dest = dest_root / source.name
    source_size = source.stat().st_size
    if dest.is_file() and dest.stat().st_size == source_size:
        return {
            "archive": source.name,
            "status": "skipped_complete",
            "bytes": source_size,
        }

    try:
        if use_robocopy and os.name == "nt":
            robocopy_file(source, dest)
        else:
            python_copy(source, dest)
        copied_size = dest.stat().st_size
        if copied_size != source_size:
            raise RuntimeError(f"size mismatch after copy: expected {source_size}, got {copied_size}")
        return {
            "archive": source.name,
            "status": "complete",
            "bytes": copied_size,
            "gib": round(copied_size / GIB, 2),
        }
    except Exception as error:
        return {
            "archive": source.name,
            "status": "failed",
            "error": str(error),
        }


def write_log(dest_root: Path, record: dict) -> None:
    log_path = dest_root / "download-log.jsonl"
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help="Folder for archives that are not yet represented in sim_full.",
    )
    parser.add_argument(
        "--sim-full",
        type=Path,
        default=DEFAULT_SIM_FULL,
        help="Existing extracted scenes; matching archive names are skipped.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel archive copies. 3-4 is usually fastest for Google Drive.",
    )
    parser.add_argument(
        "--reserve-gib",
        type=float,
        default=5.0,
        help="Keep at least this much free space on the destination drive.",
    )
    parser.add_argument(
        "--robocopy",
        action="store_true",
        help="Use Windows robocopy for each archive (often faster on NTFS).",
    )
    parser.add_argument(
        "--smallest-first",
        action="store_true",
        help="Copy smaller archives first so more packages finish earlier.",
    )
    parser.add_argument("--limit", type=int, help="Only copy the first N remaining archives.")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if not 1 <= args.workers <= 8:
        parser.error("--workers must be between 1 and 8")

    if not args.source.exists():
        parser.error(f"source not found: {args.source}")
    if not args.sim_full.exists():
        parser.error(f"sim_full not found: {args.sim_full}")

    done_scenes = scene_names(args.sim_full)
    all_archives = list_archives(args.source)
    pending = remaining_archives(args.source, args.sim_full, args.dest)
    if args.smallest_first:
        pending.sort(key=lambda path: path.stat().st_size)
    if args.limit:
        pending = pending[: args.limit]

    needed_bytes = bytes_needed(pending, args.dest)
    dest_usage = shutil.disk_usage(args.dest.parent if not args.dest.exists() else args.dest)
    free_gib = dest_usage.free / GIB
    needed_gib = needed_bytes / GIB

    say(f"source archives: {len(all_archives)}")
    say(f"scenes already in sim_full: {len(done_scenes)}")
    say(f"remaining archives to copy: {len(pending)}")
    say(f"remaining size: {needed_gib:.2f} GiB")
    say(f"destination: {args.dest}")
    say(f"destination drive free: {free_gib:.2f} GiB")
    say(f"workers: {args.workers}; robocopy={args.robocopy}")

    if pending and free_gib < needed_gib + args.reserve_gib:
        say(
            "WARNING: destination may not have enough free space. "
            f"Need about {needed_gib:.2f} GiB + reserve {args.reserve_gib:.2f} GiB."
        )

    if not pending:
        say("Nothing to do. Remaining archives are already present in --dest.")
        return

    if not args.execute:
        say("Inventory only. Add --execute to start copying.")
        say("First 10 pending archives:")
        for archive in pending[:10]:
            say(f"  {archive.name} ({archive.stat().st_size / GIB:.2f} GiB)")
        if len(pending) > 10:
            say(f"  ... and {len(pending) - 10} more")
        return

    args.dest.mkdir(parents=True, exist_ok=True)
    completed = 0
    failed = 0
    skipped = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(copy_one, archive, args.dest, args.robocopy): archive
            for archive in pending
        }
        for future in as_completed(futures):
            archive = futures[future]
            result = future.result()
            write_log(args.dest, result)
            status = result["status"]
            if status == "complete":
                completed += 1
                say(f"[{completed + failed + skipped}/{len(pending)}] copied {archive.name} ({result['gib']} GiB)")
            elif status == "skipped_complete":
                skipped += 1
                say(f"[{completed + failed + skipped}/{len(pending)}] skipped {archive.name}")
            else:
                failed += 1
                say(f"[{completed + failed + skipped}/{len(pending)}] FAILED {archive.name}: {result.get('error')}")

    say(
        f"Done. complete={completed}, skipped={skipped}, failed={failed}, "
        f"log={args.dest / 'download-log.jsonl'}"
    )


if __name__ == "__main__":
    main()
