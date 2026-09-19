import json
import re
import shutil
from collections import Counter
from pathlib import Path

SIM_FULL = Path(r"C:\Users\10115\Datasets\Sim2Real-Fire\sim_full")
SOURCE = Path(r"G:/我的云端硬盘")
LOG = SIM_FULL / "extraction-log.jsonl"
PATTERN = re.compile(r"\d{4}_\d{5}\.(zip|rar|7z)$", re.I)


def latest_records(records):
    latest = {}
    for record in records:
        latest[record["archive"]] = record
    return latest


def main():
    usage = shutil.disk_usage("C:/")
    print(f"C: free={usage.free / (1024**3):.2f} GiB, total={usage.total / (1024**3):.2f} GiB")

    sim_usage = shutil.disk_usage(SIM_FULL)
    print(
        f"sim_full drive free={sim_usage.free / (1024**3):.2f} GiB, "
        f"used on partition={sim_usage.used / (1024**3):.2f} GiB"
    )

    scenes = sorted(
        p.name for p in SIM_FULL.iterdir() if p.is_dir() and re.fullmatch(r"\d{4}_\d{5}", p.name)
    )
    records = [json.loads(line) for line in LOG.read_text(encoding="utf-8").splitlines() if line.strip()]
    latest = latest_records(records)

    print(f"scene dirs={len(scenes)}")
    print(f"unique archives in log={len(latest)}")

    status_counts = Counter(record["status"] for record in latest.values())
    print("latest status counts:")
    for status, count in status_counts.most_common():
        print(f"  {status}: {count}")

    print("\ninsufficient_space:")
    for archive, record in sorted(latest.items()):
        if record["status"] == "insufficient_space":
            print(f"  {archive}: need ~{record.get('estimated_needed_gib', '?')} GiB")

    failed_errors = Counter(record.get("error", "<no error>") for record in latest.values() if record["status"] == "failed")
    print("\nfailed error breakdown:")
    for error, count in failed_errors.most_common(10):
        print(f"  [{count}] {error}")

    unmarked = []
    for archive, record in latest.items():
        scene = Path(archive).stem
        dest = SIM_FULL / scene
        marker = dest / ".extraction-complete.json"
        if record["status"] == "failed" and dest.exists() and not marker.is_file():
            unmarked.append(scene)
    print(f"\nfailed + dir exists but no completion marker: {len(unmarked)}")
    if unmarked[:10]:
        print("  examples:", ", ".join(unmarked[:10]))

    if SOURCE.exists():
        source_archives = sorted(p.name for p in SOURCE.iterdir() if p.is_file() and PATTERN.fullmatch(p.name))
        not_started = [name for name in source_archives if name not in latest]
        print(f"\nsource archives={len(source_archives)}")
        print(f"not started yet={len(not_started)}")
        if not_started[:5]:
            print("  first not started:", ", ".join(not_started[:5]))
    else:
        print(f"\nsource missing: {SOURCE}")


if __name__ == "__main__":
    main()
