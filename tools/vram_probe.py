"""VRAM baseline probe for monitoring GPU memory usage during workload execution."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import time
from typing import Callable, Optional


def query_gpu_memory() -> tuple[int, int, int]:
    """Query total GPU memory: used, free, total in MiB."""
    cmd = [
        "nvidia-smi",
        "--query-gpu=memory.used,memory.free,memory.total",
        "--format=csv,noheader,nounits",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0 or not res.stdout.strip():
        return 0, 0, 0
    parts = res.stdout.strip().splitlines()[0].split(",")
    return int(parts[0].strip()), int(parts[1].strip()), int(parts[2].strip())


def query_compute_apps(target_pid: Optional[int] = None) -> tuple[str, str, int]:
    """Query compute processes and return (pid, process_name, used_gpu_memory_mib)."""
    cmd = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0 or not res.stdout.strip():
        return "", "", 0

    apps: list[tuple[str, str, int]] = []
    for line in res.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if len(parts) >= 3 and parts[2].isdigit():
            apps.append((parts[0], parts[1], int(parts[2])))

    if not apps:
        return "", "", 0
    if target_pid is not None:
        matched = [a for a in apps if a[0] == str(target_pid)]
        return matched[0] if matched else ("", "", 0)
    py_apps = [a for a in apps if "python" in a[1].lower()]
    return max(py_apps, key=lambda a: a[2]) if py_apps else max(apps, key=lambda a: a[2])


def run_probe(
    output_path: Path,
    interval: float = 1.0,
    duration: float = 0.0,
    command: Optional[str] = None,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, int]:
    """Run VRAM monitoring loop, writing samples to CSV and returning summary."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(shlex.split(command)) if command else None
    samples, peak_used, peak_cv, min_free = 0, 0, 0, 10**9

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "ts",
                "total_used_mib",
                "total_free_mib",
                "total_mib",
                "cv_pid",
                "cv_proc",
                "cv_used_mib",
            ]
        )
        start = clock()
        try:
            while True:
                target_pid = proc.pid if proc else None
                used, free, total = query_gpu_memory()
                cv_pid, cv_proc, cv_used = query_compute_apps(target_pid)
                writer.writerow(
                    [
                        datetime.now(timezone.utc).isoformat(),
                        used,
                        free,
                        total,
                        cv_pid,
                        cv_proc,
                        cv_used,
                    ]
                )
                f.flush()
                samples += 1
                peak_used, peak_cv, min_free = (
                    max(peak_used, used),
                    max(peak_cv, cv_used),
                    min(min_free, free),
                )
                if (proc and proc.poll() is not None) or (duration > 0.0 and (clock() - start) >= duration):
                    break
                time.sleep(interval)
        except KeyboardInterrupt:
            pass
        finally:
            if proc and proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=5)

    summary = {
        "samples": samples,
        "peak_used_mib": peak_used,
        "peak_cv_mib": peak_cv,
        "min_free_mib": 0 if min_free == 10**9 else min_free,
    }
    print(json.dumps(summary))
    return summary


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description="VRAM baseline probe")
    p.add_argument(
        "--output",
        default="docs/acceptance/T0.5/vram_baseline.csv",
        help="Output CSV path",
    )
    p.add_argument("--interval", type=float, default=1.0, help="Polling interval in seconds")
    p.add_argument("--duration", type=float, default=0.0, help="Duration in seconds")
    p.add_argument("--cmd", default=None, help="Command to profile")
    args = p.parse_args()
    run_probe(Path(args.output), args.interval, args.duration, args.cmd)


if __name__ == "__main__":
    main()
