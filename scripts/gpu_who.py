#!/usr/bin/env python3
"""Print GPU utilization and, for each GPU process, owner and command (via ps)."""

from __future__ import annotations

import csv
import io
import subprocess
import sys


def _nvidia_smi_csv(args: list[str]) -> str:
    r = subprocess.run(
        ["nvidia-smi", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode != 0:
        print(r.stderr or r.stdout, file=sys.stderr)
        sys.exit(r.returncode)
    return r.stdout


def _parse_csv_rows(text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in csv.reader(io.StringIO(text.strip())):
        if not row:
            continue
        rows.append([c.strip() for c in row])
    return rows


def _ps_line(pid: str) -> str | None:
    try:
        r = subprocess.run(
            ["ps", "-o", "user=,pid=,etime=,cmd=", "-p", pid],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if r.returncode != 0:
        return None
    line = (r.stdout or "").strip()
    return line or None


def main() -> None:
    gpu_text = _nvidia_smi_csv(
        [
            "--query-gpu=index,gpu_bus_id,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader",
        ]
    )
    gpu_rows = _parse_csv_rows(gpu_text)
    bus_to_idx: dict[str, str] = {}
    print("GPUs")
    print("-" * 72)
    for cols in gpu_rows:
        if len(cols) < 6:
            continue
        idx, bus, name, mused, mtot, util = cols[:6]
        bus_to_idx[bus] = idx
        print(f"  [{idx}] {name}  mem {mused} / {mtot}  util {util}  bus {bus}")

    apps_text = _nvidia_smi_csv(
        [
            "--query-compute-apps=gpu_bus_id,pid,process_name,used_gpu_memory",
            "--format=csv,noheader",
        ]
    )
    app_rows = _parse_csv_rows(apps_text)

    print()
    print("GPU compute processes (nvidia-smi + ps)")
    print("-" * 72)
    if not app_rows:
        print("  (none)")
        return

    for cols in app_rows:
        if len(cols) < 4:
            continue
        bus, pid, pname, gmem = cols[:4]
        gidx = bus_to_idx.get(bus, "?")
        ps_info = _ps_line(pid)
        ps_part = ps_info if ps_info else "(ps: not found or ended)"
        print(f"  GPU {gidx}  pid {pid}  {gmem}  smi_name={pname}")
        print(f"         {ps_part}")


if __name__ == "__main__":
    main()
