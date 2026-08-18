#!/usr/bin/env python3
"""Attach per-process CUDA-memory sampling to already-running servers."""

from __future__ import annotations

import argparse
import os
import threading
import time
from pathlib import Path

import run_robocasa_atomic_matrix as matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--stop-pid", type=int, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument(
        "--server", action="append", required=True,
        help="config:gpu:pid:instance_role",
    )
    parser.add_argument("--interval", type=float, default=10.0)
    return parser.parse_args()


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def main() -> None:
    args = parse_args()
    placements = []
    for value in args.server:
        fields = value.split(":", 3)
        if len(fields) != 4:
            raise SystemExit(f"invalid --server {value!r}")
        config, gpu, pid, role = fields
        placements.append({
            "config": config,
            "gpu": int(gpu),
            "pid": int(pid),
            "instance_role": role,
        })
    stop = threading.Event()
    monitor = threading.Thread(
        target=matrix.monitor_server_processes,
        args=(stop, Path(args.out), placements, args.interval, args.source),
        daemon=True,
    )
    monitor.start()
    try:
        while pid_alive(args.stop_pid):
            time.sleep(min(5.0, max(1.0, args.interval)))
    finally:
        stop.set()
        monitor.join(timeout=max(5.0, args.interval + 2.0))


if __name__ == "__main__":
    main()
