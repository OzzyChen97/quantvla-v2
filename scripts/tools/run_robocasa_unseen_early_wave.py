#!/usr/bin/env python3
"""Overlap composite_unseen quantized shards with the active seen matrix.

The full immutable unseen manifest is created up front.  Only the secondary
model replicas on GPUs 2/6/7 and shard indices 1/3/5/7 are launched here.
Rows are written directly to their manifest-declared result files.  The normal
full runner later skips completed shards and rebalances only the remainder over
primary and secondary instances.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

import run_robocasa_atomic_matrix as matrix


REPO = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = REPO / "runs/robocasa365_official_full_composite_unseen_spec.json"
DEFAULT_RUN_DIR = REPO / "runs/robocasa365_official_full_composite_unseen_paired50"
DEFAULT_CHECKPOINT = REPO / (
    "checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/"
    "target_posttraining/composite_unseen/checkpoint-60000"
)
CONFIG_IDS = {"w4a8_atmohb", "cscka_final", "cscka_final_atmohb"}
SHARD_INDICES = {1, 3, 5, 7}
EGL_POOL = [1, 2, 3, 4, 5, 6, 7]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default=str(DEFAULT_SPEC))
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--server-timeout", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build(args: argparse.Namespace) -> tuple[dict, str, Path]:
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = matrix.build_manifest(
        Path(args.spec).resolve(), run_dir, "formal", list(range(50)),
        "OpenCabinet", Path(args.checkpoint), "composite_unseen", None, None,
        8, 3600, 10, "paired", 10.0, EGL_POOL,
    )
    manifest, manifest_sha = matrix.write_or_validate_manifest(
        run_dir / "manifest.json", manifest
    )
    return manifest, manifest_sha, run_dir


def selected_instances(manifest: dict) -> list[tuple[dict, dict]]:
    selected = []
    for config in manifest["configs"]:
        if config["id"] not in CONFIG_IDS:
            continue
        replicas = matrix.server_instances(config)[1:]
        if len(replicas) != 1:
            raise RuntimeError(f"{config['id']}: expected exactly one secondary replica")
        selected.append((config, replicas[0]))
    if {config["id"] for config, _ in selected} != CONFIG_IDS:
        raise RuntimeError("unseen manifest does not contain all early-wave configs")
    return selected


def memory_requirements(manifest: dict, selected: list[tuple[dict, dict]]) -> dict[int, float]:
    client_counts = {gpu: 0 for gpu in EGL_POOL}
    for config, _ in selected:
        devices = config["shard_egl_devices"]
        for shard_index in SHARD_INDICES:
            client_counts[int(devices[shard_index])] += 1
    required = {
        gpu: 2048.0 + count * 2300.0 for gpu, count in client_counts.items()
    }
    for config, instance in selected:
        server_budget = 16384.0 if config["expected_wrapped"] else 12288.0
        gpu = int(instance["gpu"])
        required[gpu] = required.get(gpu, 2048.0) + server_budget
    return required


def run(args: argparse.Namespace) -> None:
    manifest, manifest_sha, run_dir = build(args)
    selected = selected_instances(manifest)
    required = memory_requirements(manifest, selected)
    free = matrix.gpu_free_memory_mib()
    insufficient = {
        gpu: {"free_mib": free.get(gpu), "required_mib": need}
        for gpu, need in required.items() if free.get(gpu, 0.0) < need
    }
    preflight = {
        "sampled_at": datetime.now(timezone.utc).isoformat(),
        "selected_configs": sorted(CONFIG_IDS),
        "selected_shards": sorted(SHARD_INDICES),
        "free_memory_mib": free,
        "required_free_memory_mib": required,
        "insufficient": insufficient,
    }
    (run_dir / "early_wave_preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")
    print(json.dumps(preflight, indent=2), flush=True)
    if insufficient:
        raise SystemExit(f"early-wave memory preflight failed: {insufficient}")
    if args.dry_run:
        print("[early-wave] dry-run PASS", flush=True)
        return

    servers = []
    clients = []
    monitor_stop = threading.Event()
    monitor_configs = [
        {"id": config["id"], "gpu": instance["gpu"]}
        for config, instance in selected
    ]
    monitor = threading.Thread(
        target=matrix.monitor_gpus,
        args=(monitor_stop, run_dir / "gpu_efficiency.jsonl", monitor_configs, 10.0),
        daemon=True,
    )
    monitor.start()
    server_monitor_stop = threading.Event()
    server_monitor = None
    try:
        runtime = {}
        for config, instance in selected:
            proc, handle = matrix.start_server(config, instance, manifest, run_dir)
            servers.append((proc, handle, config, instance))
        server_monitor = threading.Thread(
            target=matrix.monitor_server_processes,
            args=(
                server_monitor_stop,
                run_dir / "gpu_server_efficiency.jsonl",
                [
                    {
                        "config": config["id"], "gpu": instance["gpu"],
                        "pid": proc.pid, "instance_role": f"replica_{instance['replica']}",
                    }
                    for proc, _, config, instance in servers
                ],
                10.0,
                "early_wave",
            ),
            daemon=True,
        )
        server_monitor.start()
        for proc, _, config, instance in servers:
            key = f"{config['id']}/r{instance['replica']}"
            runtime[key] = matrix.wait_and_verify(
                config, instance, manifest, proc, args.server_timeout
            )
            print(f"[early-wave] verified {key}", flush=True)
        (run_dir / "runtime_info_early_wave.json").write_text(
            json.dumps(runtime, indent=2) + "\n"
        )
        clients = matrix.start_clients(
            manifest, manifest_sha, run_dir,
            config_ids=CONFIG_IDS,
            shard_indices=SHARD_INDICES,
            instance_mode="replica_only",
        )
        failures = []
        for proc, handle, label in clients:
            rc = proc.wait()
            handle.close()
            print(f"[early-wave] client {label} exit={rc}", flush=True)
            if rc != 0:
                failures.append((label, rc))
        if failures:
            raise SystemExit(f"early-wave clients failed: {failures}; retry resumes")
        print("[early-wave] complete", flush=True)
    finally:
        for proc, handle, _ in clients:
            matrix.stop_process(proc)
            if not handle.closed:
                handle.close()
        for proc, handle, _, _ in servers:
            matrix.stop_process(proc)
            if not handle.closed:
                handle.close()
        monitor_stop.set()
        monitor.join(timeout=15)
        server_monitor_stop.set()
        if server_monitor is not None:
            server_monitor.join(timeout=15)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    # The detached supervisor holds this lock across memory-gate/server-start
    # retries.  Direct invocations acquire it here so they are equally safe.
    if os.environ.get("ROBOCASA_EARLY_WAVE_LOCK_HELD") == "1":
        run(args)
        return
    lock_path = run_dir / "early_wave.lock"
    with open(lock_path, "a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("another early wave or full unseen runner is active") from exc
        run(args)


if __name__ == "__main__":
    main()
