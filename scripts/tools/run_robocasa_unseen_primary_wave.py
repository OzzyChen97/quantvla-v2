#!/usr/bin/env python3
"""Run the complementary unseen shards after composite_seen completes.

This wave can overlap safely with ``run_robocasa_unseen_early_wave.py``:

* FP16 runs all eight shards on its primary GPU (it has no early-wave rows).
* Each quantized primary runs even shards 0/2/4/6.
* The early replicas run only odd quantized shards 1/3/5/7.

Together the two waves cover the immutable 4-config x 8-shard matrix exactly
once while keeping GPUs 1-7 occupied.  The ordinary full runner subsequently
performs strict resume/validation and starts only genuinely missing shards.
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
EGL_POOL = [1, 2, 3, 4, 5, 6, 7]
CONFIG_SHARDS = {
    "fp16": set(range(8)),
    "w4a8_atmohb": {0, 2, 4, 6},
    "cscka_final": {0, 2, 4, 6},
    "cscka_final_atmohb": {0, 2, 4, 6},
}


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


def selected_instances(manifest: dict) -> list[tuple[dict, dict, set[int]]]:
    selected = []
    for config in manifest["configs"]:
        config_id = config["id"]
        if config_id not in CONFIG_SHARDS:
            continue
        primary = matrix.server_instances(config)[0]
        selected.append((config, primary, CONFIG_SHARDS[config_id]))
    if {config["id"] for config, _, _ in selected} != set(CONFIG_SHARDS):
        raise RuntimeError("unseen manifest does not contain all primary-wave configs")
    return selected


def memory_requirements(
    selected: list[tuple[dict, dict, set[int]]]
) -> dict[int, float]:
    client_counts = {gpu: 0 for gpu in EGL_POOL}
    for config, _, shard_indices in selected:
        devices = config["shard_egl_devices"]
        for shard_index in shard_indices:
            client_counts[int(devices[shard_index])] += 1
    required = {
        gpu: 2048.0 + count * 2300.0 for gpu, count in client_counts.items()
    }
    for config, instance, _ in selected:
        server_budget = 16384.0 if config["expected_wrapped"] else 12288.0
        gpu = int(instance["gpu"])
        required[gpu] = required.get(gpu, 2048.0) + server_budget
    return required


def run(args: argparse.Namespace) -> None:
    manifest, manifest_sha, run_dir = build(args)
    selected = selected_instances(manifest)
    required = memory_requirements(selected)
    free = matrix.gpu_free_memory_mib()
    insufficient = {
        gpu: {"free_mib": free.get(gpu), "required_mib": need}
        for gpu, need in required.items() if free.get(gpu, 0.0) < need
    }
    preflight = {
        "sampled_at": datetime.now(timezone.utc).isoformat(),
        "selected_shards": {
            config["id"]: sorted(shards) for config, _, shards in selected
        },
        "free_memory_mib": free,
        "required_free_memory_mib": required,
        "insufficient": insufficient,
    }
    (run_dir / "primary_wave_preflight.json").write_text(
        json.dumps(preflight, indent=2) + "\n"
    )
    print(json.dumps(preflight, indent=2), flush=True)
    if args.dry_run:
        print(
            "[primary-wave] dry-run PASS "
            "(memory gate is enforced when the post-seen wave starts)",
            flush=True,
        )
        return
    if insufficient:
        raise SystemExit(f"primary-wave memory preflight failed: {insufficient}")

    servers = []
    clients = []
    monitor_stop = threading.Event()
    monitor_configs = [
        {"id": config["id"], "gpu": instance["gpu"]}
        for config, instance, _ in selected
    ]
    monitor = threading.Thread(
        target=matrix.monitor_gpus,
        args=(monitor_stop, run_dir / "gpu_efficiency.jsonl", monitor_configs, 10.0),
        daemon=True,
    )
    monitor.start()
    try:
        runtime = {}
        for config, instance, _ in selected:
            proc, handle = matrix.start_server(config, instance, manifest, run_dir)
            servers.append((proc, handle, config, instance))
        for proc, _, config, instance in servers:
            key = f"{config['id']}/primary"
            runtime[key] = matrix.wait_and_verify(
                config, instance, manifest, proc, args.server_timeout
            )
            print(f"[primary-wave] verified {key}", flush=True)
        (run_dir / "runtime_info_primary_wave.json").write_text(
            json.dumps(runtime, indent=2) + "\n"
        )
        for config, _, shard_indices in selected:
            clients.extend(matrix.start_clients(
                manifest, manifest_sha, run_dir,
                config_ids={config["id"]},
                shard_indices=shard_indices,
                instance_mode="primary_only",
            ))
        failures = []
        for proc, handle, label in clients:
            rc = proc.wait()
            handle.close()
            print(f"[primary-wave] client {label} exit={rc}", flush=True)
            if rc != 0:
                failures.append((label, rc))
        if failures:
            raise SystemExit(f"primary-wave clients failed: {failures}; retry resumes")
        print("[primary-wave] complete", flush=True)
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


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if os.environ.get("ROBOCASA_PRIMARY_WAVE_LOCK_HELD") == "1":
        run(args)
        return
    lock_path = run_dir / "primary_wave.lock"
    with open(lock_path, "a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("another primary unseen wave is active") from exc
        run(args)


if __name__ == "__main__":
    main()
