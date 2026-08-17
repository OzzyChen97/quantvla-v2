#!/usr/bin/env python3
"""Launch a reproducible RoboCasa365 atomic_seen configuration matrix.

The input spec is a small JSON document with one entry per configuration::

  {"configs": [{"id": "fp16", "gpu": 1, "port": 5661,
                "expected_wrapped": 0},
               {"id": "ckaonly", "gpu": 4, "port": 5664,
                "plan": "/abs/plan.json", "act_scale": "/abs/scales.npz",
                "expected_wrapped": 90}]}

For quantized configurations ``plan`` and ``act_scale`` are mandatory.
Optional ``atm`` plus ``ohb: true`` enables static ATM/OHB.  The runner writes
an immutable manifest, verifies the live server state, and starts two
crash-tolerant client shards per configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import msgpack
import zmq

REPO = Path(__file__).resolve().parents[2]
GROOT_PY = Path("/home1/gyy/probe/miniforge3/envs/groot_test/bin/python")
CHECKPOINT = REPO / (
    "checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/"
    "target_posttraining/atomic_seen/checkpoint-60000"
)
DATA_CONFIG = "examples.RoboCasa365.custom_data_config:RoboCasa365DataConfig"
DRIVER = REPO / "scripts/tools/run_crit4_trial_driver.py"
QUANT_SERVER = REPO / "scripts/run_robocasa365_quant_serve.sh"

ATOMIC_TASKS = [
    "CloseBlenderLid",
    "CloseFridge",
    "CloseToasterOvenDoor",
    "CoffeeSetupMug",
    "NavigateKitchen",
    "OpenCabinet",
    "OpenDrawer",
    "OpenStandMixerHead",
    "PickPlaceCounterToCabinet",
    "PickPlaceCounterToStove",
    "PickPlaceDrawerToCounter",
    "PickPlaceSinkToCounter",
    "PickPlaceToasterToCounter",
    "SlideDishwasherRack",
    "TurnOffStove",
    "TurnOnElectricKettle",
    "TurnOnMicrowave",
    "TurnOnSinkFaucet",
]
DEV_TASKS = [
    "OpenCabinet",
    "OpenStandMixerHead",
    "PickPlaceDrawerToCounter",
    "CoffeeSetupMug",
]


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(obj: Any) -> str:
    data = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def artifact(path: str | None) -> dict | None:
    if not path:
        return None
    p = Path(path).resolve()
    if not p.is_file():
        raise SystemExit(f"required artifact missing: {p}")
    return {"path": str(p), "sha256": sha256_file(p), "bytes": p.stat().st_size}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--phase", choices=["smoke", "dev", "formal"], default="formal")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--server-timeout", type=int, default=1200)
    p.add_argument("--smoke-task", default="OpenCabinet")
    p.add_argument("--skip-gpu-preflight", action="store_true")
    p.add_argument("--keep-servers", action="store_true")
    return p.parse_args()


def gpu_processes() -> dict[int, list[dict[str, str]]]:
    gpu_rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()
    uuid_to_index = {}
    for row in gpu_rows:
        index, uuid = [x.strip() for x in row.split(",", 1)]
        uuid_to_index[uuid] = int(index)
    result = {index: [] for index in uuid_to_index.values()}
    proc = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory,process_name",
         "--format=csv,noheader,nounits"],
        text=True,
        stdout=subprocess.PIPE,
        check=False,
    )
    for row in proc.stdout.splitlines():
        fields = [x.strip() for x in row.split(",", 3)]
        if len(fields) != 4 or fields[0] not in uuid_to_index:
            continue
        index = uuid_to_index[fields[0]]
        result[index].append(
            {"pid": fields[1], "used_memory_mib": fields[2], "process": fields[3]}
        )
    return result


def call_endpoint(port: int, endpoint: str, timeout_ms: int = 3000) -> dict:
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
    sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
    sock.connect(f"tcp://127.0.0.1:{port}")
    try:
        sock.send(msgpack.packb({"endpoint": endpoint, "data": {}}))
        return msgpack.unpackb(sock.recv(), raw=False)
    finally:
        sock.close(linger=0)
        ctx.term()


def clean_server_env(base: dict[str, str]) -> dict[str, str]:
    env = dict(base)
    prefixes = (
        "GR00T_DUQUANT_", "GR00T_ATM_", "GR00T_OHB_", "GR00T_DENOISING_STEPS"
    )
    for key in list(env):
        if key.startswith(prefixes):
            env.pop(key, None)
    env["PYTHONPATH"] = f"{REPO / 'code'}:{REPO / 'scripts/tools'}:{env.get('PYTHONPATH', '')}"
    return env


def build_manifest(
    spec_path: Path, run_dir: Path, phase: str, seeds: list[int], smoke_task: str
) -> dict:
    spec = json.loads(spec_path.read_text())
    configs = spec.get("configs") or []
    if not configs:
        raise SystemExit("spec.configs is empty")
    ids = [c.get("id") for c in configs]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"duplicate config ids: {ids}")
    gpus = [int(c["gpu"]) for c in configs]
    if any(g not in range(1, 7) for g in gpus):
        raise SystemExit(f"only GPUs 1-6 are authorized, got {gpus}")
    if len(gpus) != len(set(gpus)):
        raise SystemExit("each formal configuration must have its own GPU")

    if smoke_task not in ATOMIC_TASKS:
        raise SystemExit(f"unknown --smoke-task {smoke_task}")
    tasks = (
        [smoke_task] if phase == "smoke"
        else list(DEV_TASKS) if phase == "dev"
        else list(ATOMIC_TASKS)
    )
    run_seeds = [seeds[0]] if phase == "smoke" else seeds
    enriched = []
    for raw in configs:
        config = {
            "id": str(raw["id"]),
            "gpu": int(raw["gpu"]),
            "port": int(raw["port"]),
            "expected_wrapped": int(raw.get("expected_wrapped", 0)),
            "plan": artifact(raw.get("plan")),
            "act_scale": artifact(raw.get("act_scale")),
            "atm": artifact(raw.get("atm")),
            "ohb": bool(raw.get("ohb", False)),
        }
        if config["expected_wrapped"] and not config["plan"]:
            raise SystemExit(f"{config['id']}: quantized config requires plan")
        if config["expected_wrapped"] and not config["act_scale"]:
            raise SystemExit(f"{config['id']}: quantized config requires act_scale")
        if config["atm"] is None and config["ohb"]:
            raise SystemExit(f"{config['id']}: OHB requires an ATM/OHB table")
        config["config_sha256"] = canonical_sha(config)
        enriched.append(config)

    shards = [tasks[::2], tasks[1::2]] if len(tasks) > 1 else [tasks]
    for config in enriched:
        config["result_files"] = [
            str((run_dir / f"{config['id']}_s{i}.jsonl").resolve())
            for i in range(len(shards))
        ]
    return {
        "schema_version": 1,
        "run_id": run_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "repo": str(REPO),
        "spec": artifact(str(spec_path)),
        "checkpoint": artifact(str(CHECKPOINT / "config.json")),
        "checkpoint_path": str(CHECKPOINT.resolve()),
        "data_config": DATA_CONFIG,
        "task_set": "atomic_seen",
        "tasks": tasks,
        "dev_tasks": [t for t in DEV_TASKS if t in tasks],
        "heldout_tasks": [t for t in tasks if t not in DEV_TASKS],
        "seeds": run_seeds,
        "shards": shards,
        "protocol": {
            "split": "target",
            "official_task_horizons": True,
            "n_action_steps": 16,
            "denoising_steps": 4,
            "paired_action_noise": True,
            "action_noise_scheme": "sha256(task,env_seed,replan_index)/torch-cpu-normal-v1",
            "render": True,
            "egl_device": 3,
        },
        "configs": enriched,
    }


def write_or_validate_manifest(path: Path, manifest: dict) -> tuple[dict, str]:
    if path.exists():
        saved = json.loads(path.read_text())
        # Timestamps are intentionally ignored; all experiment-defining data
        # must match the already frozen manifest.
        proposed = dict(manifest)
        proposed["created_at"] = saved.get("created_at")
        if saved != proposed:
            raise SystemExit(f"immutable manifest mismatch: {path}")
        manifest = saved
    else:
        path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest, sha256_file(path)


def start_server(config: dict, run_dir: Path) -> tuple[subprocess.Popen, Any]:
    env = clean_server_env(os.environ)
    env["GR00T_GPU"] = str(config["gpu"])
    env["GR00T_PORT"] = str(config["port"])
    env["GR00T_DENOISING_STEPS"] = "4"
    env["GR00T_MODEL_PATH"] = str(CHECKPOINT)
    env["GR00T_DATA_CONFIG"] = DATA_CONFIG
    if config["plan"]:
        env["GR00T_DUQUANT_PLAN"] = config["plan"]["path"]
        env["GR00T_DUQUANT_ACT_SCALE_PATH"] = config["act_scale"]["path"]
        cmd = ["bash", str(QUANT_SERVER)]
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(config["gpu"])
        cmd = [
            str(GROOT_PY), str(REPO / "scripts/inference_service.py"), "--server",
            "--model-path", str(CHECKPOINT), "--data-config", DATA_CONFIG,
            "--embodiment-tag", "new_embodiment", "--port", str(config["port"]),
            "--denoising-steps", "4",
        ]
    if config["atm"]:
        env["GR00T_ATM_ENABLE"] = "1"
        env["GR00T_OHB_ENABLE"] = "1" if config["ohb"] else "0"
        env["GR00T_ATM_ALPHA_PATH"] = config["atm"]["path"]
        env["GR00T_ATM_PER_STEP"] = "0"
    else:
        env["GR00T_ATM_ENABLE"] = "0"
        env["GR00T_OHB_ENABLE"] = "0"
        env.pop("GR00T_ATM_ALPHA_PATH", None)
    log_handle = open(run_dir / f"server_{config['id']}.log", "a", buffering=1)
    proc = subprocess.Popen(
        cmd, cwd=REPO, env=env, stdout=log_handle, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return proc, log_handle


def wait_and_verify(config: dict, proc: subprocess.Popen, timeout: int) -> dict:
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server {config['id']} exited with {proc.returncode}")
        try:
            info = call_endpoint(config["port"], "get_runtime_info")
            if "error" in info:
                raise RuntimeError(info["error"])
            if int(info.get("wrapped_layers", -1)) != config["expected_wrapped"]:
                raise RuntimeError(
                    f"wrapped mismatch {info.get('wrapped_layers')} != {config['expected_wrapped']}"
                )
            expect_atm = config["atm"] is not None
            if bool(info.get("atm_enabled")) != expect_atm:
                raise RuntimeError(f"ATM enable mismatch: {info}")
            if expect_atm and (int(info.get("atm_layers", 0)) == 0 or
                               int(info.get("ohb_layers", 0)) == 0):
                raise RuntimeError(f"ATM/OHB hooks absent: {info}")
            return info
        except Exception as exc:  # readiness includes connection timeouts
            last_error = exc
            time.sleep(3)
    raise RuntimeError(f"server {config['id']} not ready: {last_error}")


def start_clients(
    manifest: dict, manifest_sha: str, run_dir: Path
) -> list[tuple[subprocess.Popen, Any, str]]:
    children = []
    seed_arg = ",".join(str(s) for s in manifest["seeds"])
    for config in manifest["configs"]:
        for shard_index, tasks in enumerate(manifest["shards"]):
            out = config["result_files"][shard_index]
            cmd = [
                str(GROOT_PY), str(DRIVER), "--port", str(config["port"]),
                "--config", config["id"], "--tasks", ",".join(tasks),
                "--n-trials", str(len(manifest["seeds"])),
                "--trial-seeds", seed_arg, "--max-steps", "0",
                "--paired-action-noise", "--manifest-sha256", manifest_sha,
                "--config-sha256", config["config_sha256"], "--out", out,
            ]
            log_handle = open(
                run_dir / f"driver_{config['id']}_s{shard_index}.log", "a", buffering=1
            )
            proc = subprocess.Popen(
                cmd, cwd=REPO, stdout=log_handle, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            children.append((proc, log_handle, f"{config['id']}/s{shard_index}"))
    return children


def stop_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)


def main() -> None:
    args = parse_args()
    spec_path = Path(args.spec).resolve()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(v.strip()) for v in args.seeds.split(",") if v.strip()]
    if len(seeds) != len(set(seeds)) or not seeds:
        raise SystemExit("--seeds must contain unique integers")

    manifest = build_manifest(spec_path, run_dir, args.phase, seeds, args.smoke_task)
    manifest, manifest_sha = write_or_validate_manifest(
        run_dir / "manifest.json", manifest
    )
    requested_gpus = [c["gpu"] for c in manifest["configs"]]
    if not args.skip_gpu_preflight:
        busy = gpu_processes()
        conflicts = {g: busy.get(g, []) for g in requested_gpus if busy.get(g)}
        if conflicts:
            raise SystemExit(f"authorized GPUs are not idle; refusing to kill jobs: {conflicts}")

    servers: list[tuple[subprocess.Popen, Any, dict]] = []
    clients: list[tuple[subprocess.Popen, Any, str]] = []
    try:
        for config in manifest["configs"]:
            proc, handle = start_server(config, run_dir)
            servers.append((proc, handle, config))
            print(f"[matrix] server {config['id']} pid={proc.pid} gpu={config['gpu']} port={config['port']}")
        runtime = {}
        for proc, _, config in servers:
            runtime[config["id"]] = wait_and_verify(config, proc, args.server_timeout)
            print(f"[matrix] verified {config['id']}: {runtime[config['id']]}")
        (run_dir / "runtime_info.json").write_text(json.dumps(runtime, indent=2) + "\n")

        clients = start_clients(manifest, manifest_sha, run_dir)
        failures = []
        for proc, handle, label in clients:
            rc = proc.wait()
            handle.close()
            if rc != 0:
                failures.append((label, rc))
            print(f"[matrix] client {label} exit={rc}")
        if failures:
            raise SystemExit(f"matrix clients failed: {failures}; rerun resumes by task/seed")
        print(f"[matrix] complete: {len(manifest['tasks'])} tasks x "
              f"{len(manifest['seeds'])} seeds x {len(manifest['configs'])} configs")
    finally:
        for proc, handle, _ in servers:
            if not args.keep_servers:
                stop_process(proc)
            handle.close()


if __name__ == "__main__":
    main()
