#!/usr/bin/env python3
"""Launch a reproducible RoboCasa365 configuration matrix.

The input spec is a small JSON document with one entry per configuration::

  {"configs": [{"id": "fp16", "gpu": 1, "port": 5661,
                "expected_wrapped": 0},
               {"id": "ckaonly", "gpu": 4, "port": 5664,
                "plan": "/abs/plan.json", "act_scale": "/abs/scales.npz",
                "expected_wrapped": 90}]}

For quantized configurations ``plan`` and ``act_scale`` are mandatory.
Optional ``atm`` plus ``ohb: true`` enables static ATM/OHB.  The runner writes
an immutable manifest, verifies the live server state, and starts balanced,
crash-tolerant client shards per configuration.  It supports all three target
checkpoint/task-set pairs used by the official 50-task benchmark.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import msgpack
import zmq

REPO = Path(__file__).resolve().parents[2]
GROOT_PY = Path("/home1/gyy/probe/miniforge3/envs/groot_test/bin/python")
DEFAULT_CHECKPOINT = REPO / (
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
COMPOSITE_SEEN_TASKS = [
    "DeliverStraw", "GetToastedBread", "KettleBoiling", "LoadDishwasher",
    "PackIdenticalLunches", "PreSoakPan", "PrepareCoffee", "RinseSinkBasin",
    "ScrubCuttingBoard", "SearingMeat", "SetUpCuttingStation",
    "StackBowlsCabinet", "SteamInMicrowave", "StirVegetables",
    "StoreLeftoversInBowl", "WashLettuce",
]
COMPOSITE_UNSEEN_TASKS = [
    "ArrangeBreadBasket", "ArrangeTea", "BreadSelection", "CategorizeCondiments",
    "CuttingToolSelection", "GarnishPancake", "GatherTableware",
    "HeatKebabSandwich", "MakeIceLemonade", "PanTransfer", "PortionHotDogs",
    "RecycleBottlesByType", "SeparateFreezerRack", "WaffleReheat",
    "WashFruitColander", "WeighIngredients",
]
TASK_SETS = {
    "atomic_seen": ATOMIC_TASKS,
    "composite_seen": COMPOSITE_SEEN_TASKS,
    "composite_unseen": COMPOSITE_UNSEEN_TASKS,
}
TASK_HORIZONS = {
    "CloseBlenderLid": 900, "CloseFridge": 900, "CloseToasterOvenDoor": 450,
    "CoffeeSetupMug": 600, "NavigateKitchen": 450, "OpenCabinet": 1050,
    "OpenDrawer": 750, "OpenStandMixerHead": 450,
    "PickPlaceCounterToCabinet": 750, "PickPlaceCounterToStove": 600,
    "PickPlaceDrawerToCounter": 750, "PickPlaceSinkToCounter": 900,
    "PickPlaceToasterToCounter": 600, "SlideDishwasherRack": 450,
    "TurnOffStove": 750, "TurnOnElectricKettle": 450,
    "TurnOnMicrowave": 450, "TurnOnSinkFaucet": 600,
    "DeliverStraw": 2550, "GetToastedBread": 3000, "KettleBoiling": 1500,
    "LoadDishwasher": 1800, "PackIdenticalLunches": 3900, "PreSoakPan": 2400,
    "PrepareCoffee": 1800, "RinseSinkBasin": 1350, "ScrubCuttingBoard": 1200,
    "SearingMeat": 4350, "SetUpCuttingStation": 2400, "StackBowlsCabinet": 2100,
    "SteamInMicrowave": 2100, "StirVegetables": 2400,
    "StoreLeftoversInBowl": 2550, "WashLettuce": 1650,
    "ArrangeBreadBasket": 4350, "ArrangeTea": 2250, "BreadSelection": 1950,
    "CategorizeCondiments": 1650, "CuttingToolSelection": 1200,
    "GarnishPancake": 2700, "GatherTableware": 2250,
    "HeatKebabSandwich": 2700, "MakeIceLemonade": 3000, "PanTransfer": 1800,
    "PortionHotDogs": 2250, "RecycleBottlesByType": 2850,
    "SeparateFreezerRack": 2400, "WaffleReheat": 4050,
    "WashFruitColander": 3150, "WeighIngredients": 3000,
}
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


def tree_artifact(path: str | None) -> dict | None:
    """Compact, content-addressed record for a DuQuant pack directory."""
    if not path:
        return None
    root = Path(path).resolve()
    if not root.is_dir():
        raise SystemExit(f"required pack directory missing: {root}")
    files = sorted(p for p in root.rglob("*") if p.is_file())
    h = hashlib.sha256()
    total = 0
    for file_path in files:
        rel = file_path.relative_to(root).as_posix().encode()
        digest = sha256_file(file_path)
        size = file_path.stat().st_size
        h.update(rel + b"\0" + digest.encode() + b"\0" + str(size).encode() + b"\n")
        total += size
    return {
        "path": str(root), "sha256_tree": h.hexdigest(),
        "files": len(files), "bytes": total,
    }


def balanced_shards(tasks: list[str], n_shards: int) -> list[list[str]]:
    n_shards = max(1, min(n_shards, len(tasks)))
    shards: list[list[str]] = [[] for _ in range(n_shards)]
    loads = [0 for _ in range(n_shards)]
    for task in sorted(tasks, key=lambda t: (-TASK_HORIZONS[t], t)):
        index = min(range(n_shards), key=lambda i: (loads[i], i))
        shards[index].append(task)
        loads[index] += TASK_HORIZONS[task]
    return shards


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--phase", choices=["smoke", "dev", "formal"], default="formal")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--task-set", choices=sorted(TASK_SETS), default="atomic_seen")
    p.add_argument("--tasks", default=None, help="Optional comma-separated task override.")
    p.add_argument("--dev-tasks", default=None, help="Tasks used for parameter selection.")
    p.add_argument("--n-shards", type=int, default=2)
    p.add_argument(
        "--egl-device-pool", default=None,
        help=("Optional comma-separated physical GPU indices assigned round-robin "
              "to client shards. Model-server placement is unchanged."),
    )
    p.add_argument("--trial-timeout", type=int, default=3600)
    p.add_argument("--trial-batch-size", type=int, default=5)
    p.add_argument(
        "--action-noise", choices=["paired", "native"], default="paired",
        help=("paired uses deterministic common-random-number diffusion noise; "
              "native uses the model server's ordinary RNG stream."),
    )
    p.add_argument("--gpu-sample-interval", type=float, default=10.0)
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


def gpu_free_memory_mib() -> dict[int, float]:
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free",
         "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()
    result = {}
    for row in rows:
        index, free = [value.strip() for value in row.split(",", 1)]
        result[int(index)] = float(free)
    return result


def egl_pool_memory_requirements(manifest: dict) -> dict[int, float]:
    """Conservative free-memory gate for shared EGL-client scheduling.

    The observed RoboCasa EGL contexts use roughly 1.5--2.0 GiB each.  Budget
    2.3 GiB/client, 12/16 GiB for FP16/quantized servers, and a 2 GiB guard.
    This does not evict existing jobs: the detached chain simply retries while
    a pool device lacks headroom.
    """
    pool = manifest["protocol"].get("egl_device_pool")
    if not pool:
        return {}
    client_counts = {int(gpu): 0 for gpu in pool}
    for devices in manifest["protocol"]["shard_egl_devices"].values():
        for gpu in devices:
            client_counts[int(gpu)] = client_counts.get(int(gpu), 0) + 1
    requirements = {
        gpu: 2048.0 + count * 2300.0 for gpu, count in client_counts.items()
    }
    for config in manifest["configs"]:
        server_budget = 16384.0 if int(config["expected_wrapped"]) else 12288.0
        server_gpus = [int(config["gpu"])] + [
            int(replica["gpu"]) for replica in config.get("replicas", [])
        ]
        for gpu in server_gpus:
            requirements[gpu] = requirements.get(gpu, 2048.0) + server_budget
    return requirements


def monitor_gpus(
    stop: threading.Event, path: Path, configs: list[dict], interval: float
) -> None:
    gpu_to_config = {}
    for config in configs:
        gpu_to_config[int(config["gpu"])] = config["id"]
        for replica in config.get("replicas", []):
            gpu_to_config[int(replica["gpu"])] = config["id"]
    query = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    with open(path, "a", encoding="utf-8", buffering=1) as handle:
        while not stop.is_set():
            sampled_at = datetime.now(timezone.utc).isoformat()
            proc = subprocess.run(query, text=True, stdout=subprocess.PIPE, check=False)
            if proc.returncode == 0:
                for line in proc.stdout.splitlines():
                    fields = [value.strip() for value in line.split(",")]
                    if len(fields) != 4:
                        continue
                    gpu = int(fields[0])
                    if gpu not in gpu_to_config:
                        continue
                    handle.write(json.dumps({
                        "sampled_at": sampled_at,
                        "config": gpu_to_config[gpu],
                        "gpu": gpu,
                        "memory_used_mib": float(fields[1]),
                        "utilization_gpu_pct": float(fields[2]),
                        "power_draw_w": float(fields[3]),
                    }) + "\n")
            stop.wait(max(1.0, interval))


def process_rss_mib(pid: int) -> float | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    except (FileNotFoundError, PermissionError, ValueError):
        return None
    return None


def monitor_server_processes(
    stop: threading.Event,
    path: Path,
    placements: list[dict],
    interval: float,
    source: str,
) -> None:
    """Record per-server CUDA memory independently of shared device load."""
    query = [
        "nvidia-smi",
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    with open(path, "a", encoding="utf-8", buffering=1) as handle:
        while not stop.is_set():
            sampled_at = datetime.now(timezone.utc).isoformat()
            proc = subprocess.run(query, text=True, stdout=subprocess.PIPE, check=False)
            memory_by_pid = {}
            if proc.returncode == 0:
                for line in proc.stdout.splitlines():
                    fields = [value.strip() for value in line.split(",", 1)]
                    if len(fields) != 2:
                        continue
                    try:
                        memory_by_pid[int(fields[0])] = float(fields[1])
                    except ValueError:
                        continue
            for placement in placements:
                pid = int(placement["pid"])
                if pid not in memory_by_pid:
                    continue
                handle.write(json.dumps({
                    "sampled_at": sampled_at,
                    "config": placement["config"],
                    "gpu": int(placement["gpu"]),
                    "server_pid": pid,
                    "instance_role": placement["instance_role"],
                    "source": source,
                    "server_memory_used_mib": memory_by_pid[pid],
                    "server_rss_mib": process_rss_mib(pid),
                }) + "\n")
            stop.wait(max(1.0, interval))


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
    spec_path: Path, run_dir: Path, phase: str, seeds: list[int], smoke_task: str,
    checkpoint: Path, task_set: str, task_override: str | None,
    dev_override: str | None, n_shards: int, trial_timeout: int,
    trial_batch_size: int, action_noise: str, gpu_sample_interval: float,
    egl_device_pool: list[int] | None = None,
) -> dict:
    spec = json.loads(spec_path.read_text())
    configs = spec.get("configs") or []
    if not configs:
        raise SystemExit("spec.configs is empty")
    ids = [c.get("id") for c in configs]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"duplicate config ids: {ids}")
    server_placements = []
    for config in configs:
        server_placements.append((str(config["id"]), int(config["gpu"]), int(config["port"])))
        for replica_index, replica in enumerate(config.get("replicas", []), 1):
            server_placements.append(
                (f"{config['id']}/r{replica_index}", int(replica["gpu"]), int(replica["port"]))
            )
    gpus = [gpu for _, gpu, _ in server_placements]
    ports = [port for _, _, port in server_placements]
    if any(g not in range(1, 8) for g in gpus):
        raise SystemExit(f"only GPUs 1-7 are authorized, got {gpus}")
    if len(gpus) != len(set(gpus)):
        raise SystemExit(f"each model-server instance must have its own GPU: {server_placements}")
    if len(ports) != len(set(ports)):
        raise SystemExit(f"model-server ports must be unique: {server_placements}")
    if trial_timeout < 1 or trial_batch_size < 1:
        raise SystemExit("trial timeout and batch size must be positive")

    registered_tasks = list(TASK_SETS[task_set])
    requested_tasks = (
        [v.strip() for v in task_override.split(",") if v.strip()]
        if task_override else registered_tasks
    )
    unknown = sorted(set(requested_tasks) - set(registered_tasks))
    if unknown:
        raise SystemExit(f"tasks do not belong to {task_set}: {unknown}")
    # ``--smoke-task`` is irrelevant to dev/formal runs.  Rejecting the
    # default atomic task while launching a composite formal matrix made the
    # otherwise valid composite manifest impossible to create.
    if phase == "smoke" and smoke_task not in registered_tasks:
        raise SystemExit(f"unknown --smoke-task {smoke_task}")
    if task_override:
        tasks = requested_tasks
    elif phase == "smoke":
        tasks = [smoke_task]
    elif phase == "dev" and task_set == "atomic_seen":
        tasks = list(DEV_TASKS)
    else:
        tasks = requested_tasks
    dev_tasks = (
        [v.strip() for v in dev_override.split(",") if v.strip()]
        if dev_override else (list(tasks) if phase == "dev" else [])
    )
    if not set(dev_tasks).issubset(tasks):
        raise SystemExit("--dev-tasks must be a subset of the evaluated tasks")
    run_seeds = [seeds[0]] if phase == "smoke" else seeds
    checkpoint = checkpoint.resolve()
    if not (checkpoint / "config.json").is_file():
        raise SystemExit(f"checkpoint missing config.json: {checkpoint}")
    enriched = []
    pack_cache: dict[str, dict | None] = {}
    for raw in configs:
        plan_artifact = artifact(raw.get("plan"))
        packdir = raw.get("packdir")
        if plan_artifact and not packdir:
            plan_doc = json.loads(Path(plan_artifact["path"]).read_text())
            values = list((plan_doc.get("packdirs") or {}).values())
            packdir = values[0] if values else None
        pack_key = str(Path(packdir).resolve()) if packdir else ""
        if pack_key not in pack_cache:
            pack_cache[pack_key] = tree_artifact(packdir)
        config = {
            "id": str(raw["id"]),
            "gpu": int(raw["gpu"]),
            "port": int(raw["port"]),
            "expected_wrapped": int(raw.get("expected_wrapped", 0)),
            "egl_device": int(raw.get("egl_device", raw["gpu"])),
            "plan": plan_artifact,
            "packdir": pack_cache[pack_key],
            "act_scale": artifact(raw.get("act_scale")),
            "atm": artifact(raw.get("atm")),
            "ohb": bool(raw.get("ohb", False)),
            "meta": raw.get("meta") or {},
        }
        replicas = [
            {"gpu": int(replica["gpu"]), "port": int(replica["port"])}
            for replica in raw.get("replicas", [])
        ]
        if replicas:
            config["replicas"] = replicas
        if config["expected_wrapped"] and not config["plan"]:
            raise SystemExit(f"{config['id']}: quantized config requires plan")
        if config["expected_wrapped"] and not config["act_scale"]:
            raise SystemExit(f"{config['id']}: quantized config requires act_scale")
        if config["expected_wrapped"] and not config["packdir"]:
            raise SystemExit(f"{config['id']}: quantized config requires packdir")
        if config["egl_device"] not in range(1, 8):
            raise SystemExit(f"{config['id']}: EGL device must be in GPU1-7")
        if config["atm"] is None and config["ohb"]:
            raise SystemExit(f"{config['id']}: OHB requires an ATM/OHB table")
        config["config_sha256"] = canonical_sha(config)
        enriched.append(config)

    shards = balanced_shards(tasks, n_shards)
    if egl_device_pool:
        if len(egl_device_pool) != len(set(egl_device_pool)):
            raise SystemExit("--egl-device-pool must contain unique GPU indices")
        if any(gpu not in range(1, 8) for gpu in egl_device_pool):
            raise SystemExit("--egl-device-pool is restricted to GPUs 1-7")
        for config_index, config in enumerate(enriched):
            offset = config_index * len(shards)
            config["shard_egl_devices"] = [
                egl_device_pool[(offset + shard_index) % len(egl_device_pool)]
                for shard_index in range(len(shards))
            ]
            # The shard-to-render-device schedule is experiment-defining and
            # therefore participates in each row's frozen config hash.
            config.pop("config_sha256", None)
            config["config_sha256"] = canonical_sha(config)
    for config in enriched:
        config["result_files"] = [
            str((run_dir / f"{config['id']}_s{i}.jsonl").resolve())
            for i in range(len(shards))
        ]
    protocol = {
        "split": "target",
        "official_task_horizons": True,
        "n_action_steps": 16,
        "denoising_steps": 4,
        "paired_action_noise": action_noise == "paired",
        "action_noise_scheme": (
            "sha256(task,env_seed,replan_index)/torch-cpu-normal-v1"
            if action_noise == "paired" else None
        ),
        "action_noise_mode": action_noise,
        "render": True,
        "egl_devices": {c["id"]: c["egl_device"] for c in enriched},
        "scenarios_per_task": len(run_seeds),
        "trial_timeout_seconds": trial_timeout,
        "trial_batch_size": trial_batch_size,
        "gpu_sample_interval_seconds": gpu_sample_interval,
    }
    if egl_device_pool:
        protocol["egl_device_pool"] = egl_device_pool
        protocol["shard_egl_devices"] = {
            config["id"]: config["shard_egl_devices"] for config in enriched
        }
        protocol["gpu_efficiency_scope"] = (
            "model-GPU device totals; replicas and EGL clients use shared GPUs 1-7"
        )
    if any(config.get("replicas") for config in enriched):
        protocol["server_instances"] = {
            config["id"]: server_instances(config) for config in enriched
        }
    return {
        "schema_version": 1,
        "run_id": run_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "repo": str(REPO),
        "spec": artifact(str(spec_path)),
        "checkpoint": artifact(str(checkpoint / "config.json")),
        "checkpoint_path": str(checkpoint),
        "data_config": DATA_CONFIG,
        "task_set": task_set,
        "tasks": tasks,
        "dev_tasks": dev_tasks,
        "heldout_tasks": [t for t in tasks if t not in dev_tasks],
        "seeds": run_seeds,
        "shards": shards,
        "task_horizons": {t: TASK_HORIZONS[t] for t in tasks},
        "comparisons": spec.get("comparisons") or [],
        "decision": spec.get("decision") or {},
        "protocol": protocol,
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


def wait_for_overlap_waves(run_dir: Path) -> None:
    """Serialize the full runner with its optional disjoint overlap waves.

    Lock files are persistent, but kernel locks are held only while a wave is
    active.  Stale files are therefore harmless.  The full runner waits for
    both waves before applying its hash-aware completion/resume pass.
    """
    for name in ("early_wave.lock", "primary_wave.lock"):
        lock_path = run_dir / name
        with open(lock_path, "a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def server_instances(config: dict) -> list[dict[str, int]]:
    return [
        {"gpu": int(config["gpu"]), "port": int(config["port"]), "replica": 0},
        *[
            {"gpu": int(replica["gpu"]), "port": int(replica["port"]),
             "replica": replica_index}
            for replica_index, replica in enumerate(config.get("replicas", []), 1)
        ],
    ]


def start_server(
    config: dict, instance: dict, manifest: dict, run_dir: Path
) -> tuple[subprocess.Popen, Any]:
    env = clean_server_env(os.environ)
    env["GR00T_GPU"] = str(instance["gpu"])
    env["GR00T_PORT"] = str(instance["port"])
    env["GR00T_DENOISING_STEPS"] = "4"
    env["GR00T_MODEL_PATH"] = manifest["checkpoint_path"]
    env["GR00T_DATA_CONFIG"] = manifest["data_config"]
    if config["plan"]:
        env["GR00T_DUQUANT_PLAN"] = config["plan"]["path"]
        env["GR00T_DUQUANT_PACKDIR"] = config["packdir"]["path"]
        env["GR00T_DUQUANT_ACT_SCALE_PATH"] = config["act_scale"]["path"]
        cmd = ["bash", str(QUANT_SERVER)]
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(instance["gpu"])
        cmd = [
            str(GROOT_PY), str(REPO / "scripts/inference_service.py"), "--server",
            "--model-path", manifest["checkpoint_path"], "--data-config", manifest["data_config"],
            "--embodiment-tag", "new_embodiment", "--port", str(instance["port"]),
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
    suffix = "" if int(instance["replica"]) == 0 else f"_r{instance['replica']}"
    log_handle = open(run_dir / f"server_{config['id']}{suffix}.log", "a", buffering=1)
    proc = subprocess.Popen(
        cmd, cwd=REPO, env=env, stdout=log_handle, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return proc, log_handle


def wait_and_verify(
    config: dict, instance: dict, manifest: dict, proc: subprocess.Popen, timeout: int
) -> dict:
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server {config['id']} exited with {proc.returncode}")
        try:
            info = call_endpoint(instance["port"], "get_runtime_info")
            if "error" in info:
                raise RuntimeError(info["error"])
            if int(info.get("wrapped_layers", -1)) != config["expected_wrapped"]:
                raise RuntimeError(
                    f"wrapped mismatch {info.get('wrapped_layers')} != {config['expected_wrapped']}"
                )
            if str(Path(info.get("model_path", "")).resolve()) != manifest["checkpoint_path"]:
                raise RuntimeError(f"checkpoint mismatch: {info}")
            expected_plan = config["plan"]["path"] if config["plan"] else None
            if info.get("plan") != expected_plan:
                raise RuntimeError(f"plan mismatch: {info}")
            expected_scale = config["act_scale"]["path"] if config["act_scale"] else None
            if info.get("act_scale_path") != expected_scale:
                raise RuntimeError(f"A8 scale mismatch: {info}")
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
    manifest: dict, manifest_sha: str, run_dir: Path,
    *,
    config_ids: set[str] | None = None,
    shard_indices: set[int] | None = None,
    instance_mode: str = "all",
) -> list[tuple[subprocess.Popen, Any, str]]:
    """Start incomplete client shards, rebalancing them over active instances.

    ``instance_mode=replica_only`` is used by the overlap wave that starts the
    unseen quantized half-matrix on GPUs 2/6/7 while seen is still running.
    The ordinary full runner uses all instances.  On resume, completed shard
    files are skipped and only remaining shards are round-robin redistributed.
    """
    children = []
    seed_arg = ",".join(str(s) for s in manifest["seeds"])
    for config in manifest["configs"]:
        if config_ids is not None and config["id"] not in config_ids:
            continue
        instances = server_instances(config)
        if instance_mode == "replica_only":
            instances = instances[1:]
            if not instances:
                raise RuntimeError(f"{config['id']}: replica-only execution has no replica")
        elif instance_mode == "primary_only":
            instances = instances[:1]
        elif instance_mode != "all":
            raise ValueError(f"unknown instance mode: {instance_mode}")
        pending_shards = []
        for shard_index, tasks in enumerate(manifest["shards"]):
            if shard_indices is not None and shard_index not in shard_indices:
                continue
            out_path = Path(config["result_files"][shard_index])
            completed = set()
            if out_path.exists():
                for line in out_path.read_text().splitlines():
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if (
                        row.get("success") is not None
                        and not row.get("crashed")
                        and row.get("config") == config["id"]
                        and row.get("manifest_sha256") == manifest_sha
                        and row.get("config_sha256") == config["config_sha256"]
                    ):
                        completed.add((row.get("task"), row.get("seed")))
            expected = {
                (task, seed) for task in tasks for seed in manifest["seeds"]
            }
            if completed >= expected:
                continue
            pending_shards.append((shard_index, tasks))
        for pending_index, (shard_index, tasks) in enumerate(pending_shards):
            out = config["result_files"][shard_index]
            shard_egl_devices = config.get("shard_egl_devices")
            egl_device = (
                int(shard_egl_devices[shard_index])
                if shard_egl_devices else int(config["egl_device"])
            )
            instance = instances[pending_index % len(instances)]
            cmd = [
                str(GROOT_PY), str(DRIVER), "--port", str(instance["port"]),
                "--config", config["id"], "--tasks", ",".join(tasks),
                "--n-trials", str(len(manifest["seeds"])),
                "--trial-seeds", seed_arg, "--max-steps", "0",
                "--manifest-sha256", manifest_sha,
                "--config-sha256", config["config_sha256"], "--out", out,
                "--trial-timeout", str(manifest["protocol"]["trial_timeout_seconds"]),
                "--trial-batch-size", str(manifest["protocol"]["trial_batch_size"]),
                "--egl-device", str(egl_device),
            ]
            if manifest["protocol"]["paired_action_noise"]:
                cmd.append("--paired-action-noise")
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
    wait_for_overlap_waves(run_dir)
    seeds = [int(v.strip()) for v in args.seeds.split(",") if v.strip()]
    if len(seeds) != len(set(seeds)) or not seeds:
        raise SystemExit("--seeds must contain unique integers")
    egl_device_pool = None
    if args.egl_device_pool:
        try:
            egl_device_pool = [
                int(value.strip())
                for value in args.egl_device_pool.split(",") if value.strip()
            ]
        except ValueError as exc:
            raise SystemExit("--egl-device-pool must be comma-separated integers") from exc
        if not egl_device_pool:
            raise SystemExit("--egl-device-pool must not be empty")

    manifest = build_manifest(
        spec_path, run_dir, args.phase, seeds, args.smoke_task,
        Path(args.checkpoint), args.task_set, args.tasks, args.dev_tasks,
        args.n_shards, args.trial_timeout, args.trial_batch_size,
        args.action_noise, args.gpu_sample_interval,
        egl_device_pool,
    )
    manifest, manifest_sha = write_or_validate_manifest(
        run_dir / "manifest.json", manifest
    )
    requested_gpus = [
        instance["gpu"]
        for config in manifest["configs"] for instance in server_instances(config)
    ]
    if not args.skip_gpu_preflight:
        busy = gpu_processes()
        conflicts = {g: busy.get(g, []) for g in requested_gpus if busy.get(g)}
        if conflicts:
            raise SystemExit(f"authorized GPUs are not idle; refusing to kill jobs: {conflicts}")
    memory_requirements = egl_pool_memory_requirements(manifest)
    if memory_requirements:
        free_memory = gpu_free_memory_mib()
        insufficient = {
            gpu: {"free_mib": free_memory.get(gpu), "required_mib": required}
            for gpu, required in memory_requirements.items()
            if free_memory.get(gpu, 0.0) < required
        }
        preflight = {
            "sampled_at": datetime.now(timezone.utc).isoformat(),
            "free_memory_mib": free_memory,
            "required_free_memory_mib": memory_requirements,
            "insufficient": insufficient,
            "policy": "no eviction; retry until every GPU has conservative headroom",
        }
        (run_dir / "gpu_preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")
        if insufficient:
            raise SystemExit(f"EGL pool lacks free memory; refusing eviction: {insufficient}")

    servers: list[tuple[subprocess.Popen, Any, dict, dict]] = []
    clients: list[tuple[subprocess.Popen, Any, str]] = []
    monitor_stop = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_gpus,
        args=(
            monitor_stop, run_dir / "gpu_efficiency.jsonl", manifest["configs"],
            args.gpu_sample_interval,
        ),
        daemon=True,
    )
    monitor_thread.start()
    server_monitor_stop = threading.Event()
    server_monitor_thread = None
    try:
        for config in manifest["configs"]:
            for instance in server_instances(config):
                proc, handle = start_server(config, instance, manifest, run_dir)
                servers.append((proc, handle, config, instance))
                print(
                    f"[matrix] server {config['id']}/r{instance['replica']} pid={proc.pid} "
                    f"gpu={instance['gpu']} port={instance['port']}"
                )
        server_monitor_thread = threading.Thread(
            target=monitor_server_processes,
            args=(
                server_monitor_stop,
                run_dir / "gpu_server_efficiency.jsonl",
                [
                    {
                        "config": config["id"],
                        "gpu": instance["gpu"],
                        "pid": proc.pid,
                        "instance_role": (
                            "primary" if int(instance["replica"]) == 0
                            else f"replica_{instance['replica']}"
                        ),
                    }
                    for proc, _, config, instance in servers
                ],
                args.gpu_sample_interval,
                "full_matrix",
            ),
            daemon=True,
        )
        server_monitor_thread.start()
        runtime = {}
        for proc, _, config, instance in servers:
            runtime_key = (
                config["id"] if int(instance["replica"]) == 0
                else f"{config['id']}/r{instance['replica']}"
            )
            runtime[runtime_key] = wait_and_verify(
                config, instance, manifest, proc, args.server_timeout
            )
            print(f"[matrix] verified {runtime_key}: {runtime[runtime_key]}")
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
        # Client drivers are process-group leaders and their RoboCasa children
        # inherit that group.  Stop them on Ctrl-C/server failure as well as on
        # normal completion; otherwise orphaned clients keep rendering and
        # retrying against a server that the block below has already stopped.
        for proc, handle, _ in clients:
            stop_process(proc)
            if not handle.closed:
                handle.close()
        for proc, handle, _, _ in servers:
            if not args.keep_servers:
                stop_process(proc)
            if not handle.closed:
                handle.close()
        monitor_stop.set()
        monitor_thread.join(timeout=max(5.0, args.gpu_sample_interval + 2.0))
        server_monitor_stop.set()
        if server_monitor_thread is not None:
            server_monitor_thread.join(
                timeout=max(5.0, args.gpu_sample_interval + 2.0)
            )


if __name__ == "__main__":
    main()
