#!/usr/bin/env python3
"""RoboCasa365 GR00T N1.5 eval client (QuantVLA v1.4, Stage D).

Runs in the robocasa365 conda env and talks to a GR00T inference service
(scripts/inference_service.py) loaded with a target_posttraining checkpoint
and --data-config examples.RoboCasa365.custom_data_config:RoboCasa365DataConfig.

CRITICAL: import robocasa BEFORE touching sys.path — putting <repo>/code on
sys.path first shadows the real robocasa package with the repo directory and
kills task registration (396 -> 19 envs, D-023).

Metrics (v1): per-trial full-task success + episode length + failure step.
Subgoal-level metrics (avg completed subgoals, P(>=k), transition delay)
require the benchmark's per-stage annotations and land in a follow-up.

Usage:
    # server (groot_test env):
    #   python scripts/inference_service.py --model_path <robocasa365 ckpt> \
    #       --data-config examples.RoboCasa365.custom_data_config:RoboCasa365DataConfig --port 5570
    # client (robocasa365 env):
    python scripts/run_robocasa365_gr00t_eval.py --port 5570 \
        --task-set atomic_seen --n-trials 3 --tasks AddIceCubes,PrepareSmoothie \
        --out runs/robocasa365_eval/atomic_seen_smoke.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import robocasa  # noqa: F401  — MUST precede any sys.path change (D-023)
from robocasa.utils.dataset_registry import TARGET_TASKS  # noqa: E402
from robocasa.utils.dataset_registry_utils import get_task_horizon  # noqa: E402
from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]  # scripts/ -> repo root
sys.path.insert(0, str(REPO_ROOT / "code"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "tools"))

import numpy as np  # noqa: E402

# obs keys the RoboCasa365DataConfig consumes — filter the wrapper's obs
# (which also emits legacy res256/res512 aliases and extra state keys) so the
# server-side transforms only see configured modalities
SEND_VIDEO_KEYS = [
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
]
SEND_STATE_KEYS = [
    "state.end_effector_position_relative",
    "state.end_effector_rotation_relative",
    "state.gripper_qpos",
    "state.base_position",
    "state.base_rotation",
]
SEND_LANG_KEYS = ["annotation.human.task_description"]

ACTION_DIMS = {
    "end_effector_position": 3,
    "end_effector_rotation": 3,
    "gripper_close": 1,
    "base_motion": 4,
    "control_mode": 1,
}

ACTION_NOISE_SCHEME = "sha256(task,env_seed,replan_index)/torch-cpu-normal-v1"


def action_noise_seed(task: str, env_seed: int, replan_index: int) -> int:
    """Stable request seed, independent of Python hash randomization/sharding."""
    payload = f"quantvla-robocasa365-v1\0{task}\0{env_seed}\0{replan_index}".encode()
    # torch.Generator.manual_seed accepts signed 64-bit seeds portably.
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _normalize_action_chunks(action_chunk: dict) -> dict[str, np.ndarray]:
    """Return server actions as per-key ``(horizon, action_dim)`` arrays.

    ``inference_service.py`` normally removes the batch dimension and returns
    ``(16, D)``. Some service implementations preserve it as ``(1, 16, D)``;
    accepting both avoids confusing the horizon with a batch dimension.
    """
    chunks: dict[str, np.ndarray] = {}
    for key, expected_dim in ACTION_DIMS.items():
        value = np.asarray(action_chunk[f"action.{key}"])
        while value.ndim > 2 and value.shape[0] == 1:
            value = value[0]
        if value.ndim == 1:
            if expected_dim == 1:
                value = value[:, np.newaxis]
            elif value.shape[0] == expected_dim:
                value = value[np.newaxis, :]
        if value.ndim != 2 or value.shape[1] != expected_dim:
            raise ValueError(
                f"Unexpected action.{key} shape {value.shape}; expected (H, {expected_dim})"
            )
        chunks[key] = value
    return chunks


class _Gr00tZMQClient:
    """Minimal ZMQ REQ client for the GR00T inference service.

    Self-contained (zmq + msgpack + numpy only) so the robocasa365 conda env
    does not need the heavy gr00t data-stack dependencies. Protocol mirrors
    gr00t.eval.service.BaseInferenceClient: msgpack {"endpoint", "data"},
    15s recv/send timeouts.
    """

    def __init__(self, host: str = "localhost", port: int = 5570, timeout_ms: int = 15000):
        import zmq
        import msgpack

        self.msgpack = msgpack
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.sock.connect(f"tcp://{host}:{port}")

    @staticmethod
    def _encode(obj):
        if isinstance(obj, np.ndarray):
            import io

            out = io.BytesIO()
            np.save(out, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": out.getvalue()}
        return obj

    @staticmethod
    def _decode(obj):
        if isinstance(obj, dict) and "__ndarray_class__" in obj:
            import io

            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

    def call(self, endpoint: str, data: dict | None = None) -> dict:
        import zmq

        request = {"endpoint": endpoint, "data": data or {}}
        self.sock.send(self.msgpack.packb(request, default=self._encode))
        try:
            resp = self.msgpack.unpackb(self.sock.recv(), object_hook=self._decode)
        except zmq.Again:
            raise RuntimeError(f"inference server timeout on {endpoint}") from None
        if "error" in resp:
            raise RuntimeError(f"Server error: {resp['error']}")
        return resp

    def get_action(self, obs: dict, action_seed: int | None = None) -> dict:
        if action_seed is None:
            return self.call("get_action", obs)
        return self.call(
            "get_action_seeded",
            {"observations": obs, "action_seed": int(action_seed)},
        )

    def runtime_info(self) -> dict:
        return self.call("get_runtime_info")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RoboCasa365 GR00T eval client")
    p.add_argument("--port", type=int, default=5570)
    p.add_argument("--task-set", default="atomic_seen",
                   choices=["atomic_seen", "composite_seen", "composite_unseen", "custom"])
    p.add_argument("--tasks", default=None, help="comma list overriding the task set")
    p.add_argument("--split", default="target")
    p.add_argument("--n-trials", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--exact-seed",
        type=int,
        default=None,
        help=("Use this exact environment seed. This mode is reserved for the "
              "per-trial crash-tolerant driver and requires exactly one task "
              "and one trial."),
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Episode horizon override; 0 uses RoboCasa's official per-task horizon.",
    )
    p.add_argument(
        "--n-action-steps",
        type=int,
        default=16,
        help="Number of predicted action-chunk steps to execute before replanning.",
    )
    p.add_argument(
        "--paired-action-noise",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=("Use deterministic common-random-number diffusion noise keyed by "
              "task/environment-seed/replan-index."),
    )
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    else:
        tasks = list(TARGET_TASKS[args.task_set])
    if args.exact_seed is not None and (len(tasks) != 1 or args.n_trials != 1):
        raise SystemExit("--exact-seed requires exactly one task and --n-trials 1")
    client = _Gr00tZMQClient(host="localhost", port=args.port)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    t0 = time.time()
    for ti, task in enumerate(tasks):
        # enable_render=True is REQUIRED: False zero-fills camera obs and the
        # policy runs blind (D-040).
        # D-041: concurrent EGL context creation on one device deadlocks the
        # NVIDIA EGL driver (10 stuck constructions, 44s CPU / 9min wall, no
        # sockets). Serialize env construction across processes with a repo-
        # local flock (repo path, NOT /tmp — /tmp is per-launcher tmpfs).
        import fcntl
        lock_path = REPO_ROOT / "runs" / ".robocasa365_construct.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "w") as _lf:
            fcntl.flock(_lf, fcntl.LOCK_EX)
            # Use RoboCasa's own GR00T wrapper. It is the environment used by
            # the official Isaac-GR00T run_eval.py protocol and already emits
            # the exact PandaOmron keys expected by the checkpoint.
            env = RoboCasaGymEnv(
                env_name=task,
                enable_render=True,
                split=args.split,
            )
            fcntl.flock(_lf, fcntl.LOCK_UN)
        task_max_steps = args.max_steps or get_task_horizon(task)
        for trial in range(args.n_trials):
            seed = (args.exact_seed if args.exact_seed is not None
                    else args.seed * 1000 + ti * 10 + trial)
            obs, _ = env.reset(seed=seed)
            done = False
            steps = 0
            replan_index = 0
            success = False
            while not done and steps < task_max_steps:
                send_obs = {}
                for k in SEND_VIDEO_KEYS + SEND_STATE_KEYS + SEND_LANG_KEYS:
                    if k not in obs:
                        continue
                    v = obs[k]
                    if k.startswith("video."):
                        # VideoToTensor expects (T, H, W, C) sequences
                        v = np.asarray(v)
                        if v.ndim == 3:
                            v = v[np.newaxis, ...]
                    elif k.startswith("state."):
                        # state values carry (T, D) batch dims like the LIBERO
                        # obs format
                        v = np.asarray(v, dtype=np.float32)
                        if v.ndim == 1:
                            v = v[np.newaxis, :]
                    elif isinstance(v, str):
                        v = [v]  # language values travel as lists
                    send_obs[k] = v
                noise_seed = (
                    action_noise_seed(task, seed, replan_index)
                    if args.paired_action_noise else None
                )
                action_chunk = client.get_action(send_obs, action_seed=noise_seed)
                replan_index += 1
                chunks = _normalize_action_chunks(action_chunk)
                chunk_len = min(
                    args.n_action_steps,
                    task_max_steps - steps,
                    *(len(value) for value in chunks.values()),
                )
                for action_step in range(chunk_len):
                    action = {
                        f"action.{key}": np.atleast_1d(value[action_step])
                        for key, value in chunks.items()
                    }
                    obs, reward, done, truncated, info = env.step(action)
                    steps += 1
                    success = bool(info.get("success", False))
                    if success or done or truncated:
                        done = True
                        break
            results.append({
                "task": task, "trial": trial, "seed": seed,
                "success": success, "steps": steps,
                "max_steps": task_max_steps,
                "n_action_steps": args.n_action_steps,
                "replans": replan_index,
                "paired_action_noise": args.paired_action_noise,
                "action_noise_scheme": (
                    ACTION_NOISE_SCHEME if args.paired_action_noise else None
                ),
            })
            print(f"[robocasa365-eval] {task} trial {trial}: success={success} "
                  f"steps={steps} ({time.time() - t0:.0f}s)", flush=True)
            with open(out_path, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")
        env.close()
    n_ok = sum(1 for r in results if r["success"])
    print(f"[robocasa365-eval] done: {n_ok}/{len(results)} episodes "
          f"({n_ok / len(results):.1%}) -> {out_path}")


if __name__ == "__main__":
    main()
