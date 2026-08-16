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
import json
import sys
import time
from pathlib import Path

import robocasa  # noqa: F401  — MUST precede any sys.path change (D-023)
from robocasa.utils.dataset_registry import TARGET_TASKS  # noqa: E402

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
    "state.base_position",
    "state.base_rotation",
    "state.end_effector_position_relative",
    "state.end_effector_rotation_relative",
    "state.gripper_qpos",
]
SEND_LANG_KEYS = ["annotation.human.action.task_description"]


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

    def get_action(self, obs: dict) -> dict:
        import zmq

        request = {"endpoint": "get_action", "data": obs}
        self.sock.send(self.msgpack.packb(request, default=self._encode))
        try:
            resp = self.msgpack.unpackb(self.sock.recv(), object_hook=self._decode)
        except zmq.Again:
            raise RuntimeError("inference server timeout on get_action") from None
        if "error" in resp:
            raise RuntimeError(f"Server error: {resp['error']}")
        return resp


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RoboCasa365 GR00T eval client")
    p.add_argument("--port", type=int, default=5570)
    p.add_argument("--task-set", default="atomic_seen",
                   choices=["atomic_seen", "composite_seen", "composite_unseen", "custom"])
    p.add_argument("--tasks", default=None, help="comma list overriding the task set")
    p.add_argument("--split", default="target")
    p.add_argument("--n-trials", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from gr00t.eval.sim.robocasa365.gymnasium_groot import GrootRoboCasa365Env

    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    else:
        tasks = list(TARGET_TASKS[args.task_set])
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
            env = GrootRoboCasa365Env(env_name=task, enable_render=True,
                                      split=args.split)
            fcntl.flock(_lf, fcntl.LOCK_UN)
        for trial in range(args.n_trials):
            seed = args.seed * 1000 + ti * 10 + trial
            obs, _ = env.reset(seed=seed)
            done = False
            steps = 0
            success = False
            while not done and steps < args.max_steps:
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
                action_chunk = client.get_action(send_obs)
                action = {
                    f"action.{k}": np.atleast_1d(action_chunk[f"action.{k}"][0])
                    for k in ("gripper_close", "end_effector_position",
                              "end_effector_rotation", "base_motion", "control_mode")
                }
                obs, reward, done, truncated, info = env.step(action)
                steps += 1
                success = bool(info.get("success", False))
                if success:
                    done = True
            results.append({
                "task": task, "trial": trial, "seed": seed,
                "success": success, "steps": steps,
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
