#!/usr/bin/env python3
"""Collect real LIBERO observations for the CKA forensic audit (Audit 3).

Two modes (run in the libero_test conda env):

  --source env [--suite spatial|goal|object] [--tasks 0 1] [--steps N]
      Step the LIBERO env with the FP16 GR00T policy server (ZMQ, same client
      as run_libero_eval.py), collecting one GR00T-format obs dict per step
      (video.image, video.wrist_image, state.*, annotation). L2 = states the
      policy actually visits.

  --source dataset [--suite spatial] [--max-episodes N] [--max-steps-per-ep M]
      Read unlabeled demo observations from data/libero_<suite> (LeRobot
      parquet + image files), convert to the same GR00T-format dicts. L3.

Output: <out>.npz with per-key arrays (video.image: (N,1,H,W,C) uint8,
video.wrist_image likewise, state.x/y/z/roll/pitch/yaw/gripper: (N,1) float32,
annotation.human.action.task_description: (N,) object strings).

Usage:
    # L2 (needs a running FP16 server: scripts/run_inference_server.sh spatial)
    conda run -n libero_test python scripts/tools/collect_libero_states.py \
        --source env --suite spatial --port 5556 --steps 64 --out runs/libero_states_l2_spatial.npz
    # L3 (offline, no server)
    conda run -n libero_test python scripts/tools/collect_libero_states.py \
        --source dataset --suite spatial --out runs/libero_states_l3_spatial.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))


def _process_env_obs(obs: dict, lang: str) -> dict:
    """LIBERO env obs -> GR00T-format obs dict (mirrors run_libero_eval.py)."""
    from examples.Libero.eval.utils import get_libero_image, quat2axisangle

    xyz = obs["robot0_eef_pos"]
    rpy = quat2axisangle(obs["robot0_eef_quat"])
    gripper = obs["robot0_gripper_qpos"]
    img, wrist_img = get_libero_image(obs)
    return {
        "video.image": np.expand_dims(img, axis=0).astype(np.float32),
        "video.wrist_image": np.expand_dims(wrist_img, axis=0).astype(np.float32),
        "state.x": np.array([[xyz[0]]], dtype=np.float32),
        "state.y": np.array([[xyz[1]]], dtype=np.float32),
        "state.z": np.array([[xyz[2]]], dtype=np.float32),
        "state.roll": np.array([[rpy[0]]], dtype=np.float32),
        "state.pitch": np.array([[rpy[1]]], dtype=np.float32),
        "state.yaw": np.array([[rpy[2]]], dtype=np.float32),
        "state.gripper": np.expand_dims(np.asarray(gripper, dtype=np.float32), axis=0),
        "annotation.human.action.task_description": [lang],
    }


def collect_env(suite: str, port: int, tasks: list[int], steps: int, out: Path) -> None:
    import libero
    from libero.libero import benchmark

    from gr00t.eval.service import ExternalRobotInferenceClient

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[f"libero_{suite}"]()
    client = ExternalRobotInferenceClient(host="localhost", port=port)
    entries: list[dict] = []
    tasks = tasks or list(range(min(3, len(task_suite.tasks))))
    for t in tasks:
        task = task_suite.get_task(t)
        task_suite.set_task(t)
        env = task_suite.env
        env.reset()
        init_states = task_suite.get_task_init_states(t)
        env.set_init_state(init_states[0])
        env.seed(0)
        obs = env.reset()
        lang = task.language
        done = False
        for _ in range(steps):
            entries.append(_process_env_obs(obs, lang))
            a = client.get_action(entries[-1])
            # GR00T chunk -> libero 7-dim (mirrors run_libero_eval conversion)
            action = np.array([
                np.atleast_1d(a[f"action.{k}"][0])[0]
                for k in ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
            ], dtype=np.float32)
            obs, reward, done, info = env.step(action)
            if done:
                break
        print(f"[collect] task {t}: {len(entries)} obs total")
    save(entries, out)


def collect_dataset(suite: str, max_episodes: int, max_steps: int, out: Path) -> None:
    import pandas as pd
    from PIL import Image

    data_root = REPO_ROOT / "data" / f"libero_{suite}"
    chunk = data_root / "data" / "chunk-000"
    parquets = sorted(chunk.glob("episode_*.parquet"))[:max_episodes]
    entries: list[dict] = []
    for pq in parquets:
        df = pd.read_parquet(pq)
        lang = str(df.iloc[0].get("language_instruction", ""))
        for i, row in df.iloc[:: max(1, len(df) // max_steps)].iterrows():
            state = np.asarray(row["observation.state"], dtype=np.float32)
            img_path = str(row["observation.images.image"])
            wrist_path = str(row["observation.images.wrist_image"])
            img = np.asarray(Image.open(data_root / img_path), dtype=np.float32) / 255.0
            wrist = np.asarray(Image.open(data_root / wrist_path), dtype=np.float32) / 255.0
            entries.append({
                "video.image": np.expand_dims(img, axis=0),
                "video.wrist_image": np.expand_dims(wrist, axis=0),
                "state.x": np.array([[state[0]]], dtype=np.float32),
                "state.y": np.array([[state[1]]], dtype=np.float32),
                "state.z": np.array([[state[2]]], dtype=np.float32),
                "state.roll": np.array([[state[3]]], dtype=np.float32),
                "state.pitch": np.array([[state[4]]], dtype=np.float32),
                "state.yaw": np.array([[state[5]]], dtype=np.float32),
                "state.gripper": np.array([[state[6]]], dtype=np.float32),
                "annotation.human.action.task_description": [lang],
            })
        print(f"[collect] {pq.name}: {len(entries)} obs total")
        if len(entries) >= max_episodes * max_steps:
            break
    save(entries, out)


def save(entries: list[dict], out: Path) -> None:
    keys = sorted(entries[0].keys())
    arrays = {}
    for k in keys:
        vals = [e[k] for e in entries]
        if isinstance(vals[0], np.ndarray):
            arrays[k] = np.stack(vals, axis=0)
        else:
            arrays[k] = np.array(vals, dtype=object)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    print(f"[collect] saved {len(entries)} obs -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["env", "dataset"], required=True)
    ap.add_argument("--suite", default="spatial", choices=["spatial", "goal", "object", "10", "90"])
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--tasks", nargs="*", type=int, default=[])
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--max-episodes", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if args.source == "env":
        collect_env(args.suite, args.port, args.tasks, args.steps, out)
    else:
        collect_dataset(args.suite, args.max_episodes, args.max_steps, out)


if __name__ == "__main__":
    main()
