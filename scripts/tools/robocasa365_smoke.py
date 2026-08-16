#!/usr/bin/env python3
"""RoboCasa365 GR00T N1.5 checkpoint compatibility smoke (v1.4, Stage D gate).

Loads a target_posttraining checkpoint with the RoboCasa365 data config,
feeds synthetic observations (3 cameras + 5 state groups drawn from the
checkpoint's own statistics), and reports whether the 32-dim action chunk is
finite. Pass = the checkpoint loads with the local GR00T fork and produces
valid actions — the gate that unlocks the full RoboCasa365 eval harness.

Usage (groot_test env, one idle GPU):
    python scripts/tools/robocasa365_smoke.py \
        --ckpt checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/target_posttraining/atomic_seen/checkpoint-60000 \
        --device cuda:5 --n-obs 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "tools"))

DATA_CONFIG = "examples.RoboCasa365.custom_data_config:RoboCasa365DataConfig"
STATE_KEYS = [
    "state.base_position",
    "state.base_rotation",
    "state.end_effector_position_relative",
    "state.end_effector_rotation_relative",
    "state.gripper_qpos",
]


def load_stats(ckpt: Path) -> dict:
    md = json.loads((ckpt / "experiment_cfg" / "metadata.json").read_text())
    return md["new_embodiment"]["statistics"]


def make_obs(stats: dict, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    obs: dict = {}
    for cam in ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"):
        img = rng.integers(0, 255, size=(1, 256, 256, 3), dtype=np.uint8)
        obs[f"video.{cam}"] = img
    for key in STATE_KEYS:
        short = key.split(".", 1)[1]
        s = stats["state"][short]
        lo = np.asarray(s.get("min", -1.0), dtype=np.float32)
        hi = np.asarray(s.get("max", 1.0), dtype=np.float32)
        val = (lo + hi) / 2.0 + rng.standard_normal(lo.shape).astype(np.float32) * (hi - lo) / 6.0
        obs[key] = np.expand_dims(val, axis=0)
    obs["annotation.human.action.task_description"] = ["add ice cubes to the blender"]
    return obs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--denoising-steps", type=int, default=8)
    ap.add_argument("--n-obs", type=int, default=2)
    args = ap.parse_args()

    from gr00t_v2_common import ensure_flash_attn_rpath, load_policy, stack_obs

    ensure_flash_attn_rpath()
    ckpt = Path(args.ckpt)
    stats = load_stats(ckpt)
    policy = load_policy(
        str(ckpt), data_config=DATA_CONFIG, denoising_steps=args.denoising_steps,
        device=args.device, embodiment_tag="new_embodiment",
    )
    obs_list = [make_obs(stats, seed=i) for i in range(args.n_obs)]
    batched = stack_obs(obs_list)
    out = policy.get_action(batched)
    actions = {k: np.asarray(v) for k, v in out.items() if k.startswith("action.")}
    print("[robocasa365-smoke] action keys:", sorted(actions))
    ok = True
    for k, v in actions.items():
        finite = bool(np.isfinite(v).all())
        print(f"  {k}: shape={v.shape} finite={finite} range=[{v.min():.3f}, {v.max():.3f}]")
        ok = ok and finite
    print("[robocasa365-smoke] RESULT:", "PASS" if ok and actions else "FAIL")
    if not (ok and actions):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
