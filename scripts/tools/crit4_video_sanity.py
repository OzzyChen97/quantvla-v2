#!/usr/bin/env python3
"""End-to-end sanity: real-video obs through a live server, 60 steps.

Checks that policy actions differ from the blind-run prior-mean behavior
(D-040: blind runs produced near-prior outputs, eef ~ -0.93 / gripper ~ -0.36).
"""
import sys
import time

import numpy as np

sys.path.insert(0, "/home1/gyy/vla/QuantVLA/scripts")
from run_robocasa365_gr00t_eval import (
    SEND_LANG_KEYS,
    SEND_STATE_KEYS,
    SEND_VIDEO_KEYS,
    _Gr00tZMQClient,
)

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 5567
TASK = sys.argv[2] if len(sys.argv) > 2 else "OpenCabinet"
N = int(sys.argv[3]) if len(sys.argv) > 3 else 60

import robocasa  # noqa: E402 — must precede code path insert (D-023)
sys.path.insert(0, "/home1/gyy/vla/QuantVLA/code")
from gr00t.eval.sim.robocasa365.gymnasium_groot import GrootRoboCasa365Env  # noqa: E402

env = GrootRoboCasa365Env(env_name=TASK, enable_render=True, split="target")
client = _Gr00tZMQClient(host="localhost", port=PORT)
obs, _ = env.reset(seed=12345)

acts = []
t0 = time.time()
for i in range(N):
    send_obs = {}
    for k in SEND_VIDEO_KEYS + SEND_STATE_KEYS + SEND_LANG_KEYS:
        if k not in obs:
            continue
        v = obs[k]
        if k.startswith("video."):
            v = np.asarray(v)
            if v.ndim == 3:
                v = v[np.newaxis, ...]
        elif k.startswith("state."):
            v = np.asarray(v, dtype=np.float32)
            if v.ndim == 1:
                v = v[np.newaxis, :]
        elif isinstance(v, str):
            v = [v]
        send_obs[k] = v
    chunk = client.get_action(send_obs)
    action = {
        f"action.{k}": np.atleast_1d(chunk[f"action.{k}"][0])
        for k in ("gripper_close", "end_effector_position",
                  "end_effector_rotation", "base_motion", "control_mode")
    }
    acts.append({
        "eef_pos": float(np.asarray(action["action.end_effector_position"]).mean()),
        "eef_pos_std": float(np.asarray(action["action.end_effector_position"]).std()),
        "gripper": float(np.asarray(action["action.gripper_close"]).mean()),
        "base": float(np.asarray(action["action.base_motion"]).mean()),
        "ctrl": float(np.asarray(action["action.control_mode"]).mean()),
    })
    obs, *_ = env.step(action)
    if i in (0, 10, 20, 30, 40, 50, 59):
        a = acts[-1]
        print(f"step {i:3d}: eef_pos_mean={a['eef_pos']:+.3f} "
              f"(std={a['eef_pos_std']:.3f}) gripper={a['gripper']:+.3f} "
              f"base={a['base']:+.3f} ctrl={a['ctrl']:+.3f}", flush=True)

eef = np.array([a["eef_pos"] for a in acts])
grip = np.array([a["gripper"] for a in acts])
print(f"[sanity] {N} steps in {time.time()-t0:.0f}s; eef_pos mean {eef.mean():+.3f} "
      f"std {eef.std():.3f}; gripper mean {grip.mean():+.3f} std {grip.std():.3f}")
print("[sanity] verdict: ", end="")
if abs(eef.mean() + 0.93) < 0.1 and eef.std() < 0.02 and abs(grip.mean() + 0.36) < 0.1:
    print("STILL PRIOR-LIKE (policy not reacting to video!)")
else:
    print("NOT prior-like (video conditioning active)")
env.close()
