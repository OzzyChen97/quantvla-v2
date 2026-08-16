import os
import sys
import time

import robocasa  # noqa: F401  — must precede sys.path changes (D-023)
sys.path.insert(0, "/home1/gyy/vla/QuantVLA/code")
os.chdir("/home1/gyy/vla/QuantVLA")
t0 = time.time()
print("[egl3-test] importing...", flush=True)
import numpy as np
from gr00t.eval.sim.robocasa365.gymnasium_groot import GrootRoboCasa365Env
print(f"[egl3-test] import done {time.time()-t0:.0f}s", flush=True)
env = GrootRoboCasa365Env(env_name="PickPlaceDrawerToCounter",
                          enable_render=False, split="target")
print(f"[egl3-test] constructed in {time.time()-t0:.0f}s", flush=True)
obs, _ = env.reset(seed=7)
for i in range(50):
    a = {
        "action.gripper_close": np.float32(0.0),
        "action.end_effector_position": np.zeros(3, dtype=np.float32),
        "action.end_effector_rotation": np.zeros(3, dtype=np.float32),
        "action.base_motion": np.zeros(4, dtype=np.float32),
        "action.control_mode": np.float32(0.0),
    }
    obs, *_ = env.step(a)
print(f"[egl3-test] 50 steps OK, obs keys={len(obs)}", flush=True)
env.close()
print("[egl3-test] PASS", flush=True)
