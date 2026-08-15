"""GR1 tabletop env smoke test: build one env, reset, 10 zero-action steps.

Usage:
    <groot_test python> scripts/tools/gr1_env_smoke.py [task_name]
"""
import sys

import numpy as np

task = sys.argv[1] if len(sys.argv) > 1 else "PnPCupToDrawerClose"
env_name = "gr1_unified/{}_GR1ArmsAndWaistFourierHands_Env".format(task)

# Must import GrootRoboCasaEnv BEFORE gym.make to trigger namespace registration.
from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401

import gymnasium as gym

print("making env:", env_name)
env = gym.make(env_name)
print("env created OK; action_dim =", getattr(env, "action_dim", "?"))

obs = env.reset()
print("reset OK")
for i in range(10):
    obs, rew, done, info = env.step(np.zeros(env.action_dim))
    if done:
        print("episode terminated at step", i)
        obs = env.reset()
print("GR1-SMOKE-PASS")
