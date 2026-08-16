#!/usr/bin/env python3
"""Crash-tolerant per-trial driver for RoboCasa365 criterion-4 clients.

Each (task, trial) runs in its OWN python subprocess with a freshly
constructed env — the segfault seen in v2 (env.reset() on trial>0) is thus
bypassed entirely. A crashed child loses only that trial; the driver moves
on. Resume: (task, trial) pairs already present in --out are skipped, so a
driver can be relaunched after any client crash without duplicating work.

Usage:
  python3 scripts/tools/run_crit4_trial_driver.py \
      --port 5571 --tasks PickPlaceDrawerToCounter,CoffeeSetupMug \
      --n-trials 3 --max-steps 720 \
      --out runs/robocasa365_eval/v2/crit4_w6_h2.jsonl
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = "/home1/gyy/vla/QuantVLA"
CLIENT_PY = "scripts/run_robocasa365_gr00t_eval.py"
PY = "/home1/gyy/probe/miniforge3/envs/robocasa365/bin/python"


def existing(out: Path):
    done = set()
    if out.exists():
        try:
            for line in out.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("success") is not None:
                    done.add((r.get("task"), r.get("trial")))
        except json.JSONDecodeError:
            pass
    return done


def run_one(port, task, trial, seed, max_steps, attempts):
    tmp = Path(f"/tmp/crit4_drv_{port}_{task}_{trial}.jsonl")
    cmd = [PY, os.path.join(REPO, CLIENT_PY),
           "--port", str(port), "--task-set", "custom", "--tasks", task,
           "--n-trials", "1", "--seed", str(seed),
           "--max-steps", str(max_steps), "--out", str(tmp)]
    for att in range(attempts):
        if tmp.exists():
            tmp.unlink()
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, cwd=REPO, timeout=1800,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  universal_newlines=True)
        except subprocess.TimeoutExpired:
            print(f"[drv] {task} t{trial} attempt {att+1}: TIMEOUT")
            continue
        if proc.returncode == 0 and tmp.exists():
            line = tmp.read_text().strip().splitlines()
            if line:
                print(f"[drv] {task} t{trial} attempt {att+1}: OK "
                      f"({time.time()-t0:.0f}s) {line[-1]}")
                return line[-1]
        print(f"[drv] {task} t{trial} attempt {att+1}: FAILED rc={proc.returncode}")
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        for t in tail:
            print(f"[drv]   | {t}")
    print(f"[drv] {task} t{trial}: all {attempts} attempts failed -> marking crashed")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--n-trials", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=720)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--attempts", type=int, default=3)
    args = ap.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = existing(out)

    todo = [(ti, task, trial)
            for ti, task in enumerate(tasks)
            for trial in range(args.n_trials)
            if (task, trial) not in done]
    print(f"[drv] port={args.port} tasks={tasks} "
          f"remaining={len(todo)}/{len(tasks)*args.n_trials}")

    with open(out, "a", encoding="utf-8") as f:
        for ti, task, trial in todo:
            seed = args.seed * 1000 + ti * 10 + trial
            line = run_one(args.port, task, trial, seed, args.max_steps,
                           args.attempts)
            if line is None:
                f.write(json.dumps({
                    "task": task, "trial": trial, "seed": seed,
                    "success": None, "steps": None, "crashed": True,
                }) + "\n")
            else:
                f.write(line + "\n")
            f.flush()
    print(f"[drv] done -> {out}")


if __name__ == "__main__":
    main()
