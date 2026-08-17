#!/usr/bin/env python3
"""Crash-tolerant per-trial driver for RoboCasa365 criterion-4 clients.

Each (task, trial) runs in its OWN python subprocess with a freshly
constructed env — the segfault seen in v2 (env.reset() on trial>0) is thus
bypassed entirely. A crashed child loses only that trial; the driver moves
on. Resume: (task, environment-seed) pairs already present in --out are skipped, so a
driver can be relaunched after any client crash without duplicating work.

Usage:
  python3 scripts/tools/run_crit4_trial_driver.py \
      --port 5571 --tasks PickPlaceDrawerToCounter,CoffeeSetupMug \
      --n-trials 3 --max-steps 720 \
      --out runs/robocasa365_eval/v2/crit4_w6_h2.jsonl
"""
import argparse
import hashlib
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
        for line in out.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("success") is not None:
                done.add((r.get("task"), r.get("seed")))
    return done


def prune_crash_markers(out: Path):
    """Remove retryable crash markers before resume, preserving valid rows."""
    if not out.exists():
        return
    kept = []
    removed = 0
    for line in out.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            kept.append(line)
            continue
        if row.get("success") is None or row.get("crashed"):
            removed += 1
        else:
            kept.append(line)
    if removed:
        out.write_text(("\n".join(kept) + "\n") if kept else "")
        print(f"[drv] pruned {removed} retryable crash marker(s) from {out}")


def build_jobs(tasks, n_trials, trial_seeds, base_seed):
    if trial_seeds:
        seeds = [int(v.strip()) for v in trial_seeds.split(",") if v.strip()]
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("--trial-seeds must contain unique integers")
        if n_trials != len(seeds):
            raise ValueError(
                f"--n-trials={n_trials} must equal len(--trial-seeds)={len(seeds)}"
            )
        return [
            (task, trial, seed)
            for task in tasks
            for trial, seed in enumerate(seeds)
        ]
    return [
        (task, trial, base_seed * 1000 + ti * 10 + trial)
        for ti, task in enumerate(tasks)
        for trial in range(n_trials)
    ]


def run_batch(
    port, task, jobs, max_steps, attempts, paired_action_noise,
    trial_timeout, egl_device,
):
    """Run several seeds in one imported client, retaining partial progress.

    Each seed still gets a freshly constructed environment.  If native code
    crashes the child, rows flushed before the crash are returned and only the
    remaining seeds are retried.
    """
    seed_to_trial = {seed: trial for _, trial, seed in jobs}
    pending = list(seed_to_trial)
    completed = {}
    child_env = os.environ.copy()
    # Keep the driver self-contained when launched from nohup/setsid instead
    # of an interactive experiment shell. RoboCasa365's pyzmq wheel lives in
    # the repo-local user base; EGL placement is explicit per worker.
    child_env.setdefault("PYTHONUSERBASE", os.path.join(REPO, ".pyuserbase"))
    child_env["MUJOCO_EGL_DEVICE_ID"] = str(egl_device)
    child_env.setdefault("NUMBA_CACHE_DIR", os.path.join(REPO, "runs", "numba_cache"))
    for att in range(attempts):
        if not pending:
            break
        seed_arg = ",".join(str(seed) for seed in pending)
        suffix = hashlib.sha256(seed_arg.encode()).hexdigest()[:12]
        tmp = Path(f"/tmp/crit4_drv_{port}_{task}_{suffix}.jsonl")
        if tmp.exists():
            tmp.unlink()
        cmd = [PY, os.path.join(REPO, CLIENT_PY),
               "--port", str(port), "--task-set", "custom", "--tasks", task,
               "--n-trials", str(len(pending)), "--exact-seeds", seed_arg,
               "--fresh-env-per-trial", "--max-steps", str(max_steps),
               "--out", str(tmp)]
        if paired_action_noise:
            cmd.append("--paired-action-noise")
        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd, cwd=REPO, env=child_env,
                timeout=trial_timeout * len(pending),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                universal_newlines=True,
            )
        except subprocess.TimeoutExpired:
            proc = None
            print(f"[drv] {task} seeds={pending} attempt {att+1}: TIMEOUT")
        elapsed = time.time() - t0
        new_results = []
        if tmp.exists():
            for line in tmp.read_text().splitlines():
                try:
                    result = json.loads(line)
                except json.JSONDecodeError:
                    continue
                seed = result.get("seed")
                if seed not in pending or result.get("success") is None:
                    continue
                result["trial"] = seed_to_trial[seed]
                result["seed"] = seed
                new_results.append(result)
        amortized = elapsed / len(new_results) if new_results else None
        for result in new_results:
            result["driver_wall_seconds"] = amortized
            completed[result["seed"]] = result
            print(
                f"[drv] {task} t{result['trial']} seed={result['seed']} "
                f"attempt {att+1}: OK ({amortized:.0f}s amortized)"
            )
        pending = [seed for seed in pending if seed not in completed]
        if proc is not None and proc.returncode != 0:
            print(
                f"[drv] {task} attempt {att+1}: child rc={proc.returncode}; "
                f"remaining={pending}"
            )
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
            for value in tail:
                print(f"[drv]   | {value}")
    if pending:
        print(f"[drv] {task}: all {attempts} attempts failed for seeds={pending}")
    return [completed[seed] for seed in seed_to_trial if seed in completed], pending


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--config", default="unknown",
                    help="Stable configuration id written into every result row.")
    ap.add_argument("--manifest-sha256", default=None)
    ap.add_argument("--config-sha256", default=None)
    ap.add_argument("--n-trials", type=int, default=3)
    ap.add_argument(
        "--trial-seeds",
        default=None,
        help=("Explicit comma-separated environment seeds applied to every task, "
              "for example 0,1,2,3,4. Independent of task order/sharding."),
    )
    ap.add_argument(
        "--max-steps", type=int, default=0,
        help="Episode horizon override; 0 uses RoboCasa's official task horizon.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--paired-action-noise", action="store_true")
    ap.add_argument("--trial-timeout", type=int, default=3600)
    ap.add_argument("--egl-device", type=int, default=3)
    ap.add_argument(
        "--trial-batch-size", type=int, default=1,
        help=("Trials per RoboCasa child process. Every trial still constructs "
              "a fresh environment; values >1 amortize Python import cost."),
    )
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        a = build_jobs(["A", "B"], 3, "0,1,2", 9)
        b = build_jobs(["B", "A"], 3, "0,1,2", 9)
        assert {(t, s) for t, _, s in a} == {(t, s) for t, _, s in b}
        assert [s for t, _, s in a if t == "A"] == [0, 1, 2]
        assert args.trial_batch_size >= 1
        print("[drv] selftest OK (explicit seeds independent of task order/sharding)")
        return
    if not args.out:
        raise SystemExit("--out is required unless --selftest is used")

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prune_crash_markers(out)
    done = existing(out)

    try:
        jobs = build_jobs(tasks, args.n_trials, args.trial_seeds, args.seed)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    todo = [job for job in jobs if (job[0], job[2]) not in done]
    print(f"[drv] port={args.port} tasks={tasks} "
          f"remaining={len(todo)}/{len(jobs)} paired_noise={args.paired_action_noise}")

    if args.trial_batch_size < 1:
        raise SystemExit("--trial-batch-size must be >= 1")
    grouped = []
    for task in tasks:
        task_jobs = [job for job in todo if job[0] == task]
        grouped.extend(
            (task, task_jobs[start:start + args.trial_batch_size])
            for start in range(0, len(task_jobs), args.trial_batch_size)
        )

    with open(out, "a", encoding="utf-8") as f:
        for task, batch in grouped:
            results, missing = run_batch(
                args.port, task, batch, args.max_steps,
                args.attempts, args.paired_action_noise,
                args.trial_timeout, args.egl_device,
            )
            for result in results:
                result["config"] = args.config
                result["manifest_sha256"] = args.manifest_sha256
                result["config_sha256"] = args.config_sha256
                f.write(json.dumps(result) + "\n")
                f.flush()
            trial_by_seed = {seed: trial for _, trial, seed in batch}
            for seed in missing:
                f.write(json.dumps({
                    "config": args.config,
                    "manifest_sha256": args.manifest_sha256,
                    "config_sha256": args.config_sha256,
                    "task": task, "trial": trial_by_seed[seed], "seed": seed,
                    "success": None, "steps": None, "crashed": True,
                }) + "\n")
            f.flush()
    print(f"[drv] done -> {out}")


if __name__ == "__main__":
    main()
