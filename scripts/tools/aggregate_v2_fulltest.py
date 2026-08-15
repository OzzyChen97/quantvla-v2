#!/usr/bin/env python3
"""Aggregate the 8-shard dev-accept + held-out LIBERO logs into full-test tables.

Design (D-016 sharding + D-017/D-018 watchdog semantics):
  * spatial/goal/object: 2 shards x 5 configs (v2 final, uniform_w6, random
    best/median/worst) x 5 tasks x 5 eps = 50 rollouts per config per suite.
  * long (held-out libero_10): 2 shards x 3 seeds x 2 configs
    (transfer-v2, transfer-w6) x 5 tasks x 5 eps = 50 rollouts per config/seed.
  * A config only enters the FORMAL table when BOTH shards completed it
    (25 episodes / 5 tasks each); everything else is reported as PENDING.
  * Server restarts (watchdog retry) merge consecutive identical-plan blocks,
    keeping the LAST attempt (matching parse_libero_logs.py semantics).

Usage:
    python scripts/tools/aggregate_v2_fulltest.py [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[2]
LOGDIR = REPO / "runs" / "v2_gpu_logs"
PACKS = REPO / "checkpoints" / "packs" / "gr00t"

START_RE = re.compile(r"--- starting server: libero_(\w+) plan=(\S+)")
HELD_RE = re.compile(r"--- held-out: libero_(\w+) seed=(\d+) plan=(\S+)")
TASK_NAME_RE = re.compile(r"^Task: (.*)$")
TASK_SR_RE = re.compile(r"Current task success rate: ([\d.]+)")
TOTAL_SR_RE = re.compile(r"Current total success rate: ([\d.]+)")
EPS_RE = re.compile(r"# episodes completed so far: (\d+)")

EXP_EPS = 25   # per shard per config (5 tasks x 5 eps)
EXP_TASKS = 5  # per shard per config

DEV_SUITES = ("spatial", "goal", "object")
SHARD_FILES = {
    "spatial": ["liberos_spatial_s0.log", "liberos_spatial_s1.log"],
    "goal": ["liberos_goal_s0.log", "liberos_goal_s1.log"],
    "object": ["liberos_object_s0.log", "liberos_object_s1.log"],
    "long": ["liberos_long_s0.log", "liberos_long_s1.log"],
}


def split_blocks(text: str) -> List[Tuple[Optional[str], int, Dict[str, Any]]]:
    """Split a log into per-config blocks (plan basename, seed, block dict).

    Block dict keys: tasks ([(name, sr), ...]), success_rate, episodes,
    tasks_done, plan_path. seed=None for dev-accept configs.
    """
    blocks: List[Tuple[Optional[str], int, Dict[str, Any]]] = []
    cur: Optional[Dict[str, Any]] = None
    cur_plan: Optional[str] = None
    cur_seed: int = 0
    held_seed: Dict[str, int] = {}
    for line in text.splitlines():
        h = HELD_RE.search(line)
        if h:
            held_seed[h.group(3)] = int(h.group(2))
        m = START_RE.search(line)
        if m:
            if cur is not None:
                blocks.append((cur_plan, cur_seed, cur))
            cur = {"tasks": [], "success_rate": None, "episodes": None,
                   "tasks_done": 0, "plan_path": m.group(2)}
            cur_plan = Path(m.group(2)).name
            cur_seed = held_seed.get(cur_plan, 0)
            continue
        if cur is None:
            continue
        t = TASK_NAME_RE.search(line)
        if t:
            cur["tasks"].append([t.group(1), None])
        s = TASK_SR_RE.search(line)
        if s and cur["tasks"]:
            cur["tasks"][-1][1] = float(s.group(1))
        s = TOTAL_SR_RE.search(line)
        if s:
            cur["success_rate"] = float(s.group(1))
        e = EPS_RE.search(line)
        if e:
            cur["episodes"] = int(e.group(1))
        if TASK_SR_RE.search(line):
            cur["tasks_done"] += 1
    if cur is not None:
        blocks.append((cur_plan, cur_seed, cur))
    # dedupe consecutive identical plans (server restarts): keep the LAST block
    out: List[Tuple[Optional[str], int, Dict[str, Any]]] = []
    for plan, seed, b in blocks:
        if out and out[-1][0] == plan:
            out[-1] = (plan, seed, b)
        else:
            out.append((plan, seed, b))
    return out


def load_random_reps() -> Dict[str, Dict[str, str]]:
    """suite -> {best/median/worst: plan basename} over the 20 RANDOM masks only
    (uniform_w6/w4 and manual entries excluded; matches dev_accept D-019 fix)."""
    reps: Dict[str, Dict[str, str]] = {}
    for suite in DEV_SUITES:
        p = PACKS / f"baselines_{suite}_dsolver.json"
        if not p.exists():
            continue
        scored = [e for e in json.loads(p.read_text())["scored"]
                  if Path(e["file"]).name.startswith("random_")]
        scored.sort(key=lambda e: e["d_solver"])
        reps[suite] = {"best": scored[0]["file"],
                       "median": scored[len(scored) // 2]["file"],
                       "worst": scored[-1]["file"]}
    return reps


def analyze_suite(prefix: str, kind: str) -> Dict[str, Any]:
    """Merge the two shard logs of one suite/held-out group."""
    merged: List[Dict[str, Any]] = []  # one entry per logical config
    per_shard: Dict[str, Dict[str, Any]] = {}
    for fname in SHARD_FILES[prefix]:
        p = LOGDIR / fname
        text = p.read_text(errors="replace") if p.exists() else ""
        blocks = split_blocks(text)
        for plan, seed, b in blocks:
            key = plan
            if kind == "long":
                key = f"seed{seed}:{plan}"
            per_shard.setdefault(key, {})
            per_shard[key][fname] = b
    if kind == "dev":
        order = ["gr00t_quant_plan_libero_%s_adjudicated.final_plan.json" % prefix,
                 "uniform_w6.json"]
        reps = load_random_reps().get(prefix, {})
        order += [reps.get(k, "") for k in ("best", "median", "worst")]
        keys = [k for k in order if k]  # present plans in fixed order
    else:
        keys = []
        for seed in range(3):
            for plan in ("gr00t_quant_plan_long_transfer_v2.json",
                         "gr00t_quant_plan_long_transfer_w6.json"):
                keys.append(f"seed{seed}:{plan}")
    for key in keys:
        shards = per_shard.get(key, {})
        entry = {"key": key, "complete": False, "success_rate": None,
                 "episodes": 0, "tasks_done": 0, "tasks": {},
                 "shards_present": sorted(shards)}
        srs, eps, td = [], [], []
        for fname, b in shards.items():
            srs.append(b["success_rate"]); eps.append(b["episodes"] or 0)
            td.append(b["tasks_done"])
            for name, sr in b["tasks"]:
                entry["tasks"].setdefault(name, []).append(sr)
        if len(shards) == 2 and all(
                (b["episodes"] == EXP_EPS and b["tasks_done"] == EXP_TASKS)
                for b in shards.values()):
            entry["complete"] = True
            entry["success_rate"] = round(
                sum(s * e for s, e in zip(srs, eps)) / sum(eps), 4)
        entry["episodes"] = sum(eps)
        entry["tasks_done"] = sum(td)
        merged.append(entry)
    return {"prefix": prefix, "kind": kind, "configs": merged}


def fmt_sr(v: Optional[float]) -> str:
    return "—" if v is None else f"{v:.2%}".replace("%", "")


def task_table(entry: Dict[str, Any]) -> str:
    rows = []
    for name, srs in sorted(entry["tasks"].items()):
        vals = [x for x in srs if x is not None]
        v = f"{sum(vals)/len(vals):.2f}" if vals else "—"
        rows.append(f"| {name[:44]:44s} | {v} |")
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    dev = {s: analyze_suite(s, "dev") for s in DEV_SUITES}
    long = analyze_suite("long", "long")

    lines: List[str] = []
    lines.append("# LIBERO full-test aggregation")
    lines.append("")
    for suite in DEV_SUITES:
        a = dev[suite]
        lines.append(f"## {suite} (dev-accept, 2 shards x 25 eps per config)")
        for c in a["configs"]:
            status = "OK " if c["complete"] else "PENDING"
            lines.append(f"- [{status}] {c['key']}: SR={fmt_sr(c['success_rate'])} "
                         f"eps={c['episodes']}/50 tasks={c['tasks_done']}/10 "
                         f"shards={','.join(c['shards_present']) or 'none'}")
            if c["complete"]:
                lines.append(task_table(c))
        lines.append("")
    lines.append("## long (held-out libero_10, 3 seeds x {transfer-v2, transfer-w6})")
    for c in long["configs"]:
        status = "OK " if c["complete"] else "PENDING"
        lines.append(f"- [{status}] {c['key']}: SR={fmt_sr(c['success_rate'])} "
                     f"eps={c['episodes']}/50 tasks={c['tasks_done']}/10 "
                     f"shards={','.join(c['shards_present']) or 'none'}")
        if c["complete"]:
            lines.append(task_table(c))
    # long per-seed summary
    lines.append("")
    lines.append("| long config | seed0 | seed1 | seed2 | mean |")
    lines.append("|---|---|---|---|---|")
    for plan in ("transfer_v2", "transfer_w6"):
        srs = []
        for c in long["configs"]:
            if plan in c["key"]:
                srs.append(fmt_sr(c["success_rate"]) if c["complete"] else "?")
        vals = [c["success_rate"] for c in long["configs"]
                if plan in c["key"] and c["complete"]]
        mean = f"{sum(vals)/len(vals):.2f}" if vals else "?"
        lines.append(f"| {plan} | {' | '.join(srs)} | {mean} |")

    out = "\n".join(lines) + "\n"
    print(out)
    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"dev": dev, "long": long}, indent=2))
        print(f"saved -> {p}")


if __name__ == "__main__":
    main()
