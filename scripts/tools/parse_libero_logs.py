#!/usr/bin/env python3
"""Parse dev-accept LIBERO logs into per-config success-rate tables.

The orchestrator streams every config's eval into one log per suite
(runs/v2_gpu_logs/liberos_<suite>.log); configs are delimited by the
"--- starting server" lines. For each config, the LAST "Current total success
rate" line before the next config (or EOF) is the config's final success rate
over its 50 rollouts.

Usage:
    python scripts/tools/parse_libero_logs.py [--log <path> ...] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

START_RE = re.compile(r"--- starting server: libero_(\w+) plan=(\S+)")
SUCCESS_RE = re.compile(r"Current total success rate: ([\d.]+)")
TASK_RE = re.compile(r"Current task success rate: ([\d.]+)")
EPISODE_RE = re.compile(r"# episodes completed so far: (\d+)")


def parse_log(path: Path) -> Dict[str, Any]:
    configs: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = START_RE.search(line)
            if m:
                if current is not None:
                    configs.append(current)
                current = {"suite": m.group(1), "plan": m.group(2),
                           "success_rate": None, "episodes": None, "tasks_done": 0}
                continue
            if current is None:
                continue
            s = SUCCESS_RE.search(line)
            if s:
                current["success_rate"] = float(s.group(1))
            e = EPISODE_RE.search(line)
            if e:
                current["episodes"] = int(e.group(1))
            t = TASK_RE.search(line)
            if t:
                current["tasks_done"] += 1
    if current is not None:
        configs.append(current)
    # dedupe consecutive identical plans (server restarts) by keeping the LAST
    out: List[Dict[str, Any]] = []
    for c in configs:
        if out and out[-1]["plan"] == c["plan"] and out[-1]["suite"] == c["suite"]:
            out[-1].update({k: v for k, v in c.items() if v is not None})
        else:
            out.append(c)
    # completeness: a config only enters the formal table when it finished
    # 50 episodes across 10 tasks (review round 5, item 11)
    for c in out:
        c["complete"] = (c.get("episodes") == 50 and c.get("tasks_done") == 10)
    return {"file": str(path), "configs": out}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log", nargs="*", default=None, help="log paths (default: runs/v2_gpu_logs/liberos_*.log)")
    p.add_argument("--json", default=None)
    args = p.parse_args()

    logs = [Path(x) for x in args.log] if args.log else sorted(
        Path("runs/v2_gpu_logs").glob("liberos_*.log")
    )
    all_results = [parse_log(l) for l in logs]
    for r in all_results:
        print(f"=== {r['file']} ===")
        for c in r["configs"]:
            mark = "OK " if c.get("complete") else "INC"
        print(f"  [{mark}] {c['plan']:60s} SR={c['success_rate']} episodes={c['episodes']} tasks={c['tasks_done']}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"saved -> {args.json}")


if __name__ == "__main__":
    main()
