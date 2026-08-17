#!/usr/bin/env python3
"""Freeze the CS+CKA ratio from proxy reports and the preregistered dev set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


RATIOS = [8, 16, 32, 64]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--reports-dir", required=True,
                   help="Directory containing gr00t_quant_plan_..._<ratio>to1_adjudicated.report.json")
    p.add_argument("--dev-summary", default=None)
    p.add_argument("--out", required=True)
    return p.parse_args()


def find_report(root: Path, ratio: int) -> Path:
    matches = sorted(root.glob(f"*cscka_{ratio}to1_adjudicated.report.json"))
    if len(matches) != 1:
        raise SystemExit(f"ratio {ratio}: expected one report under {root}, got {matches}")
    return matches[0]


def main() -> None:
    args = parse_args()
    root = Path(args.reports_dir)
    proxy = []
    for ratio in RATIOS:
        path = find_report(root, ratio)
        report = json.loads(path.read_text())
        proxy.append({
            "ratio": ratio,
            "d_func": float(report["final"]["d_func"]),
            "d_solver": float(report["final"]["d_solver"]),
            "plan": str(Path(report["final_plan_path"]).resolve()),
            "report": str(path.resolve()),
            "source": report["meta"]["final_source"],
        })
    proxy.sort(key=lambda row: (row["d_func"], -row["ratio"]))
    decision = {"proxy_ranking": proxy, "dev_candidates": proxy[:2]}

    if args.dev_summary:
        summary = json.loads(Path(args.dev_summary).read_text())
        dev_rows = []
        for row in proxy[:2]:
            cid = f"cscka_{row['ratio']}to1"
            if cid not in summary["configs"]:
                raise SystemExit(f"dev summary lacks {cid}")
            dev_rows.append({
                **row,
                "dev_task_macro_sr": float(summary["configs"][cid]["all18_task_macro_sr"]),
            })
        # Preregistered tie break: dev macro SR, then D_func, then more CKA.
        dev_rows.sort(key=lambda row: (
            -row["dev_task_macro_sr"], row["d_func"], -row["ratio"]
        ))
        decision["dev_ranking"] = dev_rows
        decision["selected"] = dev_rows[0]
        decision["selection_rule"] = (
            "max dev task-macro SR; tie -> min D_func; tie -> larger CKA:CS ratio"
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
