#!/usr/bin/env python3
"""GR00T v2 consensus plan (review round 3 item 7; review round 4: adjudicated inputs).

The inputs MUST be the TopK-adjudicated final plans
(`*_adjudicated.final_plan.json`), NOT the pre-adjudication selector plans —
the method definition is proxy search -> config-level D_solver TopK
adjudication -> LIBERO, and the consensus step sits AFTER the adjudication.
Each input plan's meta must carry "adjudicated": true, otherwise this tool
refuses to run (the TopK result must not be silently bypassed).

The design doc §6.5 requires the three dev suites (spatial/goal/object) to
produce FP16 masks with pairwise Jaccard >= 0.7, and the final deployment to
use ONE frozen unified plan. This tool:

  1. reads the three dev-suite plan JSONs;
  2. computes pairwise FP16-mask Jaccard (hard gate: < 0.7 -> exit 1 with a
     report; the ranking is not reproducible enough to freeze);
  3. builds the consensus mask by majority vote (a layer is FP16 when >= 2 of
     the 3 plans skip it);
  4. repairs the budget (if the consensus exceeds it, quantize the layer with
     the smallest distortion-per-byte-saved first — same greedy semantics as
     the selector, driven by score/weight already stored in the plan layers);
  5. writes the frozen unified plan (schema identical to gr00t_select_plan.py
     output, plus "consensus" meta) that Long/90 use ZERO-SHOT (no re-probing
     on the Long checkpoint).

Usage:
    python scripts/tools/gr00t_consensus_plan.py \
        --plans <plan_spatial.json> <plan_goal.json> <plan_object.json> \
        --ckpt checkpoints/gr00t/libero-spatial \
        --budget uniform-w6 \
        --out checkpoints/packs/gr00t/gr00t_quant_plan_consensus.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "tools"))

from gr00t_select_plan import plan_total_bytes, read_layer_shapes  # noqa: E402

MIN_JACCARD = 0.7


def fp16_mask(plan: Dict[str, Any]) -> Tuple[str, ...]:
    return tuple(sorted(n for n, e in (plan.get("layers") or {}).items() if e.get("skip")))


def jaccard(a: Tuple[str, ...], b: Tuple[str, ...]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa or sb else 1.0


def build_consensus(
    plans: List[Dict[str, Any]],
    shapes: Dict[str, Dict[str, Any]],
    budget: float,
    row_rot: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Majority-vote skip set + budget repair; returns (plan, report)."""
    names = sorted(shapes)
    masks = [set(fp16_mask(p)) for p in plans]
    skip = {n for n in names if sum(1 for m in masks if n in m) >= 2}
    plan: Dict[str, Any] = {}
    for n in names:
        plan[n] = {"bits": 4 if n not in skip else None, "group": 64, "skip": n in skip}

    def bytes_of(p: Dict[str, Any]) -> float:
        return plan_total_bytes(p, shapes, row_rot)

    # budget repair: over budget -> quantize the skipped layer with the smallest
    # distortion-per-byte-saved (score/weight come from the dev plan entries)
    score_of = {}
    for p in plans:
        for n, e in (p.get("layers") or {}).items():
            w = e.get("weight") or 1.0
            s = e.get("score")
            if s is None:
                continue
            score_of.setdefault(n, []).append(w * float(s))
    while bytes_of(plan) > budget + 1e-3:
        candidates = []
        for n in names:
            if not plan[n]["skip"]:
                continue
            s = shapes[n]
            from gr00t_select_plan import layer_bytes_fp16, layer_bytes_quant

            saved = layer_bytes_fp16(s["out"], s["in"], s["has_bias"]) - layer_bytes_quant(
                s["out"], s["in"], s["has_bias"], 4, 64, row_rot=row_rot
            )
            if saved <= 0:
                continue
            avg_score = min(score_of.get(n, [1e9]))
            candidates.append((avg_score / saved, n))
        if not candidates:
            raise SystemExit("[consensus] budget repair impossible (no quantizable skipped layer)")
        _, n = min(candidates)
        plan[n] = {"bits": 4, "group": 64, "skip": False}

    report = {
        "n_plans": len(plans),
        "pairwise_jaccard": [
            {"pair": f"{i}-{j}", "jaccard": round(jaccard(fp16_mask(plans[i]), fp16_mask(plans[j])), 4)}
            for i in range(len(plans)) for j in range(i + 1, len(plans))
        ],
        "min_jaccard": round(min(jaccard(fp16_mask(a), fp16_mask(b))
                                   for i, a in enumerate(plans) for b in plans[i + 1:]), 4),
        "consensus_skip_count": len(skip),
        "repaired": bytes_of(plan) <= budget + 1e-3,
        "total_bytes": bytes_of(plan),
        "budget": budget,
    }
    return plan, report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GR00T v2 consensus plan (review round 3, item 7)")
    p.add_argument("--plans", nargs="+", default=None, help="2-3 dev-suite plan JSONs.")
    p.add_argument("--ckpt", default=None, help="Checkpoint for layer shapes (any dev suite).")
    p.add_argument("--budget", default="uniform-w6")
    p.add_argument("--min-jaccard", type=float, default=MIN_JACCARD)
    p.add_argument("--out", default=None)
    p.add_argument("--selftest", action="store_true")
    return p.parse_args()


def _selftest() -> None:
    import tempfile

    names = [f"L{i}" for i in range(20)]
    shapes = {n: {"out": 4096, "in": 4096, "has_bias": False} for n in names}
    fp_total = sum(4096 * 4096 * 2 for _ in names)

    def mk(skip_idx):
        return {"layers": {n: {"bits": 4 if n not in skip_idx else None, "group": 64,
                               "skip": n in skip_idx, "score": 0.01, "weight": 1.0}
                           for n in names}}

    # three highly-overlapping masks
    p1, p2, p3 = mk(set(names[:10])), mk(set(names[1:11])), mk(set(names[:10]))
    plan, rep = build_consensus([p1, p2, p3], shapes, fp_total * 0.7, "restore")
    assert rep["min_jaccard"] > 0.8, rep
    assert plan_total_bytes(plan, shapes, "restore") <= fp_total * 0.7 + 1e-3
    skip = {n for n, e in plan.items() if e["skip"]}
    assert "L0" in skip and "L11" not in skip  # majority vote
    # disjoint masks must fail the jaccard gate (checked by main, simulate here)
    p4 = mk(set(names[10:]))
    from gr00t_select_plan import plan_mask as _pm  # noqa
    j = jaccard(fp16_mask(p1), fp16_mask(p4))
    assert j < MIN_JACCARD, j
    print("[consensus] selftest OK (majority vote + budget repair + jaccard gate)")


def main() -> None:
    args = parse_args()
    if args.selftest:
        _selftest()
        return
    if not args.plans or not args.ckpt or not args.out:
        raise SystemExit("--plans/--ckpt/--out required (or --selftest)")
    plans = [json.loads(Path(x).read_text()) for x in args.plans]
    if len(plans) < 2 or len(plans) > 3:
        raise SystemExit("--plans expects 2-3 dev-suite plans")
    # review round 4: only TopK-adjudicated final plans are valid inputs —
    # the proxy-only selector plan is the INNER loop, never the frozen artifact.
    for x, pl in zip(args.plans, plans):
        if not (pl.get("meta") or {}).get("adjudicated"):
            raise SystemExit(
                f"{x} is not a TopK-adjudicated final plan (meta.adjudicated missing) — "
                "run gr00t_topk_scorer.py first and pass its .final_plan.json"
            )
    all_shapes = read_layer_shapes(Path(args.ckpt), r".*", r"^$")
    names = sorted(set().union(*(set(p.get("layers", {})) for p in plans)))
    shapes = {n: all_shapes[n] for n in names if n in all_shapes}
    missing = set(names) - set(shapes)
    if missing:
        raise SystemExit(f"{len(missing)} plan layers missing from checkpoint shapes")

    if args.budget == "uniform-w6":
        from gr00t_select_plan import layer_bytes_quant

        budget = sum(
            layer_bytes_quant(s["out"], s["in"], s["has_bias"], 6, 64, row_rot="restore")
            for s in shapes.values()
        )
    else:
        budget = float(args.budget)

    plan, report = build_consensus(plans, shapes, budget, "restore")
    if report["min_jaccard"] < args.min_jaccard:
        print("[consensus] FAIL: pairwise mask Jaccard below threshold:")
        for e in report["pairwise_jaccard"]:
            print(f"  {e['pair']}: {e['jaccard']:.3f}")
        print(f"[consensus] min {report['min_jaccard']:.3f} < {args.min_jaccard} — "
              "the dev-suite ranking is not reproducible; do NOT freeze a consensus plan.")
        sys.exit(1)

    out = {
        "meta": {
            "type": "consensus",
            "source_plans": [str(x) for x in args.plans],
            "budget_bytes": budget,
            "min_jaccard": report["min_jaccard"],
            "note": "majority vote (>=2/3) over dev-suite FP16 masks, budget-repaired; "
                    "frozen for Long/90 ZERO-SHOT evaluation (no re-probing)",
        },
        "total_bytes": report["total_bytes"],
        "budget_bytes": budget,
        "packdirs": {},
        "layers": plan,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[consensus] jaccard report: {report['pairwise_jaccard']}")
    print(f"[consensus] frozen unified plan -> {args.out} "
          f"({report['consensus_skip_count']} FP16 / {len(names) - report['consensus_skip_count']} W4, "
          f"{report['total_bytes'] / 1e6:.1f} MB / budget {budget / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
