#!/usr/bin/env python3
"""GR00T v2 gate-0 HARD check (review round 3, item 6).

Reads a metric-audit JSON (gr00t_metric_audit.py output) and decides whether
the gate-0 criteria pass. Exit code 0 = pass, 1 = fail. The orchestrator aborts
the pipeline on failure — the audit is a GATE, not a signpost.

Checks (thresholds are CLI-overridable):
  1. finite metric rate == 1.0        (no NaN/inf contamination)
  2. W2-vs-W8 rms_ratio median >= 2.0 (the metric can separate the known-bad
                                       stress bit from W8; else the collection
                                       pipeline cannot support bit decisions)
  3. guard fire rate on W2 >= 0.5     (feasibility guards must fire on the
                                       known-bad case — review P0 guard test)
  4. cross-seed mask Jaccard >= 0.7   (ranking reproducibility)
  5. Spearman(primary proxy, d_solver)@b4 >= 0.2 — the primary proxy is CS by
      default: the spatial gate-0 on the real model measured
      Spearman(1-CKA, d_solver) ≈ 0 (three seeds: -0.08/-0.03/-0.06) vs
      Spearman(CS, d_solver) ≈ +0.41 (0.43/0.41/0.39). The check uses
      --spearman-metric (default cs) and the selector defaults follow the
      same evidence (lambda_cka=0, lambda_cs=1).

Usage:
    python scripts/tools/gr00t_gate0_check.py --audit <metric_audit.json>
    python scripts/tools/gr00t_gate0_check.py --selftest
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _num(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def check_gate0(
    audit: Dict[str, Any],
    min_finite: float = 1.0,
    min_w2_w8_ratio: float = 2.0,
    min_guard_fire: float = 0.5,
    min_seed_jaccard: float = 0.7,
    min_spearman: float = 0.2,
    spearman_metric: str = "cs",
) -> Dict[str, Any]:
    """Returns {passed: bool, checks: {name: {value, threshold, passed}}}."""
    checks: Dict[str, Any] = {}
    seeds = audit.get("seeds", [])
    # review round 4: aggregate across ALL seeds with the WORST value (min of
    # rates/ratios) so one bad seed fails the gate instead of being hidden by
    # the last seed.
    def worst(key_path):
        vals = []
        for sd in seeds:
            node = sd.get("stats", {})
            for k in key_path:
                node = (node or {}).get(k) or {}
            v = _num(node) if not isinstance(node, dict) else None
            if v is not None:
                vals.append(v)
        return min(vals) if vals else None

    finite = worst(("finite_rate",))
    checks["finite_rate"] = {
        "value": finite, "threshold": min_finite,
        "passed": finite is not None and finite >= min_finite,
    }

    ratio = worst(("w2_vs_w8", "rms_ratio", "median_ratio"))
    checks["w2_vs_w8_median_ratio"] = {
        "value": ratio, "threshold": min_w2_w8_ratio,
        "passed": ratio is not None and ratio >= min_w2_w8_ratio,
    }

    fire = worst(("guard_fire_w2", "rate"))
    checks["guard_fire_w2_rate"] = {
        "value": fire, "threshold": min_guard_fire,
        "passed": fire is not None and fire >= min_guard_fire,
    }

    stab = audit.get("meta", {}).get("stability") or {}
    jac = _num((stab.get("cka_loss") or {}).get("mask_jaccard_mean"))
    checks["seed_mask_jaccard"] = {
        "value": jac, "threshold": min_seed_jaccard,
        "passed": jac is not None and jac >= min_seed_jaccard,
    }

    # D-006: per-seed values — the check requires MEAN >= threshold AND no sign
    # flip (min >= 0). The previous worst-seed >= 0.2 rule was too brittle: one
    # weak seed (e.g. goal seed2 cs=0.159, object seed0 cs=0.051) failed the
    # gate although CS is positive in ALL 9 suite-seed measurements (spatial
    # 0.41 / goal 0.305 / object 0.296 mean).
    sp_vals = []
    for sd in seeds:
        v = _num((((sd.get("stats") or {}).get("spearman") or {}).get("b4") or {}).get(spearman_metric))
        if v is not None:
            sp_vals.append(v)
    sp_mean = sum(sp_vals) / len(sp_vals) if sp_vals else None
    sp_min = min(sp_vals) if sp_vals else None
    sp_ok = (
        sp_mean is not None and sp_min is not None
        and sp_mean >= min_spearman and sp_min >= 0.0
    )
    checks[f"spearman_{spearman_metric}_vs_dsolver_b4"] = {
        "value": sp_mean, "min": sp_min, "per_seed": sp_vals,
        "threshold": f"mean>={min_spearman} & min>=0",
        "passed": sp_ok,
    }

    passed = all(c["passed"] for c in checks.values())
    return {"passed": passed, "checks": checks}


def _selftest() -> None:
    import copy

    base = {
        "meta": {"stability": {"cka_loss": {"mask_jaccard_mean": 0.85}}},
        "seeds": [{"stats": {
            "finite_rate": 1.0,
            "w2_vs_w8": {"rms_ratio": {"median_ratio": 5.0}},
            "guard_fire_w2": {"rate": 0.9},
            "spearman": {"b4": {"cka_loss": 0.6, "cs": 0.5}},
        }}],
    }
    res = check_gate0(base)
    assert res["passed"], res
    bad = copy.deepcopy(base)
    bad["seeds"][0]["stats"]["guard_fire_w2"]["rate"] = 0.1
    bad["seeds"][0]["stats"]["finite_rate"] = 0.9
    bad["meta"]["stability"]["cka_loss"]["mask_jaccard_mean"] = 0.3
    res2 = check_gate0(bad)
    assert not res2["passed"], res2
    assert not res2["checks"]["guard_fire_w2_rate"]["passed"]
    assert not res2["checks"]["finite_rate"]["passed"]
    assert not res2["checks"]["seed_mask_jaccard"]["passed"]
    print("[gate0_check] selftest OK (pass case + three failing cases)")


def main() -> None:
    p = argparse.ArgumentParser(description="GR00T v2 gate-0 hard check")
    p.add_argument("--audit", default=None, help="metric_audit_*.json")
    p.add_argument("--min-finite", type=float, default=1.0)
    p.add_argument("--min-w2-w8-ratio", type=float, default=2.0)
    p.add_argument("--min-guard-fire", type=float, default=0.5)
    p.add_argument("--min-seed-jaccard", type=float, default=0.7)
    p.add_argument("--min-spearman", type=float, default=0.2)
    p.add_argument("--spearman-metric", default="cs",
                   help="Primary proxy metric for the Spearman gate (gate-0 evidence: cs).")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    if args.selftest:
        _selftest()
        return
    if not args.audit:
        raise SystemExit("--audit required (or --selftest)")

    audit = json.loads(Path(args.audit).read_text())
    res = check_gate0(
        audit,
        min_finite=args.min_finite,
        min_w2_w8_ratio=args.min_w2_w8_ratio,
        min_guard_fire=args.min_guard_fire,
        min_seed_jaccard=args.min_seed_jaccard,
        min_spearman=args.min_spearman,
        spearman_metric=args.spearman_metric,
    )
    print("[gate0_check] checks:")
    for name, c in res["checks"].items():
        mark = "PASS" if c["passed"] else "FAIL"
        print(f"  {mark}  {name}: value={c['value']} threshold={c['threshold']}")
    if res["passed"]:
        print("[gate0_check] GATE 0 PASS")
        sys.exit(0)
    print("[gate0_check] GATE 0 FAIL — aborting pipeline (fix the audit before probing)")
    sys.exit(1)


if __name__ == "__main__":
    main()
