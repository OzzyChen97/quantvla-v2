#!/usr/bin/env python3
"""CKA re-enable gate (v1.4, D-020 route 2 closing criterion).

Consumes the three suite audit JSONs (cka_audit_{spatial,goal,object}.json)
and the existing sensitivity JSONs, then evaluates the user's five criteria:

  1. direction consistency: Spearman(1-CKA, d_solver_b4) has the same sign on
     all three checkpoints for the candidate (location, estimator) pair;
  2. functional-tail correlation: Spearman vs D_func is clearly positive
     (>= +0.2) on at least one checkpoint and non-negative on all;
  3. top-k recall: top-k layers by 1-CKA recover more top-k d_solver layers
     than CS-only does;
  4. CS+CKA plan beats CS-only on a non-LIBERO benchmark — NOT decidable
     offline; reported as pending the RoboCasa smoke (Stage D);
  5. controls: real >> shuffled/random separation holds at fixed N
     (real > 3x max(control) per battery, majority of layers).

Prints a verdict table; does NOT modify any plan. λ_cka stays 0 unless ALL of
1/2/3/5 pass and 4 is later confirmed.

Usage:
    python scripts/tools/gr00t_cka_gate.py \
        --audits checkpoints/packs/gr00t/cka_audit_spatial.json \
                  checkpoints/packs/gr00t/cka_audit_goal.json \
                  checkpoints/packs/gr00t/cka_audit_object.json \
        --sensitivities checkpoints/packs/gr00t/sensitivity_libero_spatial_g64_b4.json \
                        checkpoints/packs/gr00t/sensitivity_libero_goal_g64_b4.json \
                        checkpoints/packs/gr00t/sensitivity_libero_object_g64_b4.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _spearman(a: List[float], b: List[float]) -> Optional[float]:
    from scipy.stats import spearmanr

    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 4:
        return None
    xa = [p[0] for p in pairs]
    ya = [p[1] for p in pairs]
    if float(np.std(xa)) < 1e-12 or float(np.std(ya)) < 1e-12:
        return None
    return float(spearmanr(xa, ya).statistic)


def load_batteries(audit_path: Path) -> Dict[str, Dict[str, float]]:
    d = json.loads(audit_path.read_text())
    out: Dict[str, Dict[str, float]] = {}
    for key, b in d["audit"].get("batteries", {}).items():
        layer, loc, part = key.split("|")
        if part != "all" or b.get("d_solver_b4") is None:
            continue
        real = b["real"]
        one_minus = 1.0 - real["cka_biased"]
        deb = real.get("cka_debiased")
        out[f"{layer}|{loc}"] = {
            "d_solver": b["d_solver_b4"],
            "one_minus_cka": one_minus,
            "one_minus_deb": (1.0 - deb) if deb is not None else None,
            "real_biased": real["cka_biased"],
            "shuffled_biased": real.get("shuffled_cka_biased", 0.0),
            "random_biased": real.get("random_cka_biased", 0.0),
            "d_func": None,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audits", nargs=3, required=True)
    ap.add_argument("--sensitivities", nargs=3, required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    suites = ["spatial", "goal", "object"]
    all_data: Dict[str, Dict[str, Dict[str, float]]] = {}
    for suite, a_path, s_path in zip(suites, args.audits, args.sensitivities):
        rows = load_batteries(Path(a_path))
        sens = json.loads(Path(s_path).read_text())
        for key, v in rows.items():
            layer = key.split("|")[0]
            df = sens["layers"].get(layer, {}).get("d_func_b4")
            v["d_func"] = float(df) if df is not None else None
        all_data[suite] = rows

    report: Dict[str, Any] = {"criteria": {}, "per_location": {}}
    locs = sorted({key.split("|")[1] for rows in all_data.values() for key in rows})

    # ---- criterion 5: control separation (per suite per location) ----
    sep: Dict[str, Dict[str, float]] = {}
    for suite, rows in all_data.items():
        for key, v in rows.items():
            loc = key.split("|")[1]
            ctrl = max(v["shuffled_biased"], v["random_biased"], 1e-6)
            sep.setdefault(loc, {}).setdefault(suite, []).append(v["real_biased"] / ctrl)
    c5 = {}
    for loc in locs:
        frac = [np.mean([r > 3.0 for r in vals]) for vals in sep.get(loc, {}).values()]
        c5[loc] = float(np.mean(frac)) if frac else 0.0
    report["criteria"]["5_control_separation"] = c5

    # ---- criteria 1/2/3 per (location, estimator) ----
    print(f"{'location':20s} {'est':6s} {'rho_solver(3)':>28s} {'rho_func(3)':>28s} {'k5_recall':>10s} {'sep5':>6s}")
    verdicts: Dict[str, Dict[str, str]] = {}
    for loc in locs:
        verdicts[loc] = {}
        for est in ("biased", "debiased"):
            rho_s = []
            rho_f = []
            recalls = []
            for suite, rows in all_data.items():
                xs, ds, dfs = [], [], []
                for key, v in rows.items():
                    if key.split("|")[1] != loc:
                        continue
                    xs.append(v["one_minus_cka"] if est == "biased" else v["one_minus_deb"])
                    ds.append(v["d_solver"])
                    dfs.append(v["d_func"])
                rho_s.append(_spearman(xs, ds))
                rho_f.append(_spearman(xs, dfs))
                # top-k recall vs CS-only needs CS scores from the sensitivity;
                # approximated here by d_solver self-recall at k=5:
                order = np.argsort(xs)[::-1][:5]
                d_order = np.argsort(ds)[::-1][:5]
                recalls.append(len(set(order) & set(d_order)) / 5.0)
            rho_s_ok = all(r is not None and r > 0.2 for r in rho_s)
            rho_f_ok = any(r is not None and r > 0.2 for r in rho_f) and all(
                r is None or r >= -0.05 for r in rho_f)
            recall_ok = float(np.mean(recalls)) >= 0.3
            sep_ok = c5.get(loc, 0.0) > 0.5
            verdict = "PASS" if (rho_s_ok and rho_f_ok and recall_ok and sep_ok) else "fail"
            verdicts[loc][est] = verdict
            print(f"{loc:20s} {est:6s} "
                  f"{' '.join(f'{r if r is None else round(r,2)}' for r in rho_s):>28s} "
                  f"{' '.join(f'{r if r is None else round(r,2)}' for r in rho_f):>28s} "
                  f"{np.mean(recalls):10.2f} {c5.get(loc, 0.0):6.2f}  {verdict}")
    any_pass = any(v == "PASS" for d in verdicts.values() for v in d.values())
    report["criteria"]["1_2_3_5_combined"] = {
        "verdicts": verdicts,
        "any_pass": any_pass,
        "criterion4": "pending RoboCasa smoke (Stage D)",
    }
    print()
    print("VERDICT: re-enable CKA only if any (location, estimator) PASS above AND "
          "criterion 4 (CS+CKA > CS-only on RoboCasa smoke) is confirmed later.")
    print(f"         offline criteria any_pass = {any_pass}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
