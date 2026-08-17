#!/usr/bin/env python3
"""Plan-specific STATIC ATM/OHB calibration wrapper (v1.4, D-020 route 3).

Runs the per-step calibrator (calibrate_atm_perstep_gr00t.py) with --plan
<v1.4 quant plan> — plan-aware neutral forcing for all-FP16 attention blocks,
A8-scale-shared calibration, CV_t statistics — then strips the per-step
tables. The emitted artifact is a STATIC-only JSON:

    { "<attn layer>": {"all": [...], "beta_perhead": [...]}, ... }

consumed by enable_dit_atm_if_configured with GR00T_ATM_PER_STEP=0 (the
default), i.e. exactly the deployment mode of the v1.4 config matrix
(uniform W6 / v1.4 + static ATM/OHB). The CV_t sidecar records whether the
static table was judged sufficient for each head.

Usage (groot_test env, one idle GPU):
    python scripts/tools/calibrate_atm_static_plan.py --suite spatial \
        --plan checkpoints/packs/gr00t/gr00t_quant_plan_libero_spatial_v14.final_plan.json \
        --out checkpoints/packs/gr00t/atm_alpha_beta_static_spatial_v14.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v1.4 static plan-specific ATM/OHB calibration wrapper")
    p.add_argument(
        "--suite",
        default="spatial",
        choices=["spatial", "goal", "object", "90", "10", "robocasa365_atomic"],
    )
    p.add_argument("--plan", required=True, help="v1.4 quant plan JSON (final_plan).")
    p.add_argument("--act-scale-path", default=None,
                   help="Shared plan-specific A8 scale artifact (.npz).")
    p.add_argument("--out", required=True)
    p.add_argument("--n-obs", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--pass-through", nargs=argparse.REMAINDER, default=[],
                   help="Extra args forwarded to calibrate_atm_perstep_gr00t.py.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tmp_out = str(Path(args.out).with_suffix(".perstep_tmp.json"))
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "tools" / "calibrate_atm_perstep_gr00t.py"),
        "--suite", args.suite,
        "--plan", args.plan,
        "--out", tmp_out,
        "--n-obs", str(args.n_obs),
        "--device", args.device,
    ]
    if args.act_scale_path:
        cmd += ["--act-scale-path", args.act_scale_path]
    cmd += args.pass_through
    print(f"[calibrate-static] running: {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    if r.returncode != 0:
        raise SystemExit(f"[calibrate-static] per-step calibration failed (exit {r.returncode})")

    data = json.loads(Path(tmp_out).read_text())
    static = {}
    for layer, entry in data.items():
        assert "all" in entry and "beta_perhead" in entry, f"unexpected entry for {layer}"
        static[layer] = {"all": entry["all"], "beta_perhead": entry["beta_perhead"]}
    sidecar_path = tmp_out.replace(".json", ".cv_stats.json")
    sidecar = json.loads(Path(sidecar_path).read_text()) if Path(sidecar_path).exists() else {}
    n_heads = sum(len(v["all"]) for v in static.values())
    n_forced = sum(1 for m in sidecar.get("plan_marks", {}).values() if m.get("forced_neutral"))
    print(f"[calibrate-static] {len(static)} layers, {n_heads} heads; "
          f"plan-aware forced-neutral blocks: {n_forced}; "
          f"CV_t static_sufficient={sidecar.get('cv_stats', {}).get('static_sufficient')}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(static, f, indent=2)
    with open(str(out_path).replace(".json", ".meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "static_only": True,
            "plan": args.plan,
            "plan_sha256": _sha256(args.plan),
            "act_scale_path": args.act_scale_path,
            "act_scale_sha256": _sha256(args.act_scale_path) if args.act_scale_path else None,
            "per_step_source": tmp_out,
            "plan_marks": sidecar.get("plan_marks", {}),
            "cv_stats": sidecar.get("cv_stats", {}),
            "deploy_env": "GR00T_ATM_ENABLE=1 GR00T_OHB_ENABLE=1 "
                          "GR00T_ATM_ALPHA_PATH=<this file> (GR00T_ATM_PER_STEP=0, default)",
        }, f, indent=2)
    Path(tmp_out).unlink(missing_ok=True)
    print(f"[calibrate-static] saved static artifact -> {args.out}")
    print(f"[calibrate-static] meta -> {str(out_path).replace('.json', '.meta.json')}")
    print("[calibrate-static] done.")


if __name__ == "__main__":
    main()
