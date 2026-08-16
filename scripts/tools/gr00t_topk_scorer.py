#!/usr/bin/env python3
"""GR00T v2 TopK D_solver adjudicator (v1.3 eight-step pipeline, §3.1 step 4-7).

Reads the diverse TopK from a selector plan JSON, scores EVERY candidate with
the config-level D_solver under TRUE deployment semantics (each plan is loaded
via GR00T_DUQUANT_PLAN: quantized layers are DuQuant-wrapped, skip layers stay
the original FP16 nn.Linear — not the weight_bits=0 approximation), verifies
the wrapped-layer count, then applies select_final() (min D_solver, 5% tie set
broken by the canonical proxy objective) and writes the FINAL plan.

Usage (groot_test env, one idle GPU):

    python scripts/tools/gr00t_topk_scorer.py \
        --plan checkpoints/packs/gr00t/gr00t_quant_plan_libero_spatial.json \
        --ckpt checkpoints/gr00t/libero-spatial --suite spatial \
        --packdir checkpoints/packs/gr00t/duquant_packed_libero_spatial_w4a8_b64c32ls015 \
        --out checkpoints/packs/gr00t/gr00t_quant_plan_libero_spatial_final.json
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "tools"))

from gr00t_v2_common import (  # noqa: E402
    DEFAULT_EXCLUDE,
    DEFAULT_INCLUDE,
    SUITE_DIRS,
    ensure_flash_attn_rpath,
    load_policy,
    make_obs,
    resolve_data_config,
    restore_quant_env,
    set_quant_env,
    strip_quant_env,
)
from gr00t_sensitivity_probe import run_rollouts, solver_divergence  # noqa: E402
from gr00t_select_plan import select_final  # noqa: E402


# --------------------------------------------------------------------------- #
# Plan materialization (offline-testable)
# --------------------------------------------------------------------------- #
def build_topk_plan_files(selector_plan: Dict[str, Any], packdir: str, out_dir: Path) -> List[Dict[str, Any]]:
    """Materialize each TopK entry into a full GR00T_DUQUANT_PLAN JSON file."""
    layers_all = selector_plan.get("layers", {})
    group = int(next(iter(selector_plan.get("packdirs", {})))) if selector_plan.get("packdirs") else 64
    out_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for i, entry in enumerate(selector_plan.get("topk", [])):
        skip = set(entry.get("skip_layers", []))
        plan_layers = {}
        for name in layers_all:
            if name in skip:
                plan_layers[name] = {"bits": None, "group": group, "skip": True}
            else:
                plan_layers[name] = {"bits": 4, "group": group, "skip": False}
        plan = {
            "meta": {
                "topk_source": entry.get("source"),
                "topk_index": i,
                "parent_plan": entry.get("objective"),
            },
            "packdirs": {str(group): packdir},
            "layers": plan_layers,
        }
        path = out_dir / f"topk_{i}_{entry.get('source', 'plan').replace('(', '_').replace(')', '_').replace(',', '_')}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(plan, f, indent=2)
        out.append({"file": str(path), "topk": entry, "plan": plan})
    return out


def count_wrapped(model: torch.nn.Module) -> int:
    from gr00t.quantization.duquant_layers import DuQuantLinear

    return sum(1 for m in model.modules() if isinstance(m, DuQuantLinear))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GR00T v2 TopK D_solver adjudicator (v1.3)")
    p.add_argument("--plan", default=None, help="Selector plan JSON with the diverse TopK.")
    p.add_argument("--ckpt", default=None, help="Checkpoint dir.")
    p.add_argument("--suite", default="spatial", choices=["spatial", "goal", "object", "90", "10"])
    p.add_argument("--data-config", default=None,
                   help="Default: resolved per suite via SUITE_DATA_CONFIG (goal -> MeanStd).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--denoising-steps", type=int, default=8)
    p.add_argument("--n-obs", type=int, default=16, help="Synthetic obs for D_solver (round 3: 16, with paired-bootstrap significance).")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--gamma", type=float, default=1.2)
    p.add_argument("--metric", default="d_solver", choices=["d_solver", "d_func"],
                   help="v1.4: adjudication metric (d_func = tail-aware functional metric).")
    p.add_argument("--group", type=int, default=64)
    p.add_argument("--ls", type=float, default=0.15)
    p.add_argument("--act-pct", type=float, default=99.9)
    p.add_argument("--row-rot", default="restore")
    p.add_argument("--calib-steps", type=int, default=32)
    p.add_argument("--packdir", default=None)
    p.add_argument("--act-scale-path", default=None,
                   help="Shared plan-specific A8 scale artifact (.npz).")
    p.add_argument("--out", default=None)
    p.add_argument("--tol", type=float, default=0.05, help="select_final relative tolerance.")
    p.add_argument("--selftest", action="store_true", help="Offline materialization + select_final test.")
    return p.parse_args()


def _selftest() -> None:
    import tempfile

    sel_plan = {
        "layers": {f"L{i}": {"bits": 4, "group": 64, "skip": False} for i in range(20)},
        "packdirs": {"64": "/tmp/pack"},
        "topk": [
            {"source": "greedy", "objective": 0.10, "skip_layers": [f"L{i}" for i in range(0, 8)], "d_solver": None},
            {"source": "milp", "objective": 0.11, "skip_layers": [f"L{i}" for i in range(2, 10)], "d_solver": None},
            {"source": "lambda(1,0.5)", "objective": 0.12, "skip_layers": [f"L{i}" for i in range(4, 12)], "d_solver": None},
        ],
    }
    with tempfile.TemporaryDirectory() as tmp:
        files = build_topk_plan_files(sel_plan, "/tmp/pack", Path(tmp))
        assert len(files) == 3
        d = json.loads(Path(files[0]["file"]).read_text())
        assert d["layers"]["L0"]["skip"] is True and d["layers"]["L8"]["skip"] is False
        # select_final: d_solver 0.20/0.21/0.205 -> min 0.20; tie set +5% -> 0.20..0.21
        # -> both 0.20 and 0.205 and 0.21? tol 5% of 0.20 = 0.01 -> T = {0.20, 0.205, 0.21}? no: 0.21 ≤ 0.21 yes
        scored = [
            {"source": "a", "d_solver": 0.200, "proxy": 0.10},
            {"source": "b", "d_solver": 0.205, "proxy": 0.09},
            {"source": "c", "d_solver": 0.230, "proxy": 0.01},
        ]
        fin = select_final(scored, tol=0.05)
        assert fin["source"] == "b", f"select_final picked {fin}"  # tie set {0.200,0.205}; min proxy 0.09
    print("[topk_scorer] selftest OK (plan materialization + select_final tie set)")


def main() -> None:
    args = parse_args()
    if args.selftest:
        _selftest()
        return

    if not args.plan or not args.ckpt:
        raise SystemExit("--plan/--ckpt are required (or use --selftest)")
    args.data_config = resolve_data_config(args.suite, args.data_config)
    sel_plan = json.loads(Path(args.plan).read_text())
    topk = sel_plan.get("topk", [])
    if not topk:
        raise SystemExit("selector plan has no topk entries")
    print(f"[topk_scorer] {len(topk)} TopK candidates from {args.plan}")

    if args.packdir is None:
        args.packdir = sel_plan.get("packdirs", {}).get(str(args.group))
        if not args.packdir:
            raise SystemExit("--packdir required (not present in the plan)")
    suite_dir = SUITE_DIRS[args.suite]

    ensure_flash_attn_rpath()
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    # ---- FP16 reference pass ----
    saved_env = strip_quant_env()
    policy_fp = load_policy(args.ckpt, data_config=args.data_config, denoising_steps=args.denoising_steps, device=args.device)
    model_fp = policy_fp.model
    horizon = int(model_fp.action_head.config.action_horizon)
    action_dim = int(model_fp.action_head.config.action_dim)
    obs_list = [make_obs(rng, "libero") for _ in range(args.n_obs)]
    noises = [torch.randn(horizon, action_dim) for _ in obs_list]
    fp_traj = run_rollouts(model_fp, policy_fp, obs_list, noises, args.batch_size)
    del model_fp, policy_fp
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    restore_quant_env(saved_env)

    # ---- ONE fixed calibration buffer for every TopK plan (review round 2,
    #      items 1/5): D_solver must be scored in the SAME frozen-A8 state the
    #      deployment will use, and on identical calibration data ----
    from gr00t_v2_common import ensure_a8_calibrated, fixed_calibration_buffer

    n_warm_obs = args.calib_steps * args.batch_size
    warm_obs, warm_noises, warm_sha = fixed_calibration_buffer(
        0, n_warm_obs, horizon, action_dim, fmt="libero"
    )
    print(f"[topk_scorer] fixed calibration buffer: {n_warm_obs} obs, sha256={warm_sha[:16]}...")

    # ---- materialize + score each TopK plan in true deployment semantics ----
    out_dir = Path(args.out + ".topk_tmp") if args.out else Path(args.plan).parent / "topk_tmp"
    files = build_topk_plan_files(sel_plan, args.packdir, out_dir)
    expected_total = len(sel_plan.get("layers", {}))
    scored: List[Dict[str, Any]] = []
    for f in files:
        plan_path = f["file"]
        entry = f["topk"]
        expected_wrapped = expected_total - len(entry.get("skip_layers", []))
        strip_quant_env()
        set_quant_env(DEFAULT_INCLUDE, DEFAULT_EXCLUDE, args.packdir,
                      bits_default=4, group=args.group, ls=args.ls,
                      act_pct=args.act_pct, calib_steps=args.calib_steps,
                      row_rot=args.row_rot, act_dynamic=False)
        import os
        os.environ["GR00T_DUQUANT_PLAN"] = plan_path
        t0 = time.time()
        policy_q = load_policy(args.ckpt, data_config=args.data_config, denoising_steps=args.denoising_steps, device=args.device)
        # wrap verification + static A8 calibration on the shared fixed buffer
        # (review round 2, item 1: D_solver was previously scored in the
        # provisional per-forward scale state, NOT the deployed frozen state)
        ensure_a8_calibrated(
            policy_q, warm_obs, warm_noises, args.batch_size,
            act_dynamic=False, expected_wrapped=expected_wrapped,
            act_scale_path=args.act_scale_path,
            act_scale_meta={
                "buffer_sha256": warm_sha,
                "data_config": args.data_config,
                "act_percentile": args.act_pct,
                "calib_batches": args.calib_steps,
                "denoising_steps": args.denoising_steps,
            },
        )
        n_wrapped = count_wrapped(policy_q.model)
        q_traj = run_rollouts(policy_q.model, policy_q, obs_list, noises, args.batch_size)
        mean_div, per_obs = solver_divergence(fp_traj, q_traj, args.gamma)
        # v1.4: tail-aware functional metric from the SAME paired trajectories
        from gr00t_func_metrics import d_func

        df = d_func(fp_traj, q_traj, args.gamma)
        scored.append({
            "source": entry.get("source"),
            "d_solver": mean_div,
            "d_solver_std": float(np.std(per_obs)) if len(per_obs) > 1 else 0.0,
            "d_func": df["d_func"],
            "d_func_components": {k: v for k, v in df.items() if k != "per_obs"},
            "proxy": float(entry.get("objective", float("inf"))),
            "n_wrapped": n_wrapped,
            "plan_file": plan_path,
            "_per_obs": per_obs,
            "_per_obs_func": df["per_obs"],
        })
        del policy_q, q_traj
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[topk_scorer] {entry['source']}: D_solver = {mean_div:.5f} ± "
              f"{scored[-1]['d_solver_std']:.5f} (wrapped {n_wrapped}) ({time.time() - t0:.1f}s)")

    # paired-bootstrap significance: best vs runner-up over shared obs indices
    # (review round 3, item 9 — a fixed 5% tie rule on 8-16 point estimates is
    # not a statistical decision; report the bootstrap evidence alongside it.
    # v1.4: the bootstrap uses the per-obs divergence list regardless of the
    # adjudication metric — D_func components are per-plan aggregates.)
    bootstrap = None
    if len(scored) >= 2 and all(len(s.get("_per_obs", [])) > 1 for s in scored):
        srt = sorted(scored, key=lambda c: c[args.metric])
        best, runner = srt[0], srt[1]
        pa = np.asarray(best["_per_obs"])
        pr = np.asarray(runner["_per_obs"])
        n = min(len(pa), len(pr))
        pa, pr = pa[:n], pr[:n]
        rng_boot = np.random.default_rng(123)
        idx = rng_boot.integers(0, n, size=(2000, n))
        diff = (pa[idx] - pr[idx]).mean(axis=1)  # best − runner (negative = best wins)
        bootstrap = {
            "best": best["source"], "runner_up": runner["source"],
            "mean_diff": float(np.mean(diff)),
            "p_best_wins": float((diff <= 0).mean()),
        }
    for s_ in scored:
        s_.pop("_per_obs", None)
        s_.pop("_per_obs_func", None)

    final = select_final(scored, tol=args.tol, key=args.metric)
    if final is None:
        raise SystemExit(f"[topk_scorer] select_final returned None (no {args.metric} scores)")
    final_plan = json.loads(Path(final["plan_file"]).read_text())
    base = Path(args.out or str(Path(args.plan).parent / (Path(args.plan).stem + "_adjudicated")))
    report = {
        "meta": {
            "parent_plan": args.plan, "suite": args.suite, "n_obs": args.n_obs,
            "tol": args.tol, "final_source": final["source"],
            "metric": args.metric,
            "calibration_buffer_sha256": warm_sha,
            "paired_bootstrap": bootstrap,
            "note": "true deployment semantics via GR00T_DUQUANT_PLAN (skip = unwrapped FP16); "
                    f"select_final = min {args.metric}, 5% tie set broken by canonical proxy; "
                    "one shared fixed A8 calibration buffer for all plans; "
                    "bootstrap significance uses per-obs divergences for both metrics",
        },
        "scored": scored,
        "final": {k: v for k, v in final.items() if k != "plan_file"},
        "final_plan_path": str(base.with_suffix(".final_plan.json")),
    }
    # review round 2, item 2: the deployable plan is a SEPARATE file whose
    # top level is exactly {packdirs, layers, meta} — the GR00T_DUQUANT_PLAN
    # loader reads top-level keys, so a nested final_plan would be silently
    # ignored (falling back to uniform W4).
    final_plan["meta"] = dict(final_plan.get("meta") or {})
    final_plan["meta"]["adjudicated"] = True
    final_plan["meta"]["final_source"] = final["source"]
    final_plan["meta"]["final_metric"] = args.metric
    final_plan["meta"][f"final_{args.metric}"] = final[args.metric]
    if args.metric == "d_func" and "d_solver" in final:
        final_plan["meta"]["final_d_solver"] = final["d_solver"]
    with open(base.with_suffix(".report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(base.with_suffix(".final_plan.json"), "w", encoding="utf-8") as f:
        json.dump(final_plan, f, indent=2)
    print(f"[topk_scorer] final = {final['source']} ({args.metric} {final[args.metric]:.5f})")
    print(f"[topk_scorer] report -> {base.with_suffix('.report.json')}")
    print(f"[topk_scorer] deployable plan -> {base.with_suffix('.final_plan.json')}")
    print("[topk_scorer] NEXT: run_quantvla.sh with "
          f"GR00T_DUQUANT_PLAN={base.with_suffix('.final_plan.json')}, then LIBERO acceptance (§6.5)")


if __name__ == "__main__":
    main()
