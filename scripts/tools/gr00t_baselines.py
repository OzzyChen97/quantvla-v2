#!/usr/bin/env python3
"""GR00T v2 baselines (v1.3, design doc §6.5): plan generators + two-stage screening.

The two comparisons the method MUST pass (§6.5):
  (1) v2 plan  vs  random W4/FP16 masks (same W4 layer count AND byte budget)
  (2) v2 plan  vs  uniform W6 (same static-byte budget)

Also generates the other Phase-1 controls: size-based greedy, v1-style manual
mask, uniform W4.

P0-6 (correctness review) fixes vs the first version:
  - the candidate set is EXACTLY the ref-plan layer keys (the 116 targets),
    never vision/projector/encoder linears;
  - random masks match the v2 plan's STATIC BYTES (|ΔC| ≤ 0.5%), not just the
    W4 layer count;
  - stage-2 loads every plan via GR00T_DUQUANT_PLAN — skip layers stay the
    original unwrapped FP16 nn.Linear (true deployment semantics) — and the
    wrapped-layer count is verified (n_applied == expected) before scoring;
  - the static A8 calibration runs to completion on every plan load.

Generate mode (CPU, offline):
    python scripts/tools/gr00t_baselines.py --mode generate \
        --ckpt checkpoints/gr00t/libero-spatial \
        --ref-plan checkpoints/packs/gr00t/gr00t_quant_plan_libero_spatial.json \
        --n-random 20 --out-dir checkpoints/packs/gr00t/baselines_spatial

Stage-2 mode (GPU): score every generated plan with the config-level D_solver
under TRUE deployment semantics, then report best/median/worst representatives:

    python scripts/tools/gr00t_baselines.py --mode stage2 --suite spatial \
        --plans-dir checkpoints/packs/gr00t/baselines_spatial \
        --out checkpoints/packs/gr00t/baselines_spatial_dsolver.json
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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
from gr00t_select_plan import layer_bytes_fp16, layer_bytes_quant, plan_total_bytes, read_layer_shapes  # noqa: E402

GROUP = 64
BYTE_TOL = 0.005  # |C_random − C_v2| / C_v2 ≤ 0.5%


# --------------------------------------------------------------------------- #
# Plan generators (CPU, offline-testable)
# --------------------------------------------------------------------------- #
def make_plan(shapes: Dict[str, Dict[str, Any]], w4_names: List[str]) -> Dict[str, Any]:
    w4 = set(w4_names)
    return {
        n: ({"bits": 4, "group": GROUP, "skip": False} if n in w4
            else {"bits": None, "group": GROUP, "skip": True})
        for n in shapes
    }


def _byte_delta_per_layer(shapes: Dict[str, Dict[str, Any]], row_rot: str) -> Dict[str, float]:
    """C_fp − C_w4 per layer (the savings of quantizing the layer to W4)."""
    return {
        n: layer_bytes_fp16(s["out"], s["in"], s["has_bias"])
        - layer_bytes_quant(s["out"], s["in"], s["has_bias"], 4, GROUP, row_rot=row_rot)
        for n, s in shapes.items()
    }


def gen_random_masks_bytes(
    shapes: Dict[str, Dict[str, Any]],
    n_w4: int,
    target_bytes: float,
    k: int,
    seed: int = 0,
    tol: float = BYTE_TOL,
    max_iters: int = 300,
    row_rot: str = "restore",
) -> List[Dict[str, Any]]:
    """k random W4/FP16 masks with the same W4 count AND matching static bytes
    (|ΔC|/C_v2 ≤ tol) via local swap refinement (P0-6)."""
    rng = random.Random(seed)
    names = list(shapes)
    n_w4 = max(0, min(n_w4, len(names)))
    savings = _byte_delta_per_layer(shapes, row_rot)
    fp_total = sum(layer_bytes_fp16(s["out"], s["in"], s["has_bias"]) for s in shapes.values())
    target_savings = fp_total - target_bytes
    plans: List[Dict[str, Any]] = []
    attempts = 0
    while len(plans) < k and attempts < k * 20:
        attempts += 1
        w4: Set[str] = set(rng.sample(names, n_w4))
        cur_savings = sum(savings[n] for n in w4)
        best = (w4.copy(), cur_savings)
        for _ in range(max_iters):
            delta = cur_savings - target_savings
            if abs(delta) <= tol * target_bytes:
                break
            w4_list = list(w4)
            out_list = [n for n in names if n not in w4]
            best_swap: Optional[Tuple[str, str, float]] = None  # (in, out, new_savings)
            for a in w4_list:
                for b in out_list:
                    cand = cur_savings - savings[a] + savings[b]
                    if best_swap is None or abs(cand - target_savings) < abs(best_swap[2] - target_savings):
                        best_swap = (a, b, cand)
            if best_swap is None:
                break
            a, b, cand = best_swap
            if abs(cand - target_savings) >= abs(cur_savings - target_savings):
                break  # no improving swap
            w4.remove(a)
            w4.add(b)
            cur_savings = cand
            if abs(cur_savings - target_savings) < abs(best[1] - target_savings):
                best = (w4.copy(), cur_savings)
        final_mask, final_savings = best
        plan = make_plan(shapes, sorted(final_mask))
        final_bytes = fp_total - final_savings
        if abs(final_bytes - target_bytes) / max(target_bytes, 1.0) > tol:
            continue  # reject unmatched masks (rare with fine-grained savings)
        plans.append(plan)
    return plans


def gen_size_based(shapes: Dict[str, Dict[str, Any]], n_w4: int) -> Dict[str, Any]:
    """Quantize the n_w4 LARGEST layers (static bytes) — controls for 'the
    selector only moves bytes to big layers'."""
    names = sorted(
        shapes,
        key=lambda n: layer_bytes_fp16(shapes[n]["out"], shapes[n]["in"], shapes[n]["has_bias"]),
        reverse=True,
    )
    return make_plan(shapes, names[:n_w4])


def gen_manual_mask(shapes: Dict[str, Dict[str, Any]], n_w4: int) -> Dict[str, Any]:
    """v1-style expert heuristic at the SAME W4 count: keep attention input
    projections (q/k/v) FP16 (v1's 'attention is fragile' rule), quantize the
    rest first; fill up from the q/k/v set only if needed."""
    qkv = [n for n in shapes if any(f".{p}" in n for p in ("q_proj", "k_proj", "v_proj"))]
    rest = [n for n in shapes if n not in set(qkv)]
    w4 = rest[:n_w4]
    if len(w4) < n_w4:
        w4 += qkv[: n_w4 - len(w4)]
    return make_plan(shapes, w4)


def gen_uniform(shapes: Dict[str, Dict[str, Any]], bits: int) -> Dict[str, Any]:
    return {n: {"bits": bits, "group": GROUP, "skip": False} for n in shapes}


def pick_representatives(scored: List[Dict[str, Any]]) -> Dict[str, Any]:
    """best / median / worst by D_solver from a scored plan list."""
    valid = [c for c in scored if c.get("d_solver") is not None]
    if not valid:
        return {"n": 0, "best": None, "median": None, "worst": None}
    s = sorted(valid, key=lambda c: c["d_solver"])
    return {"n": len(s), "best": s[0], "median": s[len(s) // 2], "worst": s[-1]}


def count_wrapped(model: torch.nn.Module) -> int:
    from gr00t.quantization.duquant_layers import DuQuantLinear

    return sum(1 for m in model.modules() if isinstance(m, DuQuantLinear))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GR00T v2 baselines (v1.3, §6.5, P0-6)")
    p.add_argument("--mode", default="generate", choices=["generate", "stage2", "selftest"])
    p.add_argument("--suite", default="spatial", choices=["spatial", "goal", "object", "90", "10"])
    p.add_argument("--model-path", default=None)
    p.add_argument("--ckpt", default=None, help="Checkpoint for shapes (generate mode).")
    p.add_argument("--ref-plan", default=None, help="v2 plan JSON: candidate set + W4 count + bytes.")
    p.add_argument("--n-random", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--byte-tol", type=float, default=BYTE_TOL)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--plans-dir", default=None, help="Plans dir for stage2.")
    p.add_argument("--out", default=None)
    p.add_argument("--n-obs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--gamma", type=float, default=1.2)
    p.add_argument("--denoising-steps", type=int, default=8)
    p.add_argument("--data-config", default=None,
                   help="Default: resolved per suite via SUITE_DATA_CONFIG (goal -> MeanStd).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--group", type=int, default=64)
    p.add_argument("--ls", type=float, default=0.15)
    p.add_argument("--act-pct", type=float, default=99.9)
    p.add_argument("--row-rot", default="restore")
    p.add_argument("--calib-steps", type=int, default=32)
    p.add_argument("--packdir", default=None)
    p.add_argument("--act-scale-path", default=None,
                   help="Shared plan-specific A8 scale artifact (.npz): reuse the SAME frozen scales across baseline scoring, TopK, calibrator and server.")
    return p.parse_args()


def _selftest() -> None:
    rng = random.Random(0)
    shapes = {f"L{i}": {"out": 2048 + i * 128, "in": 2048, "has_bias": False} for i in range(40)}
    n_w4 = 25
    fp_total = sum(layer_bytes_fp16(s["out"], s["in"], s["has_bias"]) for s in shapes.values())
    # reference bytes: an arbitrary feasible plan
    ref = make_plan(shapes, [f"L{i}" for i in range(n_w4)])
    target = plan_total_bytes(ref, shapes, "restore")
    masks = gen_random_masks_bytes(shapes, n_w4, target, k=8, seed=1)
    assert len(masks) == 8, f"only {len(masks)} byte-matched masks"
    for m in masks:
        assert sum(1 for v in m.values() if not v["skip"]) == n_w4
        assert abs(plan_total_bytes(m, shapes, "restore") - target) / target <= BYTE_TOL + 1e-9
    sz = gen_size_based(shapes, n_w4)
    sz_w4 = [n for n, v in sz.items() if not v["skip"]]
    expect = sorted(shapes, key=lambda n: shapes[n]["out"] * shapes[n]["in"], reverse=True)[:n_w4]
    assert set(sz_w4) == set(expect), "size mask must pick the largest layers"
    man = gen_manual_mask(shapes, n_w4)
    assert sum(1 for v in man.values() if not v["skip"]) == n_w4
    uni4 = gen_uniform(shapes, 4)
    assert all(not v["skip"] for v in uni4.values())
    scored = [{"index": i, "d_solver": 0.1 + (i % 7) * 0.01} for i in range(7)]
    rep = pick_representatives(scored)
    assert rep["n"] == 7 and rep["best"]["index"] == 0 and rep["worst"]["index"] == 6
    print("[baselines] selftest OK (byte-matched random masks + generators + representatives)")
    print(f"  random masks   {len(masks)} × {n_w4} W4, |Δbytes|/C ≤ {BYTE_TOL:.1%}")
    print(f"  size mask      largest-{n_w4} layers ✓ | manual mask {n_w4} W4 ✓")


def _generate(args: argparse.Namespace) -> None:
    if not args.ckpt or not args.ref_plan:
        raise SystemExit("--mode generate needs --ckpt and --ref-plan")
    all_shapes = read_layer_shapes(Path(args.ckpt), r".*", r"^$")
    ref = json.loads(Path(args.ref_plan).read_text())
    # P0-6: the candidate space is EXACTLY the ref-plan layer keys
    cand_keys = list(ref.get("layers", {}).keys())
    shapes = {n: all_shapes[n] for n in cand_keys if n in all_shapes}
    missing = set(cand_keys) - set(shapes)
    if missing:
        raise SystemExit(f"[baselines] {len(missing)} ref-plan layers missing from checkpoint shapes")
    n_w4 = sum(1 for v in ref.get("layers", {}).values() if not v.get("skip"))
    target_bytes = float(ref.get("total_bytes") or plan_total_bytes(ref["layers"], shapes, args.row_rot))
    print(f"[baselines] v2 plan: {n_w4}/{len(shapes)} W4 layers, {target_bytes / 1e6:.1f} MB; "
          f"candidate space = ref-plan keys only")
    out_dir = Path(args.out_dir or str(Path(args.ref_plan).parent / f"baselines_{args.suite}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    plans: Dict[str, List[Dict[str, Any]]] = {}
    plans["random"] = gen_random_masks_bytes(shapes, n_w4, target_bytes, k=args.n_random,
                                             seed=args.seed, tol=args.byte_tol, row_rot=args.row_rot)
    plans["size_based"] = [gen_size_based(shapes, n_w4)]
    plans["manual_v1_style"] = [gen_manual_mask(shapes, n_w4)]
    plans["uniform_w4"] = [gen_uniform(shapes, 4)]
    plans["uniform_w6"] = [gen_uniform(shapes, 6)]

    for tag, plist in plans.items():
        for i, pl in enumerate(plist):
            name = f"{tag}_{i}" if len(plist) > 1 else tag
            out = {
                "meta": {
                    "baseline": tag, "index": i, "n_w4": n_w4, "n_layers": len(shapes),
                    "reference_plan": args.ref_plan, "target_bytes": target_bytes,
                    "note": "same W4 count as the v2 plan AND byte-matched within "
                            f"{args.byte_tol:.1%} (random); size/manual match the W4 count; "
                            "uniform W6 is the same-static-byte-budget control",
                },
                "total_bytes": plan_total_bytes(pl, shapes, args.row_rot),
                "layers": pl,
            }
            with open(out_dir / f"{name}.json", "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
    print(f"[baselines] wrote {sum(len(v) for v in plans.values())} plans -> {out_dir}")


def _stage2(args: argparse.Namespace) -> None:
    args.data_config = resolve_data_config(args.suite, args.data_config)
    plans_dir = Path(args.plans_dir or str(REPO_ROOT / "checkpoints/packs/gr00t" / f"baselines_{args.suite}"))
    plan_files = sorted(plans_dir.glob("*.json"))
    if not plan_files:
        raise SystemExit(f"no plan files under {plans_dir}")
    plans = []
    for pf in plan_files:
        d = json.loads(pf.read_text())
        d["_file"] = pf.name
        plans.append(d)
    print(f"[baselines] stage2: {len(plans)} plans from {plans_dir}")

    suite_dir = SUITE_DIRS[args.suite]
    if args.model_path is None:
        args.model_path = str(REPO_ROOT / "checkpoints" / "gr00t" / suite_dir)
    if args.packdir is None:
        args.packdir = str(
            REPO_ROOT / "checkpoints/packs/gr00t"
            / f"duquant_packed_libero_{args.suite}_w4a8_b64c32ls015"
        )
    ensure_flash_attn_rpath()
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    # ---- FP16 reference pass ----
    saved_env = strip_quant_env()
    policy_fp = load_policy(args.model_path, data_config=args.data_config, denoising_steps=args.denoising_steps, device=args.device)
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

    # ---- ONE fixed calibration buffer shared by EVERY plan (review round 2,
    #      item 5): all plans must freeze their A8 scales on identical data ----
    from gr00t_v2_common import ensure_a8_calibrated, fixed_calibration_buffer

    n_warm_obs = args.calib_steps * args.batch_size
    warm_obs, warm_noises, warm_sha = fixed_calibration_buffer(
        0, n_warm_obs, horizon, action_dim, fmt="libero"
    )
    print(f"[baselines] fixed calibration buffer: {n_warm_obs} obs, sha256={warm_sha[:16]}...")

    # ---- score each plan under TRUE deployment semantics (GR00T_DUQUANT_PLAN) ----
    tmp_dir = plans_dir / ".stage2_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    scored = []
    for d in plans:
        plan_path = tmp_dir / d["_file"]
        plan_doc = {"packdirs": {str(args.group): args.packdir}, "layers": d["layers"],
                    "meta": d.get("meta", {})}
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(plan_doc, f, indent=2)
        expected_wrapped = sum(1 for v in d["layers"].values() if not v.get("skip"))

        strip_quant_env()
        set_quant_env(DEFAULT_INCLUDE, DEFAULT_EXCLUDE, args.packdir,
                      bits_default=4, group=args.group, ls=args.ls,
                      act_pct=args.act_pct, calib_steps=args.calib_steps,
                      row_rot=args.row_rot, act_dynamic=False)
        os.environ["GR00T_DUQUANT_PLAN"] = str(plan_path)
        t0 = time.time()
        policy_q = load_policy(args.model_path, data_config=args.data_config, denoising_steps=args.denoising_steps, device=args.device)
        n_wrapped = count_wrapped(policy_q.model)
        if n_wrapped != expected_wrapped:
            raise SystemExit(
                f"[baselines] wrap mismatch for {d['_file']}: {n_wrapped} wrapped, "
                f"expected {expected_wrapped} — plan not applied in deployment semantics"
            )
        # A8 calibration to completion in the plan state on the SHARED buffer
        # (or load the shared frozen artifact via --act-scale-path)
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

        q_traj = run_rollouts(policy_q.model, policy_q, obs_list, noises, args.batch_size)
        mean_div, per_obs = solver_divergence(fp_traj, q_traj, args.gamma)
        scored.append({
            "file": d["_file"], "tag": d.get("meta", {}).get("baseline"),
            "d_solver": mean_div,
            "d_solver_std": float(np.std(per_obs)) if len(per_obs) > 1 else 0.0,
            "n_wrapped": n_wrapped,
        })
        del policy_q, q_traj
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[baselines] {d['_file']}: D_solver = {mean_div:.5f} ± "
              f"{scored[-1]['d_solver_std']:.5f} (wrapped {n_wrapped}) ({time.time() - t0:.1f}s)")

    rep = pick_representatives(scored)
    out = {
        "meta": {"suite": args.suite, "n_obs": args.n_obs, "plans_dir": str(plans_dir),
                 "calibration_buffer_sha256": warm_sha,
                 "note": "true deployment semantics via GR00T_DUQUANT_PLAN; "
                         "one shared fixed A8 calibration buffer for all plans; "
                         "take best/median/worst to LIBERO (two-stage protocol, §6.5)"},
        "scored": scored,
        "representatives": rep,
    }
    out_path = Path(args.out or str(plans_dir / "dsolver_report.json"))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[baselines] report -> {out_path}")
    if rep["n"]:
        for k in ("best", "median", "worst"):
            r = rep[k]
            print(f"[baselines] {k}: {r['file']} D_solver={r['d_solver']:.5f}")


def main() -> None:
    args = parse_args()
    if args.mode == "selftest":
        _selftest()
    elif args.mode == "generate":
        _generate(args)
    else:
        _stage2(args)


if __name__ == "__main__":
    main()
