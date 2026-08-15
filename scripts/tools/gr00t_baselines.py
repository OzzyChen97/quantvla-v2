#!/usr/bin/env python3
"""GR00T v2 baselines (v1.3, design doc §6.5): plan generators + two-stage screening.

The two comparisons the method MUST pass (§6.5):
  (1) v2 plan  vs  random W4/FP16 masks (same W4 layer count)  — ranking is real
  (2) v2 plan  vs  uniform W6 (same static-byte budget)        — beats uniformity

Also generates the other Phase-1 controls: size-based greedy, v1-style manual
mask (same W4 count), uniform W4.

Generate mode (CPU, offline):
    python scripts/tools/gr00t_baselines.py --mode generate \
        --ckpt checkpoints/gr00t/libero-spatial \
        --ref-plan checkpoints/packs/gr00t/gr00t_quant_plan_libero_spatial.json \
        --n-random 20 --out-dir checkpoints/packs/gr00t/baselines_spatial

Stage-2 mode (GPU): scores every generated plan with the config-level D_solver
(FP16 vs complete plan, wrapped-pipeline approximation: skip layers run at
weight_bits=0 — the full-precision weight path of the same wrapper), reports the
distribution and picks best/median/worst representatives to take to LIBERO:

    python scripts/tools/gr00t_baselines.py --mode stage2 --suite spatial \
        --plans-dir checkpoints/packs/gr00t/baselines_spatial \
        --out checkpoints/packs/gr00t/baselines_spatial_dsolver.json
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "tools"))

from gr00t_v2_common import (  # noqa: E402
    PACKDIR_TEMPLATE,
    SUITE_DIRS,
    ensure_flash_attn_rpath,
    load_policy,
    make_obs,
    restore_quant_env,
    set_quant_env,
    strip_quant_env,
)
from gr00t_sensitivity_probe import (  # noqa: E402
    REF_BITS,
    run_rollouts,
    set_all_bits,
    set_single_layer_bits,
    solver_divergence,
)
from gr00t_select_plan import layer_bytes_fp16, plan_total_bytes, read_layer_shapes  # noqa: E402

GROUP = 64


# --------------------------------------------------------------------------- #
# Plan generators (CPU, offline-testable)
# --------------------------------------------------------------------------- #
def make_plan(shapes: Dict[str, Dict[str, Any]], w4_names: List[str]) -> Dict[str, Any]:
    w4 = set(w4_names)
    plan = {}
    for n in shapes:
        if n in w4:
            plan[n] = {"bits": 4, "group": GROUP, "skip": False}
        else:
            plan[n] = {"bits": None, "group": GROUP, "skip": True}
    return plan


def gen_random_masks(
    shapes: Dict[str, Dict[str, Any]], n_w4: int, k: int, seed: int = 0
) -> List[Dict[str, Any]]:
    """k random W4/FP16 masks with EXACTLY n_w4 W4 layers (same count as the v2
    plan — the fair comparison of selection QUALITY)."""
    rng = random.Random(seed)
    names = list(shapes)
    n_w4 = max(0, min(n_w4, len(names)))
    plans = []
    for _ in range(k):
        w4 = rng.sample(names, n_w4)
        plans.append(make_plan(shapes, w4))
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
    return {
        "n": len(s),
        "best": s[0],
        "median": s[len(s) // 2],
        "worst": s[-1],
    }


# --------------------------------------------------------------------------- #
# Stage-2: config-level D_solver screening (GPU)
# --------------------------------------------------------------------------- #
def apply_plan(model: torch.nn.Module, plan: Dict[str, Any]) -> int:
    """Approximate a plan inside the wrapped pipeline: W4 layers at weight_bits=4,
    skip layers at weight_bits=0 (the full-precision path of the same wrapper —
    NOT the pure FP16 layer, but the documented screening approximation)."""
    set_all_bits(model, REF_BITS)
    count = 0
    for n in plan:
        if plan[n].get("skip"):
            continue
        if set_single_layer_bits(model, n, int(plan[n]["bits"])):
            count += 1
    return count


def score_plans(
    policy_q: Any, fp_traj: torch.Tensor, plans: List[Dict[str, Any]],
    obs_list: List[Dict[str, Any]], noises: List[torch.Tensor],
    batch_size: int, gamma: float,
) -> List[Dict[str, Any]]:
    model_q = policy_q.model
    out: List[Dict[str, Any]] = []
    for i, plan in enumerate(plans):
        n_applied = apply_plan(model_q, plan)
        q_traj = run_rollouts(model_q, policy_q, obs_list, noises, batch_size)
        mean_div, per_obs = solver_divergence(fp_traj, q_traj, gamma)
        out.append({
            "index": i, "d_solver": mean_div,
            "d_solver_std": float(np.std(per_obs)) if len(per_obs) > 1 else 0.0,
            "n_applied": n_applied, "plan": plan,
        })
        del q_traj
        gc.collect()
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GR00T v2 baselines (v1.3, §6.5)")
    p.add_argument("--mode", default="generate", choices=["generate", "stage2", "selftest"])
    p.add_argument("--suite", default="spatial", choices=["spatial", "goal", "object", "90", "10"])
    p.add_argument("--model-path", default=None)
    p.add_argument("--ckpt", default=None, help="Checkpoint for shapes (generate mode).")
    p.add_argument("--ref-plan", default=None, help="v2 plan JSON: read the W4 layer count.")
    p.add_argument("--n-random", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--plans-dir", default=None, help="Plans dir for stage2.")
    p.add_argument("--out", default=None)
    p.add_argument("--n-obs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--gamma", type=float, default=1.2)
    p.add_argument("--denoising-steps", type=int, default=8)
    p.add_argument("--data-config", default="examples.Libero.custom_data_config:LiberoDataConfig")
    p.add_argument("--device", default="cuda")
    p.add_argument("--group", type=int, default=64)
    p.add_argument("--ls", type=float, default=0.15)
    p.add_argument("--act-pct", type=float, default=99.9)
    p.add_argument("--row-rot", default="restore")
    p.add_argument("--calib-steps", type=int, default=32)
    p.add_argument("--packdir", default=None)
    return p.parse_args()


def _selftest() -> None:
    rng = random.Random(0)
    shapes = {f"L{i}": {"out": 2048 + i * 128, "in": 2048, "has_bias": False} for i in range(40)}
    n_w4 = 25
    masks = gen_random_masks(shapes, n_w4, k=10, seed=1)
    assert len(masks) == 10
    for m in masks:
        assert sum(1 for v in m.values() if not v["skip"]) == n_w4
    sz = gen_size_based(shapes, n_w4)
    sz_w4 = [n for n, v in sz.items() if not v["skip"]]
    expect = sorted(shapes, key=lambda n: shapes[n]["out"] * shapes[n]["in"], reverse=True)[:n_w4]
    assert set(sz_w4) == set(expect), "size mask must pick the largest layers"
    man = gen_manual_mask(shapes, n_w4)
    assert sum(1 for v in man.values() if not v["skip"]) == n_w4
    uni4 = gen_uniform(shapes, 4)
    assert all(not v["skip"] for v in uni4.values())
    scored = [
        {"index": i, "d_solver": 0.1 + (i % 7) * 0.01, "plan": uni4} for i in range(7)
    ]
    rep = pick_representatives(scored)
    assert rep["n"] == 7 and rep["best"]["index"] == 0 and rep["worst"]["index"] == 6
    print("[baselines] selftest OK (generators + representative picking)")
    print(f"  random masks   {len(masks)} × {n_w4} W4 (same count as v2 plan)")
    print(f"  size mask      largest-{n_w4} layers ✓ | manual mask {n_w4} W4 ✓")
    print(f"  representatives best {rep['best']['d_solver']:.4f} / median {rep['median']['d_solver']:.4f} / worst {rep['worst']['d_solver']:.4f}")


def _generate(args: argparse.Namespace) -> None:
    if not args.ckpt or not args.ref_plan:
        raise SystemExit("--mode generate needs --ckpt and --ref-plan")
    shapes = read_layer_shapes(Path(args.ckpt), r".*", r"^$")
    ref = json.loads(Path(args.ref_plan).read_text())
    n_w4 = sum(1 for v in ref.get("layers", {}).values() if not v.get("skip"))
    n_layers = len(shapes)
    print(f"[baselines] v2 plan: {n_w4}/{len(ref.get('layers', {}))} W4 layers; "
          f"checkpoint shapes: {n_layers}")
    out_dir = Path(args.out_dir or str(Path(args.ref_plan).parent / f"baselines_{args.suite}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    plans: Dict[str, Any] = {}
    plans["random"] = gen_random_masks(shapes, n_w4, k=args.n_random, seed=args.seed)
    plans["size_based"] = [gen_size_based(shapes, n_w4)]
    plans["manual_v1_style"] = [gen_manual_mask(shapes, n_w4)]
    plans["uniform_w4"] = [gen_uniform(shapes, 4)]
    plans["uniform_w6"] = [gen_uniform(shapes, 6)]

    for tag, plist in plans.items():
        for i, pl in enumerate(plist):
            name = f"{tag}_{i}" if len(plist) > 1 else tag
            out = {
                "meta": {
                    "baseline": tag, "index": i,
                    "n_w4": n_w4, "n_layers": n_layers,
                    "reference_plan": args.ref_plan,
                    "note": "same W4 count as the v2 plan for fair comparison of "
                            "selection quality (random/size/manual); uniform W6 is the "
                            "same-static-byte-budget control",
                },
                "total_bytes": plan_total_bytes(pl, shapes, "restore"),
                "layers": pl,
            }
            with open(out_dir / f"{name}.json", "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
    print(f"[baselines] wrote {sum(len(v) for v in plans.values())} plans -> {out_dir}")


def _stage2(args: argparse.Namespace) -> None:
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
            / PACKDIR_TEMPLATE.format(suite=args.suite, g=args.group, calib=args.calib_steps, ls=str(args.ls).replace(".", ""))
        )
    ensure_flash_attn_rpath()
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

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

    set_quant_env(
        "", "", args.packdir, bits_default=4, group=args.group, ls=args.ls,
        act_pct=args.act_pct, calib_steps=args.calib_steps, row_rot=args.row_rot, act_dynamic=False,
    )
    policy_q = load_policy(args.model_path, data_config=args.data_config, denoising_steps=args.denoising_steps, device=args.device)

    scored = []
    for d in plans:
        t0 = time.time()
        res = score_plans(policy_q, fp_traj, [d["layers"]], obs_list, noises, args.batch_size, args.gamma)[0]
        res["file"] = d["_file"]
        res["tag"] = d.get("meta", {}).get("baseline")
        scored.append(res)
        print(f"[baselines] {d['_file']}: D_solver = {res['d_solver']:.5f} ± {res['d_solver_std']:.5f} ({time.time() - t0:.1f}s)")

    rep = pick_representatives(scored)
    out = {
        "meta": {"suite": args.suite, "n_obs": args.n_obs, "plans_dir": str(plans_dir),
                 "note": "screening stage; take best/median/worst to LIBERO (two-stage protocol, §6.5)"},
        "scored": [{k: v for k, v in s.items() if k != "plan"} for s in scored],
        "representatives": {k: ({kk: vv for kk, vv in v.items() if kk != "plan"} if v else None)
                            for k, v in rep.items()},
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
