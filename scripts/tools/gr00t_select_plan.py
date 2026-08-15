#!/usr/bin/env python3
"""GR00T v2 mixed-precision plan selector (P1-G).

Reads gr00t_sensitivity.json (P0-G output) and produces gr00t_quant_plan.json:
per-layer {bits, group, skip} under a byte budget.

Objective structure (v1.1, closes the review findings — see design doc §3.1):

    min over (b, g)   L(b, g) = Σ_i w_i · S_i(b_i, g_i)
    s.t.              Σ_i C_i(b_i, g_i) ≤ Budget

    S_i(b_i, g_i) = λ_cka·(1−ĈKA_i(b_i,g_i)) + λ_cs·D̂_CS,i(b_i,g_i)

  - S_i is the ADDITIVE layer-level proxy: CKA/CS are measured under the
    single-layer intervention protocol of the v1.2 reference (target layer at
    bit b, every other layer at weight_bits=0 inside the wrapped pipeline —
    NOT weight_bits=16, which is itself quantized), so summing over layers is
    an explicit first-order attribution, not an exact decomposition.
  - w_i = per-layer importance weight: the normalized single-layer solver
    divergence measured at one fixed probing bit (default 4), intervention vs
    the same reference; winsorized to [0.5, 2.0], uniform on Σ d = 0. It is an
    importance RANKING weight, NOT a per-bit distortion table.
  - The global solver divergence D_solver(b, g) of a COMPLETE mixed config is
    NOT additive and does NOT enter Σ_i; it is used only for the final TopK
    selection, whose rule is select_final(): lexicographic (min D_solver,
    ties within a 5% relative tolerance broken by min L_proxy). Q-DiT's lesson
    (layer MSE does not correlate with final quality) is exactly why the final
    validation is a separate stage.

Candidate semantics (unambiguous):
  - skip  = do NOT wrap the layer; it stays the original FP16 nn.Linear.
            Cost: 2·d_out·d_in (+bias) bytes. Distortion: 0 (it IS the FP16
            reference). NOT pruning — 0-bit pruning is a separate P4 option
            whose distortion must be measured (≠ 0).
  - bits ∈ {8, 6, 4, 3, 2}: DuQuant-wrapped integer quantization.
  - b=16: excluded from the search space — the wrapped-16-bit variant costs
            MORE bytes than FP16 (weights 2B + scales + rotations) while adding
            nonzero distortion, so it is strictly dominated by skip.

Budget model: static weight-storage bytes (weights + per-output-channel scales
+ R_in/R_out rotation matrices + permutation + bias), theoretical 1:1 packing
without alignment/container overhead. It is NOT peak memory, activation memory,
or BitOps; 3/6-bit entries idealize perfect bit-packing. g is the DuQuant
rotation block size (affects only the R matrices; scales stay d_out·2), NOT a
Q-DiT-style per-group quantization granularity (which would cost
d_out·ceil(d_in/g) scales — a possible P4 extension).

Usage:
    python scripts/tools/gr00t_select_plan.py \
        --sensitivity checkpoints/packs/gr00t/sensitivity_libero_spatial_g64_b2_3_4_6_8_16.json \
        --ckpt checkpoints/gr00t/libero-spatial \
        --out checkpoints/packs/gr00t/gr00t_quant_plan_libero_spatial.json \
        --emit-env
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))

BITS_ORDER = [8, 6, 4, 3, 2]  # quantization options (b=16 dominated by skip=FP16)


# --------------------------------------------------------------------------- #
# Byte accounting (mirrors scripts/tools/calc_gr00t_duquant_memory.py)
# --------------------------------------------------------------------------- #
def layer_bytes_quant(out: int, inn: int, has_bias: bool, bits: int, group: int,
                      row_rot: str = "restore", permute: bool = False) -> float:
    mem = out * inn * bits / 8.0  # 理论紧密打包（3/6-bit 理想化，无对齐/容器开销）
    mem += out * 2.0  # FP16 per-output-channel weight scales（与 group 无关）
    n_in = math.ceil(inn / group)
    mem += n_in * group * group * 4.0  # R_in 旋转块（g 只影响这里）
    if row_rot in ("restore", "propagate"):
        n_out = math.ceil(out / group)
        mem += n_out * group * group * 4.0  # R_out
    if permute:
        mem += inn * 4.0
    if has_bias:
        mem += out * 2.0
    return mem


def layer_bytes_fp16(out: int, inn: int, has_bias: bool) -> float:
    return out * inn * 2.0 + (out * 2.0 if has_bias else 0.0)


# --------------------------------------------------------------------------- #
# Checkpoint shapes
# --------------------------------------------------------------------------- #
def read_layer_shapes(ckpt: Path, include: str, exclude: str) -> Dict[str, Dict[str, Any]]:
    """Read 2D linear-layer shapes from safetensors (multi-file aware)."""
    files: List[Path] = []
    if ckpt.is_dir():
        files = sorted(ckpt.glob("model*.safetensors"))
        single = ckpt / "model.safetensors"
        if single.exists():
            files = [single] + [f for f in files if f.name != "model.safetensors"]
    else:
        files = [ckpt]
    if not files:
        raise FileNotFoundError(f"No safetensors found under {ckpt}")

    inc = re.compile(include)
    exc = re.compile(exclude)
    shapes: Dict[str, Dict[str, Any]] = {}
    try:
        from safetensors import safe_open
    except ImportError:
        raise SystemExit("safetensors is required (groot_test env has it)")

    for f in files:
        with safe_open(str(f), framework="pt") as sf:
            for key in sf.keys():
                if not key.endswith(".weight"):
                    continue
                name = key[:-7]
                tensor = sf.get_tensor(key)
                if tensor.ndim != 2:
                    continue
                if inc.search(name) and not exc.search(name):
                    shapes[name] = {
                        "out": tensor.shape[0],
                        "in": tensor.shape[1],
                        "has_bias": f"{name}.bias" in sf.keys(),
                    }
    return shapes


# --------------------------------------------------------------------------- #
# Score fusion (joint min-max over all (layer, bit) pairs per term)
# --------------------------------------------------------------------------- #
def _joint_norm(values: Dict[Tuple[str, int], Optional[float]]) -> Dict[Tuple[str, int], float]:
    """Joint min-max over ALL (layer, bit) pairs — keeps ΔS/Δbyte ratios
    comparable across layers AND across bit levels. Constant input -> zeros."""
    present = {k: v for k, v in values.items() if v is not None}
    if not present:
        return {}
    lo, hi = min(present.values()), max(present.values())
    if hi - lo < 1e-12:
        return {k: 0.0 for k in present}
    return {k: (v - lo) / (hi - lo) for k, v in present.items()}


def build_scores(
    sensitivity: Dict[str, Any],
    layer_names: List[str],
    lambda_cka: float,
    lambda_cs: float,
) -> Dict[str, Dict[int, float]]:
    """Additive layer proxy S_i(b) = λ_cka·norm(1−cka) + λ_cs·norm(cs).

    Normalization is joint min-max per term across all (layer, bit) pairs.
    """
    lay = sensitivity.get("layers", {})
    cka_loss: Dict[Tuple[str, int], Optional[float]] = {}
    cs_vals: Dict[Tuple[str, int], Optional[float]] = {}
    for n in layer_names:
        for b in BITS_ORDER:
            entry = lay.get(n, {}).get(f"b{b}", {})
            cka = entry.get("cka")
            cs = entry.get("cs")
            cka_loss[(n, b)] = (1.0 - cka) if cka is not None else None
            cs_vals[(n, b)] = cs if cs is not None else None
    cka_n = _joint_norm(cka_loss)
    cs_n = _joint_norm(cs_vals)

    scores: Dict[str, Dict[int, float]] = {}
    for n in layer_names:
        for b in BITS_ORDER:
            s = 0.0
            w_sum = 0.0
            if (n, b) in cka_n:
                s += lambda_cka * cka_n[(n, b)]
                w_sum += lambda_cka
            if (n, b) in cs_n:
                s += lambda_cs * cs_n[(n, b)]
                w_sum += lambda_cs
            scores.setdefault(n, {})[b] = s / w_sum if w_sum > 0 else 0.0
    return scores


def build_weights(sensitivity: Dict[str, Any], layer_names: List[str]) -> Dict[str, float]:
    """Layer importance weights from the single-layer solver divergence measured
    at one fixed probing bit (default b=4, intervention-vs-reference). Ranking
    weight only — NOT a per-bit distortion.

    Definition (v1.2, boundary cases closed):
      w_i = n·d_i / Σ_j d_j, with
      - Σ_j d_j ≤ 0  -> all w_i = 1 (uniform);
      - layers without a measurement get the mean d;
      - winsorize to [0.5, 2.0] after normalization so a single extreme layer
        cannot zero out every other weight (n_rollout_obs=4 is a small sample;
        weight variance is recorded and sample size should grow in P4)."""
    lay = sensitivity.get("layers", {})
    raw: Dict[str, float] = {}
    for n in layer_names:
        d = None
        for key, val in lay.get(n, {}).items():
            if key.startswith("d_solver_b") and not key.endswith("_std"):
                d = float(val)
            elif key.startswith("d_action_b") and not key.endswith("_std"):
                d = float(val)
        if d is not None:
            raw[n] = d
    if not raw:
        return {n: 1.0 for n in layer_names}
    mean_d = sum(raw.values()) / len(raw)
    filled = {n: raw.get(n, mean_d) for n in layer_names}
    s = sum(filled.values())
    if s <= 0:
        # boundary: all divergences zero (or negative) -> uniform weights
        return {n: 1.0 for n in layer_names}
    w = {n: v / s * len(layer_names) for n, v in filled.items()}
    # outlier protection: winsorize to [0.5, 2.0] so a single extreme layer
    # (e.g. tiny ||x_fp|| denominator) cannot zero out every other weight.
    # NOTE: n_rollout_obs=4 is a small sample — weight variance is recorded in
    # the plan (weight_std) and the sample size should grow in P4.
    return {n: min(2.0, max(0.5, v)) for n, v in w.items()}


def select_final(topk: List[Dict[str, Any]], tol: float = 0.05) -> Optional[Dict[str, Any]]:
    """TopK final-selection rule (v1.2, lexicographic — now well-defined).

    Among candidate configs, each entry must carry "d_solver" (config-level
    global divergence, measured on GPU by paired rollout of the COMPLETE
    config) and "proxy" (the search objective Σ w_i·S_i).

        d_min  = min_c D_solver(c)
        T      = { c : D_solver(c) ≤ d_min + tol·max(d_min, ε) }   (tie set)
        c*     = argmin_{c ∈ T} L_proxy(c)

    Rationale: D_solver is the task-level metric (Q-DiT: proxy metrics do not
    correlate with final quality); the proxy only breaks near-ties. tol is a
    relative tolerance (default 5%).
    """
    cands = [c for c in topk if c.get("d_solver") is not None]
    if not cands:
        return None
    d_min = min(float(c["d_solver"]) for c in cands)
    ties = [
        c for c in cands
        if float(c["d_solver"]) <= d_min + tol * max(d_min, 1e-12)
    ]
    return min(ties, key=lambda c: float(c.get("proxy", float("inf"))))


# --------------------------------------------------------------------------- #
# Solvers (objective = Σ_i w_i·S_i(b_i); skip = 0 distortion, FP16 cost)
# --------------------------------------------------------------------------- #
def plan_total_bytes(plan: Dict[str, Any], shapes: Dict[str, Dict[str, Any]], row_rot: str) -> float:
    total = sum(layer_bytes_fp16(s["out"], s["in"], s["has_bias"]) for s in shapes.values())
    for name, s in shapes.items():
        entry = plan.get(name)
        if not entry or entry.get("skip"):
            continue
        b = int(entry["bits"])
        g = int(entry.get("group", 64))
        total -= layer_bytes_fp16(s["out"], s["in"], s["has_bias"]) - layer_bytes_quant(
            s["out"], s["in"], s["has_bias"], b, g, row_rot=row_rot
        )
    return total


def plan_objective(plan: Dict[str, Any], scores: Dict[str, Dict[int, float]],
                   weights: Dict[str, float]) -> float:
    return sum(
        weights[n] * (scores[n].get(plan[n]["bits"], 0.0) if plan[n]["bits"] is not None else 0.0)
        for n in plan
    )


def greedy_plan(
    shapes: Dict[str, Dict[str, Any]],
    scores: Dict[str, Dict[int, float]],
    weights: Dict[str, float],
    budget: float,
    row_rot: str,
) -> Tuple[Dict[str, Any], float]:
    # state: bits=None -> skip (FP16, cost 2P, distortion 0) ; else quantized bit
    plan: Dict[str, Any] = {n: {"bits": None, "group": 64, "skip": True} for n in shapes}

    def bytes_of(name: str) -> float:
        s = shapes[name]
        b = plan[name]["bits"]
        if b is None:
            return layer_bytes_fp16(s["out"], s["in"], s["has_bias"])
        return layer_bytes_quant(s["out"], s["in"], s["has_bias"], b, plan[name]["group"], row_rot=row_rot)

    def score_of(name: str) -> float:
        b = plan[name]["bits"]
        return 0.0 if b is None else scores[name].get(b, 0.0)

    while plan_total_bytes(plan, shapes, row_rot) > budget:
        best: Optional[Tuple[float, str, int]] = None
        for name in shapes:
            cur = plan[name]["bits"]
            if cur is None:
                nxt = BITS_ORDER[0]  # skip(FP16) -> W8
            else:
                idx = BITS_ORDER.index(cur)
                if idx == len(BITS_ORDER) - 1:
                    continue  # already W2
                nxt = BITS_ORDER[idx + 1]
            if scores[name].get(nxt) is None:
                continue  # no score at the target bit -> leave untouched
            delta_bytes = bytes_of(name) - layer_bytes_quant(
                shapes[name]["out"], shapes[name]["in"], shapes[name]["has_bias"],
                nxt, plan[name]["group"], row_rot=row_rot,
            )
            if delta_bytes <= 0:
                continue
            delta_score = weights[name] * (scores[name][nxt] - score_of(name))
            ratio = delta_score / delta_bytes
            if best is None or ratio < best[0]:
                best = (ratio, name, nxt)
        if best is None:
            break
        _, name, nxt = best
        plan[name]["bits"] = nxt
        plan[name]["skip"] = False
    return plan, plan_objective(plan, scores, weights)


def evolution_plan(
    shapes: Dict[str, Dict[str, Any]],
    scores: Dict[str, Dict[int, float]],
    weights: Dict[str, float],
    budget: float,
    row_rot: str,
    npop: int,
    niter: int,
    topk: int,
    seed: int = 0,
) -> Tuple[Dict[str, Any], float]:
    import random

    rng = random.Random(seed)
    names = list(shapes)
    choices = [None] + BITS_ORDER  # None = skip (FP16)

    def random_ind() -> Dict[str, Any]:
        ind = {n: {"bits": None, "group": 64, "skip": True} for n in shapes}
        for n in names:
            if rng.random() < 0.9:
                b = rng.choice(BITS_ORDER)
                ind[n]["bits"] = b
                ind[n]["skip"] = False
        return ind

    def fitness(ind: Dict[str, Any]) -> float:
        total = plan_total_bytes(ind, shapes, row_rot)
        obj = plan_objective(ind, scores, weights)
        penalty = max(0.0, total - budget) / budget * 10.0
        return obj + penalty

    pop = [random_ind() for _ in range(npop)]
    for _ in range(niter):
        pop.sort(key=fitness)
        pop = pop[:topk]
        new_pop: List[Dict[str, Any]] = []
        while len(new_pop) < npop:
            a, b = rng.choice(pop), rng.choice(pop)
            child = {}
            for n in names:
                src = a if rng.random() < 0.5 else b
                child[n] = dict(src[n])
            if rng.random() < 0.2:
                n = rng.choice(names)
                c = rng.choice(choices)
                child[n]["bits"] = c
                child[n]["skip"] = c is None
            new_pop.append(child)
        pop = pop + new_pop
    pop.sort(key=fitness)
    best = pop[0]

    # project the winner back into the feasible region (downgrade moves only)
    while plan_total_bytes(best, shapes, row_rot) > budget:
        cand: Optional[Tuple[float, str, int]] = None
        for n in names:
            cur = best[n]["bits"]
            if cur is None:
                nxt = BITS_ORDER[0]
            else:
                idx = BITS_ORDER.index(cur)
                if idx == len(BITS_ORDER) - 1:
                    continue
                nxt = BITS_ORDER[idx + 1]
            if scores[n].get(nxt) is None:
                continue
            s = shapes[n]
            delta_bytes = (
                (layer_bytes_fp16(s["out"], s["in"], s["has_bias"]) if cur is None
                 else layer_bytes_quant(s["out"], s["in"], s["has_bias"], cur, best[n]["group"], row_rot=row_rot))
                - layer_bytes_quant(s["out"], s["in"], s["has_bias"], nxt, best[n]["group"], row_rot=row_rot)
            )
            if delta_bytes <= 0:
                continue
            ratio = (weights[n] * (scores[n][nxt] - (0.0 if cur is None else scores[n].get(cur, 0.0)))) / delta_bytes
            if cand is None or ratio < cand[0]:
                cand = (ratio, n, nxt)
        if cand is None:
            break
        _, n, nxt = cand
        best[n]["bits"] = nxt
        best[n]["skip"] = False

    return best, plan_objective(best, scores, weights)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GR00T v2 mixed-precision plan selector (P1-G)")
    p.add_argument("--sensitivity", required=True, help="gr00t_sensitivity.json from P0-G.")
    p.add_argument("--ckpt", required=True, help="Checkpoint dir (or single safetensors).")
    p.add_argument("--out", required=True, help="Output plan JSON path.")
    p.add_argument("--include", default=r".*(backbone\.eagle_model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|action_head\.model\.transformer_blocks\.\d+\.ff\.net\.(0\.proj|2)).*")
    p.add_argument("--exclude", default=r"(?:^|\.)(vision|radio|norm|ln|layernorm|embed|lm_head|attn1)(?:\.|$)")
    p.add_argument("--lambda-cka", type=float, default=1.0, help="Weight of the (1-CKA) proxy term.")
    p.add_argument("--lambda-cs", type=float, default=1.0, help="Weight of the CS-divergence proxy term.")
    p.add_argument("--group", type=int, default=64, help="DuQuant rotation block size (all plan layers; multi-block is P4).")
    p.add_argument("--row-rot", default="restore")
    p.add_argument("--budget", default="auto", help="Static byte budget or 'auto' (= v1 W4A8 plan bytes).")
    p.add_argument("--min-bits", type=int, default=4,
                   help="最低 bit 档（默认 4）：v1.2 实测发现 CKA/CS 对 W2/W3 的"
                        "输出幅度爆炸失明（几何保持但幅度放大数倍 → 下游 A8 饱和 → "
                        "成功率崩溃），故默认把搜索空间限制在 v1 验证过的区间。"
                        "设 2 可恢复全搜索空间（实验用）。")
    p.add_argument("--solver", default="greedy", choices=["greedy", "evolution"])
    p.add_argument("--npop", type=int, default=20)
    p.add_argument("--niter", type=int, default=10)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--packdir", default=None, help="DuQuant pack dir recorded in the plan.")
    p.add_argument("--emit-env", action="store_true", help="Print export lines for run_quantvla.sh.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 搜索空间按 --min-bits 收紧（见参数说明：CKA/CS 对低 bit 的幅度失真失明）
    BITS_ORDER[:] = [b for b in BITS_ORDER if b >= args.min_bits]
    if not BITS_ORDER:
        raise SystemExit(f"--min-bits {args.min_bits} 过滤后无可用 bit 档")

    with open(args.sensitivity, "r", encoding="utf-8") as f:
        sens = json.load(f)

    shapes = read_layer_shapes(Path(args.ckpt), args.include, args.exclude)
    print(f"[select] layers with shapes: {len(shapes)}")

    # keep only layers present in the sensitivity file
    sens_names = set(sens.get("layers", {}).keys())
    shapes = {n: s for n, s in shapes.items() if n in sens_names}
    print(f"[select] layers in sensitivity ∩ checkpoint: {len(shapes)}")

    scores = build_scores(sens, list(shapes), args.lambda_cka, args.lambda_cs)
    weights = build_weights(sens, list(shapes))

    if args.budget == "auto":
        # v1 reference: every target layer at W4 g=64, everything else FP16
        v1 = {n: {"bits": 4, "group": args.group, "skip": False} for n in shapes}
        budget = plan_total_bytes(v1, shapes, args.row_rot)
        print(f"[select] budget (v1 W4A8 static-byte reference): {budget / 1e6:.1f} MB")
    else:
        budget = float(args.budget)

    if args.solver == "greedy":
        plan, obj = greedy_plan(shapes, scores, weights, budget, args.row_rot)
    else:
        plan, obj = evolution_plan(shapes, scores, weights, budget, args.row_rot, args.npop, args.niter, args.topk)

    total = plan_total_bytes(plan, shapes, args.row_rot)
    n_quant = sum(1 for v in plan.values() if not v["skip"])
    print(f"[select] plan: {n_quant}/{len(shapes)} layers quantized, "
          f"bytes {total / 1e6:.1f} MB (budget {budget / 1e6:.1f} MB), proxy objective {obj:.4f}")

    out_plan: Dict[str, Any] = {
        "meta": {
            "sensitivity": args.sensitivity,
            "ckpt": args.ckpt,
            "solver": args.solver,
            "lambda": {"cka": args.lambda_cka, "cs": args.lambda_cs},
            "row_rot": args.row_rot,
            "objective": "Σ w_i·S_i(b_i) — 层代理目标（一阶归因）；全局 D_solver 需对完整配置做 GPU 配对 rollout 裁决",
            "budget_semantics": "静态权重存储字节（理论紧密打包；不含激活/峰值显存/时延/BitOps）",
            "skip_semantics": "不量化、保留 FP16（成本 2·d_out·d_in+bias 字节，失真 0）；0-bit 剪枝为 P4 独立选项",
        },
        "budget_bytes": budget,
        "total_bytes": total,
        "objective": obj,
        "packdirs": {str(args.group): args.packdir} if args.packdir else {},
        "layers": {},
    }
    for n, s in shapes.items():
        entry = plan[n]
        lay = sens.get("layers", {}).get(n, {})
        key = f"b{entry['bits']}" if entry["bits"] is not None else None
        d_ref = None
        d_ref_std = None
        for k2, v2 in lay.items():
            if k2.startswith("d_solver_b") and k2.endswith("_std"):
                d_ref_std = float(v2)
            elif k2.startswith("d_solver_b") or k2.startswith("d_action_b"):
                d_ref = float(v2)
        out_plan["layers"][n] = {
            "bits": entry["bits"],
            "group": entry["group"],
            "skip": entry["skip"],
            "score": scores[n].get(entry["bits"], 0.0) if entry["bits"] is not None else 0.0,
            "weight": weights[n],
            "weight_std": d_ref_std,
            "cka": lay.get(key, {}).get("cka") if key else None,
            "cs": lay.get(key, {}).get("cs") if key else None,
            "d_solver_ref": d_ref,
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_plan, f, indent=2)
    print(f"[select] saved plan -> {args.out}")
    print("[select] NEXT: GPU 侧对 TopK 完整配置跑配对 rollout 得到全局 D_solver，")
    print("[select]       按 select_final() 规则裁决（字典序：min D_solver，")
    print("[select]       相对差 ≤5% 时以 proxy 目标打破平局），再按 LIBERO 协议验收")

    if args.emit_env:
        print("\n# --- run_quantvla.sh usage ---")
        print(f"export GR00T_DUQUANT_PLAN={args.out}")
        if args.packdir:
            print(f"export GR00T_DUQUANT_PACKDIR={args.packdir}")
        print("# ./scripts/run_quantvla.sh libero_<suite>")


if __name__ == "__main__":
    main()
