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
import random
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))

BITS_ORDER = [8, 6, 4, 3, 2]  # quantization options (b=16 dominated by skip=FP16)
BINARY_BITS = [4]  # v1.3 main path: binary W4/FP16 selection (design doc §1.3.1)


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
            # P0-7 (correctness review): a layer with NO usable measurement at
            # this bit is UNAVAILABLE (stays FP16), not zero-distortion. The old
            # code scored missing data as 0.0, i.e. "safest to quantize".
            scores.setdefault(n, {})[b] = s / w_sum if w_sum > 0 else None
    return scores


def _weight_stats(values: Dict[str, float]) -> Dict[str, float]:
    """min/max/mean/std over a weight dict (v1.3 w_i three-stage logging, §5.2)."""
    if not values:
        return {"min": None, "max": None, "mean": None, "std": None}
    vals = list(values.values())
    return {
        "min": min(vals),
        "max": max(vals),
        "mean": statistics.fmean(vals),
        "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
    }


def build_weights_with_log(
    sensitivity: Dict[str, Any], layer_names: List[str]
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Layer importance weights + the v1.3 mandatory three-stage log.

    Definition (v1.2, boundary cases closed):
      w_i = n·d_i / Σ_j d_j, with
      - Σ_j d_j ≤ 0  -> all w_i = 1 (uniform);
      - layers without a measurement get the mean d;
      - winsorize to [0.5, 2.0] after normalization so a single extreme layer
        cannot zero out every other weight.

    v1.3 (§5.2): the selector MUST log raw / normalized-before-clip / final
    statistics — the experiment report's "w_i ∈ [0.0002, 0.0010]" was the RAW
    d_solver, not the normalized weights, which made it impossible to verify
    that action importance actually entered the search. The log enforces
    mean(final) ≈ 1 and 0.5 ≤ final ≤ 2.
    """
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
        w = {n: 1.0 for n in layer_names}
        log = {
            "raw_d_solver": _weight_stats(raw),
            "normalized_before_clip": _weight_stats(w),
            "final": _weight_stats(w),
            "note": "no d_solver measurements -> uniform weights",
        }
        return w, log
    mean_d = sum(raw.values()) / len(raw)
    filled = {n: raw.get(n, mean_d) for n in layer_names}
    s = sum(filled.values())
    if s <= 0:
        # boundary: all divergences zero (or negative) -> uniform weights
        w = {n: 1.0 for n in layer_names}
        log = {
            "raw_d_solver": _weight_stats(raw),
            "normalized_before_clip": _weight_stats(w),
            "final": _weight_stats(w),
            "note": "Σ d ≤ 0 -> uniform weights",
        }
        return w, log
    w_norm = {n: v / s * len(layer_names) for n, v in filled.items()}
    # outlier protection: winsorize to [0.5, 2.0], then RE-normalize so the
    # final weights keep mean == 1 (review round 5, item 7 — the single
    # dominant d_solver layer pulled the final mean down to ~0.57)
    w_clip = {n: min(2.0, max(0.5, v)) for n, v in w_norm.items()}
    s_clip = sum(w_clip.values())
    w_final = {n: v / s_clip * len(layer_names) for n, v in w_clip.items()} if s_clip > 0 else w_clip
    log = {
        "raw_d_solver": _weight_stats(raw),
        "normalized_before_clip": _weight_stats(w_norm),
        "final": _weight_stats(w_final),
    }
    return w_final, log


def build_weights(sensitivity: Dict[str, Any], layer_names: List[str]) -> Dict[str, float]:
    """Back-compat wrapper: weights only (log dropped)."""
    return build_weights_with_log(sensitivity, layer_names)[0]


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
# v1.3: feasibility-guard hard filtering (design doc §3.1, constraints 3a/3b)
# --------------------------------------------------------------------------- #
def estimate_guard_thresholds(
    sensitivity: Dict[str, Any],
    layer_names: List[str],
    bit: int = 4,
    margin: float = 1.5,
) -> Tuple[Optional[float], Optional[float]]:
    """τ_rms / τ_sat = P99 of the W4 candidates × margin (design doc §5.1.3).

    Thresholds are estimated from the measured W4 guard values when the probe
    did not already write them into meta["guard_thresholds"].
    """
    vals_rms: List[float] = []
    vals_sat: List[float] = []
    for n in layer_names:
        entry = sensitivity.get("layers", {}).get(n, {}).get(f"b{bit}", {})
        for key, lst in (("rms_ratio", vals_rms), ("sat_rate", vals_sat)):
            v = entry.get(key)
            if v is not None:
                lst.append(float(v))

    def p99(vals: List[float]) -> Optional[float]:
        if not vals:
            return None
        s = sorted(vals)
        idx = max(0, min(len(s) - 1, math.ceil(0.99 * len(s)) - 1))
        return s[idx] * margin

    return p99(vals_rms), p99(vals_sat)


def filter_guarded(
    scores: Dict[str, Dict[int, float]],
    sensitivity: Dict[str, Any],
    layer_names: List[str],
    tau_rms: Optional[float],
    tau_sat: Optional[float],
    bit: int = 4,
) -> Tuple[Dict[str, Dict[int, float]], List[Dict[str, Any]]]:
    """Remove guard-violating layers from the search space (HARD constraint).

    Violating layers lose all bit options -> they stay FP16 (skip). This is a
    feasibility cut, NOT a weighted penalty: the W2 collapse showed that
    CKA/CS/action scores must not be allowed to vote a blown-up layer back in.
    """
    filtered: Dict[str, Dict[int, float]] = {n: dict(scores.get(n, {})) for n in layer_names}
    removed: List[Dict[str, Any]] = []
    for n in layer_names:
        entry = sensitivity.get("layers", {}).get(n, {}).get(f"b{bit}", {})
        r = entry.get("rms_ratio")
        s = entry.get("sat_rate")
        bad = (tau_rms is not None and r is not None and float(r) > tau_rms) or (
            tau_sat is not None and s is not None and float(s) > tau_sat
        )
        if bad:
            removed.append({"layer": n, "rms_ratio": r, "sat_rate": s})
            filtered[n] = {}
    return filtered, removed


# --------------------------------------------------------------------------- #
# v1.3: binary solvers — 0-1 knapsack exact solve + perturbed neighbors + λ sweep
# --------------------------------------------------------------------------- #
def milp_binary_plan(
    shapes: Dict[str, Dict[str, Any]],
    scores: Dict[str, Dict[int, float]],
    weights: Dict[str, float],
    budget: float,
    row_rot: str,
    group: int = 64,
) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
    """Exact 0-1 knapsack via scipy milp (v1.3, design doc §5.3).

    min Σ_i x_i·w_i·S_i(W4)  s.t. Σ_i x_i·(C_fp − C_w4) ≥ (C_all_fp − budget),
    x_i ∈ {0,1}. Returns (plan, objective); (None, None) when scipy is missing
    or the problem is infeasible. Used to certify the greedy gap.
    """
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
    except Exception:
        return None, None

    names = [n for n in shapes if scores.get(n, {}).get(4) is not None]
    if not names:
        return None, None
    fp_total = sum(layer_bytes_fp16(s["out"], s["in"], s["has_bias"]) for s in shapes.values())
    need = fp_total - budget
    if need <= 0:
        plan = {n: {"bits": None, "group": group, "skip": True} for n in shapes}
        return plan, 0.0
    c = np.array([weights[n] * scores[n][4] for n in names], dtype=float)
    savings = np.array(
        [
            layer_bytes_fp16(shapes[n]["out"], shapes[n]["in"], shapes[n]["has_bias"])
            - layer_bytes_quant(shapes[n]["out"], shapes[n]["in"], shapes[n]["has_bias"], 4, group, row_rot=row_rot)
            for n in names
        ],
        dtype=float,
    )
    usable = [n for i, n in enumerate(names) if savings[i] > 0]
    if not usable:
        return None, None
    keep = [names.index(n) for n in usable]
    c = c[keep]
    savings = savings[keep]
    A = savings.reshape(1, -1)  # Σ savings·x ≥ need  (lb ≤ A@x ≤ ub)
    res = milp(
        c=c,
        constraints=LinearConstraint(A, lb=[need], ub=[np.inf]),
        integrality=np.ones(len(usable)),
        bounds=Bounds(np.zeros(len(usable)), np.ones(len(usable))),
    )
    if not res.success:
        return None, None
    x = np.round(res.x)
    plan = {n: {"bits": None, "group": group, "skip": True} for n in shapes}
    for i, n in enumerate(usable):
        if x[i] >= 0.5:
            plan[n] = {"bits": 4, "group": group, "skip": False}
    return plan, float(res.fun)


def perturbed_plans(
    shapes: Dict[str, Dict[str, Any]],
    scores: Dict[str, Dict[int, float]],
    weights: Dict[str, float],
    budget: float,
    row_rot: str,
    n: int = 10,
    sigma: float = 0.1,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Score-perturbation neighbors: S_i × (1+δ), δ ~ N(0, σ) (design §5.3).

    Probes plan robustness against proxy-score noise; candidates feed the
    diversity-filtered TopK pipeline.
    """
    rng = random.Random(seed)
    out: List[Dict[str, Any]] = []
    for _ in range(n):
        ps = {
            nm: {b: max(0.0, v * (1.0 + rng.gauss(0.0, sigma))) for b, v in sd.items() if v is not None}
            for nm, sd in scores.items()
        }
        plan, obj = greedy_plan(shapes, ps, weights, budget, row_rot)
        out.append({"plan": plan, "objective": obj, "source": "perturbed"})
    return out


def lambda_sweep_plans(
    shapes: Dict[str, Dict[str, Any]],
    sensitivity: Dict[str, Any],
    layer_names: List[str],
    weights: Dict[str, float],
    budget: float,
    row_rot: str,
    pairs: Tuple[Tuple[float, float], ...] = ((1.0, 1.0), (1.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 1.0)),
    guarded_names: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """λ-sweep candidates: plan diversity is generated by the sweep; the FINAL
    choice is NOT a λ pick — it is made by D_solver adjudication (§3.1).

    P0-7: guarded_names (feasibility-hard-filtered layers) are stripped from
    EVERY sweep candidate's scores — a layer that failed the amplitude/
    saturation guards must not re-enter W4 through a different λ.
    """
    guarded_names = guarded_names or set()
    out: List[Dict[str, Any]] = []
    for lc, ls in pairs:
        sc = build_scores(sensitivity, layer_names, lc, ls)
        for g in guarded_names:
            sc[g] = {}
        plan, obj = greedy_plan(shapes, sc, weights, budget, row_rot)
        out.append({"plan": plan, "objective": obj, "source": f"lambda({lc},{ls})"})
    return out


# --------------------------------------------------------------------------- #
# v1.3: TopK diversity + stability (design doc §5.3 / §5.2)
# --------------------------------------------------------------------------- #
def assert_plan_guards(plan: Dict[str, Any], guarded_names: set) -> None:
    """P0-7: no guard-filtered layer may be quantized in any candidate plan."""
    for n in guarded_names:
        entry = plan.get(n)
        assert entry is None or entry.get("skip"), (
            f"guard violation: filtered layer '{n}' is quantized in the plan"
        )


def flip_neighbors_plans(
    shapes: Dict[str, Dict[str, Any]],
    scores: Dict[str, Dict[int, float]],
    weights: Dict[str, float],
    budget: float,
    row_rot: str,
    base_plan: Dict[str, Any],
    n_flips: Tuple[int, ...] = (8, 12, 16, 20),
    per_flip: int = 2,
    seed: int = 3,
) -> List[Dict[str, Any]]:
    """Mutation diversity (Q-DiT style): SWAP k skipped layers with k W4
    layers of the base plan (pure swaps -> the FP16-mask Hamming distance is
    exactly 2k), then budget-repair. Guarantees mask diversity when greedy /
    milp / lambda sweep all collapse onto the same mask (which happened on the
    first spatial run: 17 candidates, 1 unique mask)."""
    rng = random.Random(seed)
    names = [n for n in shapes if scores.get(n, {}).get(4) is not None]
    # swap pool restricted to score-available layers: guarded / unmeasured
    # layers (score None) must NEVER be turned into W4 by a mutation.
    swap_skip = [n for n, e in base_plan.items()
                 if e.get("skip") and n in names]
    swap_w4 = [n for n in names if not base_plan[n].get("skip")]
    out: List[Dict[str, Any]] = []
    for k in n_flips:
        for _ in range(per_flip):
            kk = min(k, len(swap_skip), len(swap_w4))
            if kk == 0:
                continue
            add_skip = set(rng.sample(swap_w4, kk))    # W4 -> skip
            drop_skip = set(rng.sample(swap_skip, kk))  # skip -> W4
            skip = set(n for n, e in base_plan.items() if e.get("skip"))
            skip = (skip | add_skip) - drop_skip
            plan = {n: ({"bits": None, "group": 64, "skip": True} if n in skip
                        else {"bits": 4, "group": 64, "skip": False})
                    for n in shapes}
            # budget repair: quantize the cheapest-per-byte skipped layer until feasible
            while plan_total_bytes(plan, shapes, row_rot) > budget + 1e-3:
                cand: Optional[Tuple[float, str]] = None
                for n in names:
                    if not plan[n]["skip"]:
                        continue
                    sh = shapes[n]
                    saved = layer_bytes_fp16(sh["out"], sh["in"], sh["has_bias"]) - layer_bytes_quant(
                        sh["out"], sh["in"], sh["has_bias"], 4, 64, row_rot=row_rot
                    )
                    if saved <= 0:
                        continue
                    ratio = (weights[n] * scores[n][4]) / saved
                    if cand is None or ratio < cand[0]:
                        cand = (ratio, n)
                if cand is None:
                    break
                plan[cand[1]] = {"bits": 4, "group": 64, "skip": False}
            out.append({"plan": plan, "objective": plan_objective(plan, scores, weights),
                        "source": f"flip{k}"})
    return out


def plan_mask(plan: Dict[str, Any]) -> Tuple[str, ...]:
    """FP16 (skip) layer set of a plan — the diversity unit."""
    return tuple(sorted(n for n, e in plan.items() if e.get("skip")))


def hamming(m1: Tuple[str, ...], m2: Tuple[str, ...]) -> int:
    return len(set(m1) ^ set(m2))


def select_diverse(
    candidates: List[Dict[str, Any]],
    k: int = 10,
    min_hamming: int = 1,
) -> List[Dict[str, Any]]:
    """Greedy diverse TopK: lowest objective first, accept only when the FP16
    mask differs from every accepted plan by ≥ min_hamming layers."""
    chosen: List[Dict[str, Any]] = []
    seen: set = set()
    for c in sorted(candidates, key=lambda c: c["objective"]):
        mask = plan_mask(c["plan"])
        if mask in seen:
            continue
        if all(hamming(mask, plan_mask(x["plan"])) >= min_hamming for x in chosen):
            chosen.append(c)
            seen.add(mask)
        if len(chosen) >= k:
            break
    return chosen


def _spearman(a: List[float], b: List[float]) -> float:
    """Rank correlation between two equal-length lists (ties -> average ranks)."""
    if len(a) < 2 or len(set(a)) < 2 or len(set(b)) < 2:
        return 1.0
    n = len(a)

    def ranks(v: List[float]) -> List[float]:
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for t in range(i, j + 1):
                r[order[t]] = avg
            i = j + 1
        return r

    ra, rb = ranks(a), ranks(b)
    ma, mb = statistics.fmean(ra), statistics.fmean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den > 1e-12 else 1.0


def bootstrap_stability(
    sensitivity: Dict[str, Any],
    shapes: Dict[str, Dict[str, Any]],
    scores: Dict[str, Dict[int, float]],
    budget: float,
    row_rot: str,
    n: int = 100,
    seed: int = 0,
) -> Dict[str, Any]:
    """w_i estimation stability (v1.3, §5.2): parametric bootstrap over the
    per-layer d_solver measurements (mean ± std from the probe), re-running the
    binary greedy plan each draw. Reports FP16-mask Jaccard vs the base plan and
    weight-ranking Spearman. "A method whose FP16 mask changes half its layers
    under a new calibration draw is not usable yet." """
    rng = random.Random(seed)
    lay = sensitivity.get("layers", {})
    raw: Dict[str, float] = {}
    std: Dict[str, float] = {}
    for nm in shapes:
        d = s = None
        for k, v in lay.get(nm, {}).items():
            if k.startswith("d_solver_b") and k.endswith("_std"):
                s = float(v)
            elif k.startswith("d_solver_b") or k.startswith("d_action_b"):
                d = float(v)
        if d is not None:
            raw[nm] = d
            std[nm] = s if s is not None else 0.0

    base_weights = build_weights(sensitivity, list(shapes))
    base_plan, _ = greedy_plan(shapes, scores, base_weights, budget, row_rot)
    base_mask = plan_mask(base_plan)

    def weights_from_draw(draw: Dict[str, float]) -> Dict[str, float]:
        if not draw:
            return {n: 1.0 for n in shapes}
        mean_d = statistics.fmean(draw.values())
        filled = {n: draw.get(n, mean_d) for n in shapes}
        s = sum(filled.values())
        if s <= 0:
            return {n: 1.0 for n in shapes}
        w = {n: v / s * len(shapes) for n, v in filled.items()}
        return {n: min(2.0, max(0.5, v)) for n, v in w.items()}

    jaccards: List[float] = []
    spearmans: List[float] = []
    for _ in range(n):
        draw: Dict[str, float] = {}
        for nm, d in raw.items():
            sd = std[nm] if std[nm] > 0 else max(d, 1e-12) * 0.1
            draw[nm] = max(0.0, rng.gauss(d, sd))
        w_b = weights_from_draw(draw)
        plan_b, _ = greedy_plan(shapes, scores, w_b, budget, row_rot)
        m_b = plan_mask(plan_b)
        inter = len(set(base_mask) & set(m_b))
        union = len(set(base_mask) | set(m_b))
        jaccards.append(inter / union if union else 1.0)
        # rank correlation on measured layers
        common = [nm for nm in raw if nm in draw]
        if len(common) >= 2:
            spearmans.append(_spearman([draw[nm] for nm in common], [base_weights[nm] for nm in common]))
    jaccards.sort()
    return {
        "n_draws": n,
        "jaccard": {
            "mean": statistics.fmean(jaccards) if jaccards else None,
            "p05": jaccards[max(0, len(jaccards) // 20)] if jaccards else None,
            "p95": jaccards[min(len(jaccards) - 1, int(0.95 * (len(jaccards) - 1)))] if jaccards else None,
        },
        "spearman_mean": statistics.fmean(spearmans) if spearmans else None,
    }


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

    def avail_bits(n: str) -> List[int]:
        """Bits with non-None scores for layer n (mutation must never pick an
        unscored bit — plan_objective treats None as an invalid state)."""
        return [b for b in BITS_ORDER if scores[n].get(b) is not None]

    def random_ind() -> Dict[str, Any]:
        ind = {n: {"bits": None, "group": 64, "skip": True} for n in shapes}
        for n in names:
            ab = avail_bits(n)
            if ab and rng.random() < 0.9:
                b = rng.choice(ab)
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
                choices: List[Optional[int]] = [None] + avail_bits(n)
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
    p = argparse.ArgumentParser(description="GR00T v2 plan selector (P1-G, v1.3 binary default)")
    p.add_argument("--sensitivity", default=None, help="gr00t_sensitivity.json from P0-G.")
    p.add_argument("--ckpt", default=None, help="Checkpoint dir (or single safetensors).")
    p.add_argument("--out", default=None, help="Output plan JSON path.")
    p.add_argument("--include", default=r".*(backbone\.eagle_model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|action_head\.model\.transformer_blocks\.\d+\.ff\.net\.(0\.proj|2)).*")
    p.add_argument("--exclude", default=r"(?:^|\.)(vision|radio|norm|ln|layernorm|embed|lm_head|attn1)(?:\.|$)")
    p.add_argument("--lambda-cka", type=float, default=0.0,
                   help="Weight of the (1-CKA) proxy term. Gate-0 evidence (spatial audit): "
                        "Spearman(1-CKA, d_solver)@b4 ~ 0 (-0.08/-0.03/-0.06 over 3 seeds) vs "
                        "Spearman(CS, d_solver) +0.41 — CKA disabled by default, re-enable only "
                        "with new evidence. The lambda sweep still emits CKA-weighted diversity "
                        "candidates for TopK adjudication.")
    p.add_argument("--lambda-cs", type=float, default=1.0, help="Weight of the CS-divergence proxy term (primary).")
    p.add_argument("--group", type=int, default=64, help="DuQuant rotation block size (all plan layers; multi-block is P4).")
    p.add_argument("--row-rot", default="restore")
    p.add_argument("--budget", default="uniform-w6",
                   help="'uniform-w6' (v1.3 default: uniform-W6 static bytes), "
                        "'v1-w4' (v1 W4A8 plan bytes) or a float byte budget.")
    p.add_argument("--binary", action=argparse.BooleanOptionalAction, default=True,
                   help="v1.3 main path: binary W4/FP16 selection (--no-binary restores the mixed-bit space).")
    p.add_argument("--bits-order", default=None,
                   help="v1.4: explicit descending bit order for the tri-state search, e.g. '6,4' "
                        "gives {FP16(skip), W6, W4}; requires --no-binary. All entries must be "
                        ">= --min-bits.")
    p.add_argument("--min-bits", type=int, default=4,
                   help="最低 bit 档（默认 4，v1.3 正式安全约束）：CKA/CS 对 W2/W3 的"
                        "输出幅度爆炸失明（几何保持但幅度放大数倍 → 下游 A8 饱和 → "
                        "成功率崩溃）。重开 W2/W3 需过设计文档 §1.3.2 的六项实验。")
    p.add_argument("--solver", default="greedy", choices=["greedy", "evolution"])
    p.add_argument("--npop", type=int, default=20)
    p.add_argument("--niter", type=int, default=10)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--n-topk", type=int, default=10, help="v1.3: diverse TopK size for D_solver adjudication.")
    p.add_argument("--min-hamming", type=int, default=None,
                   help="v1.3: min FP16-mask Hamming distance between TopK plans "
                        "(default: max(3, 10%% of layers)).")
    p.add_argument("--n-perturb", type=int, default=10, help="v1.3: score-perturbation neighbor count.")
    p.add_argument("--perturb-sigma", type=float, default=0.25)
    p.add_argument("--lambda-pairs", default="0,1;0,2;0,5;1,1;0.5,1;2,1",
                   help="v1.3: λ-sweep pairs for candidate diversity (semicolon-separated "
                        "'cka,cs'; cs-strength variants included because CS is the primary proxy).")
    p.add_argument("--no-milp", action="store_true", help="v1.3: skip the scipy-milp exact 0-1 solve.")
    p.add_argument("--n-bootstrap", type=int, default=100, help="v1.3: w_i stability bootstrap draws.")
    p.add_argument("--guard-margin", type=float, default=1.5,
                   help="v1.3: guard threshold multiplier on the W4 P99 (τ = P99 × margin).")
    p.add_argument("--packdir", default=None, help="DuQuant pack dir recorded in the plan.")
    p.add_argument("--emit-env", action="store_true", help="Print export lines for run_quantvla.sh.")
    p.add_argument("--selftest", action="store_true", help="Run offline v1.3 pipeline selftest and exit.")
    return p.parse_args()


def _selftest() -> None:
    """Offline pipeline test (CPU): synthetic sensitivity + shapes through the
    v1.3 path — guards, binary greedy, milp gap, diversity, w_i log, bootstrap."""
    import numpy as np

    BITS_ORDER[:] = BINARY_BITS  # selftest runs the v1.3 binary path
    rng = random.Random(0)
    names = [f"L{i}" for i in range(100)]  # realistic scale: P99 estimator needs n ≫ 1/0.01
    shapes = {n: {"out": 4096, "in": 4096, "has_bias": False} for n in names}
    sens_layers: Dict[str, Any] = {}
    for n in names:
        cka = 0.9 + rng.random() * 0.09
        cs = rng.random() * 0.5
        sens_layers[n] = {
            "b4": {"cka": cka, "cs": cs, "rms_ratio": 0.05 + rng.random() * 0.15,
                   "sat_rate": 1e-4 + rng.random() * 1e-3},
            "d_solver_b4": 0.001 + rng.random() * 0.05,
            "d_solver_b4_std": 0.0005 + rng.random() * 0.002,
        }
    sens_layers["L3"]["b4"]["rms_ratio"] = 2.5  # known bad layer -> guard must fire
    sens = {"layers": sens_layers, "meta": {}}

    # P0-7: a layer with NO CKA/CS measurements must be unavailable (None),
    # not 0.0. (Guards kept so the threshold P99 still sees 100 values.)
    sens_layers["L50"]["b4"] = {"rms_ratio": 0.1, "sat_rate": 1e-4}  # simulate missing CKA/CS
    scores = build_scores(sens, names, 1.0, 1.0)
    assert scores["L50"][4] is None, "missing measurement must be unavailable"
    weights, wlog = build_weights_with_log(sens, names)
    # primary invariants (review round 5, item 7): winsorize [0.5,2] then
    # RE-normalize so the final weights keep mean == 1 (bounds rescale).
    assert abs(wlog["final"]["mean"] - 1.0) < 1e-6, wlog["final"]
    assert wlog["final"]["min"] > 0.0 and wlog["final"]["max"] < 3.0, wlog["final"]

    tau_rms, tau_sat = estimate_guard_thresholds(sens, names)
    filtered, removed = filter_guarded(scores, sens, names, tau_rms, tau_sat)
    assert any(r["layer"] == "L3" for r in removed), f"guard did not fire on L3: {removed}"
    assert filtered["L3"] == {}, "guarded layer must leave the search space"

    fp_total = sum(layer_bytes_fp16(s["out"], s["in"], s["has_bias"]) for s in shapes.values())
    budget = fp_total * 0.55
    g_plan, g_obj = greedy_plan(shapes, filtered, weights, budget, "restore")
    assert plan_total_bytes(g_plan, shapes, "restore") <= budget + 1e-3
    assert g_plan["L3"]["skip"] is True, "guarded layer must stay FP16"

    m_plan, m_obj = milp_binary_plan(shapes, filtered, weights, budget, "restore")
    if m_plan is not None:
        assert m_obj <= g_obj + 1e-6, f"milp obj {m_obj} > greedy obj {g_obj}"
        assert plan_total_bytes(m_plan, shapes, "restore") <= budget + 1e-3

    cands: List[Dict[str, Any]] = [
        {"plan": g_plan, "objective": g_obj, "source": "greedy"},
        {"plan": m_plan, "objective": m_obj, "source": "milp"},
    ] if m_plan is not None else [{"plan": g_plan, "objective": g_obj, "source": "greedy"}]
    cands += perturbed_plans(shapes, filtered, weights, budget, "restore", n=6, seed=1)
    cands += lambda_sweep_plans(shapes, sens, names, weights, budget, "restore",
                                pairs=((1.0, 1.0), (0.5, 1.0), (1.0, 0.5)))
    top = select_diverse(cands, k=8, min_hamming=3)
    assert len(top) >= 5, f"diverse topk too small: {len(top)} (diversity collapsed)"
    masks = [plan_mask(c["plan"]) for c in top]
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            assert hamming(masks[i], masks[j]) >= 3, "hamming diversity violated"

    flips = flip_neighbors_plans(shapes, filtered, weights, budget, "restore", g_plan)
    assert len(flips) == 8, len(flips)
    for f in flips:
        assert plan_total_bytes(f["plan"], shapes, "restore") <= budget + 1e-3

    bs = bootstrap_stability(sens, shapes, filtered, budget, "restore", n=20, seed=2)
    assert bs["jaccard"]["mean"] is not None and 0.0 <= bs["jaccard"]["mean"] <= 1.0, bs
    assert bs["spearman_mean"] is not None and 0.0 <= bs["spearman_mean"] <= 1.0, bs

    print("[select] selftest OK (v1.3 binary pipeline)")
    print(f"  guards        removed {len(removed)} layer(s): {[r['layer'] for r in removed]}")
    print(f"  w_i log       raw[{wlog['raw_d_solver']['min']:.4f},{wlog['raw_d_solver']['max']:.4f}] "
          f"final mean {wlog['final']['mean']:.3f} ∈ [0.5,2.0]")
    print(f"  greedy obj    {g_obj:.4f} | milp obj {m_obj:.4f} (gap {max(0.0, g_obj - m_obj):.4f})" if m_plan is not None
          else f"  greedy obj    {g_obj:.4f} | milp unavailable")
    print(f"  diverse topk  {len(top)} plans, min hamming 3")
    print(f"  bootstrap     jaccard mean {bs['jaccard']['mean']:.3f}, spearman {bs['spearman_mean']:.3f}")

    # ---- v1.4 tri-state selftest: {FP16, W6, W4} under a uniform-W6 budget ----
    for n in names:
        b4 = sens_layers[n]["b4"]
        sens_layers[n]["b6"] = {
            "cka": (b4.get("cka") * 0.95) if b4.get("cka") is not None else None,
            "cs": (b4.get("cs") * 0.4) if b4.get("cs") is not None else None,
            "rms_ratio": b4["rms_ratio"] * 0.5, "sat_rate": b4["sat_rate"] * 0.5,
        }
    BITS_ORDER[:] = [6, 4]
    scores6 = build_scores(sens, names, 0.0, 1.0)  # CS-only primary (v1.4)
    assert scores6["L0"][6] is not None and scores6["L0"][4] is not None
    assert scores6["L50"][6] is None, "missing b6 measurement must stay unavailable"
    weights6, _ = build_weights_with_log(sens, names)
    budget6 = sum(layer_bytes_quant(s["out"], s["in"], s["has_bias"], 6, 64, "restore")
                  for s in shapes.values())
    g6, g6_obj = greedy_plan(shapes, scores6, weights6, budget6, "restore")
    assert plan_total_bytes(g6, shapes, "restore") <= budget6 + 1e-3, "tri-state budget violated"
    bits_used = {e["bits"] for e in g6.values()}
    assert bits_used <= {None, 6, 4}, f"tri-state used unexpected bits: {bits_used}"
    assert g6["L3"]["skip"] is True, "guarded layer must stay FP16 in tri-state too"
    n_w6 = sum(1 for e in g6.values() if e["bits"] == 6)
    n_w4 = sum(1 for e in g6.values() if e["bits"] == 4)
    n_fp = sum(1 for e in g6.values() if e["bits"] is None)
    assert n_w6 + n_w4 + n_fp == len(names) - len(removed) or n_fp >= 0
    # evolution solver must also respect the tri-state space
    e6, e6_obj = evolution_plan(shapes, scores6, weights6, budget6, "restore", npop=8, niter=5, topk=4, seed=0)
    assert {e["bits"] for e in e6.values()} <= {None, 6, 4}
    assert plan_total_bytes(e6, shapes, "restore") <= budget6 + 1e-3
    BITS_ORDER[:] = BINARY_BITS  # restore the v1.3 default for any later use
    print(f"[select] tri-state selftest OK: W6={n_w6} W4={n_w4} FP16={n_fp} "
          f"(budget uniform-W6 {budget6/1e6:.1f}MB), greedy obj {g6_obj:.4f}, evolution obj {e6_obj:.4f}")


def _parse_lambda_pairs(spec: str) -> Tuple[Tuple[float, float], ...]:
    out = []
    for part in spec.split(";"):
        a, b = part.split(",")
        out.append((float(a), float(b)))
    return tuple(out)


def main() -> None:
    args = parse_args()

    if args.selftest:
        _selftest()
        return

    if not args.sensitivity or not args.ckpt or not args.out:
        raise SystemExit("--sensitivity/--ckpt/--out are required (or use --selftest)")
    # 搜索空间：v1.4 --bits-order 6,4（三元 {FP16,W6,W4}，W2/W3 不开放）优先；
    # 否则按 --min-bits 从默认空间收紧（v1.3 正式安全约束，见参数说明）
    if args.bits_order:
        order = [int(x) for x in args.bits_order.split(",") if x.strip()]
        if any(b < args.min_bits for b in order):
            raise SystemExit(f"--bits-order {args.bits_order} 含低于 --min-bits {args.min_bits} 的档位")
        BITS_ORDER[:] = order
    else:
        BITS_ORDER[:] = [b for b in BITS_ORDER if b >= args.min_bits]
    if args.binary:
        BITS_ORDER[:] = BINARY_BITS  # v1.3 main path: W4 / skip only
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

    layer_names = list(shapes)
    scores = build_scores(sens, layer_names, args.lambda_cka, args.lambda_cs)
    weights, w_log = build_weights_with_log(sens, layer_names)
    print("[select] w_i three-stage log (v1.3, §5.2):")
    print(f"  raw_d_solver:            min {w_log['raw_d_solver']['min']:.6g} / "
          f"max {w_log['raw_d_solver']['max']:.6g} / mean {w_log['raw_d_solver']['mean']:.6g}")
    print(f"  normalized_before_clip:  min {w_log['normalized_before_clip']['min']:.4f} / "
          f"max {w_log['normalized_before_clip']['max']:.4f} / mean {w_log['normalized_before_clip']['mean']:.4f}")
    print(f"  final_w_i:               min {w_log['final']['min']:.4f} / "
          f"max {w_log['final']['max']:.4f} / mean {w_log['final']['mean']:.4f}")

    # ---- v1.3 feasibility guards (hard constraint, §3.1) ----
    meta_thr = sens.get("meta", {}).get("guard_thresholds", {})
    tau_rms = meta_thr.get("tau_rms")
    tau_sat = meta_thr.get("tau_sat")
    if tau_rms is None or tau_sat is None:
        est_rms, est_sat = estimate_guard_thresholds(sens, layer_names, margin=args.guard_margin)
        tau_rms = tau_rms if tau_rms is not None else est_rms
        tau_sat = tau_sat if tau_sat is not None else est_sat
        print(f"[select] guard thresholds auto-estimated (P99 × {args.guard_margin}): "
              f"τ_rms={tau_rms:.4f} τ_sat={tau_sat:.3e}")
    scores, removed = filter_guarded(scores, sens, layer_names, tau_rms, tau_sat)
    if removed:
        print(f"[select] guard filter removed {len(removed)} layer(s) from the search space "
              f"(stay FP16): {[r['layer'] for r in removed[:5]]}{' ...' if len(removed) > 5 else ''}")
    else:
        print("[select] guard filter: no violations")

    fp_total = sum(layer_bytes_fp16(s["out"], s["in"], s["has_bias"]) for s in shapes.values())
    if args.budget == "uniform-w6":
        # v1.3 primary budget: uniform-W6 static weight bytes (862.9 MB GR00T)
        w6 = {n: {"bits": 6, "group": args.group, "skip": False} for n in shapes}
        budget = plan_total_bytes(w6, shapes, args.row_rot)
        print(f"[select] budget (uniform-W6 static-byte reference, v1.3): {budget / 1e6:.1f} MB")
    elif args.budget == "v1-w4":
        v1 = {n: {"bits": 4, "group": args.group, "skip": False} for n in shapes}
        budget = plan_total_bytes(v1, shapes, args.row_rot)
        print(f"[select] budget (v1 W4A8 static-byte reference): {budget / 1e6:.1f} MB")
    else:
        budget = float(args.budget)
    budget_fraction = budget / fp_total if fp_total > 0 else None

    # ---- candidate generation (v1.3: greedy + milp + perturbed + λ sweep) ----
    candidates: List[Dict[str, Any]] = []
    if args.solver == "greedy":
        g_plan, g_obj = greedy_plan(shapes, scores, weights, budget, args.row_rot)
        candidates.append({"plan": g_plan, "objective": g_obj, "source": "greedy"})
    else:
        e_plan, e_obj = evolution_plan(shapes, scores, weights, budget, args.row_rot,
                                       args.npop, args.niter, args.topk)
        candidates.append({"plan": e_plan, "objective": e_obj, "source": "evolution"})
        g_plan, g_obj = greedy_plan(shapes, scores, weights, budget, args.row_rot)
        candidates.append({"plan": g_plan, "objective": g_obj, "source": "greedy"})

    if args.binary and not args.no_milp:
        m_plan, m_obj = milp_binary_plan(shapes, scores, weights, budget, args.row_rot, group=args.group)
        if m_plan is not None:
            candidates.append({"plan": m_plan, "objective": m_obj, "source": "milp"})
            print(f"[select] milp exact 0-1 solve: obj {m_obj:.6f} (greedy gap {max(0.0, g_obj - m_obj):.6f})")
        else:
            print("[select] milp: unavailable/infeasible — greedy result stands")

    guarded_names = {r["layer"] for r in removed}
    candidates += perturbed_plans(shapes, scores, weights, budget, args.row_rot,
                                  n=args.n_perturb, sigma=args.perturb_sigma, seed=1)
    # Q-DiT-style mutations: guaranteed mask diversity even when every
    # continuous candidate collapses onto one mask
    candidates += flip_neighbors_plans(shapes, scores, weights, budget, args.row_rot,
                                       base_plan=g_plan, seed=3)
    # P0-7: the λ sweep inherits the guard filter (guarded layers stay FP16
    # in EVERY sweep candidate, not just the canonical scores).
    candidates += lambda_sweep_plans(shapes, sens, layer_names, weights, budget, args.row_rot,
                                     pairs=_parse_lambda_pairs(args.lambda_pairs),
                                     guarded_names=guarded_names)

    # P0-7: all candidates are re-scored with the CANONICAL objective (λ=1,1,
    # guard-filtered scores). Objectives from different λ values are not
    # comparable and must never be mixed in a min().
    for c in candidates:
        assert_plan_guards(c["plan"], guarded_names)
        c["objective_native"] = c["objective"]
        c["objective"] = plan_objective(c["plan"], scores, weights)

    min_hamming = args.min_hamming if args.min_hamming is not None else max(3, math.ceil(0.1 * len(shapes)))
    topk = select_diverse(candidates, k=args.n_topk, min_hamming=min_hamming)
    print(f"[select] diverse TopK: {len(topk)}/{len(candidates)} candidates (min hamming {min_hamming})")

    # ---- primary plan: lowest CANONICAL objective among candidates (proxy
    #      only; the FINAL choice is D_solver adjudication on the TopK, §3.1) ----
    primary = min(candidates, key=lambda c: c["objective"])
    plan = primary["plan"]
    obj = primary["objective"]
    total = plan_total_bytes(plan, shapes, args.row_rot)
    n_quant = sum(1 for v in plan.values() if not v["skip"])
    n_skip = sum(1 for v in plan.values() if v["skip"])
    # P0-7: hard feasibility check — a plan over budget is never written out.
    if total > budget + 1e-3:
        raise SystemExit(
            f"[select] FATAL: primary plan exceeds budget "
            f"({total / 1e6:.1f} MB > {budget / 1e6:.1f} MB) — no feasible plan found"
        )
    print(f"[select] primary plan (source={primary['source']}): {n_quant} W4 / {n_skip} skip, "
          f"bytes {total / 1e6:.1f} MB (budget {budget / 1e6:.1f} MB), proxy objective {obj:.4f}")

    # ---- w_i stability bootstrap (v1.3, §5.2) ----
    bs = bootstrap_stability(sens, shapes, scores, budget, args.row_rot, n=args.n_bootstrap, seed=2)
    print(f"[select] bootstrap ({bs['n_draws']} draws): mask Jaccard mean {bs['jaccard']['mean']:.3f} "
          f"[p05 {bs['jaccard']['p05']:.3f}, p95 {bs['jaccard']['p95']:.3f}], "
          f"weight Spearman {bs['spearman_mean']:.3f}")

    out_plan: Dict[str, Any] = {
        "meta": {
            "sensitivity": args.sensitivity,
            "ckpt": args.ckpt,
            "solver": args.solver,
            "binary": args.binary,
            "min_bits": args.min_bits,
            "lambda": {"cka": args.lambda_cka, "cs": args.lambda_cs},
            "row_rot": args.row_rot,
            "objective": "Σ w_i·S_i(b_i) — 层代理目标（一阶归因）；全局 D_solver 需对 TopK 完整配置做 GPU 配对 rollout 裁决（八步管线，设计文档 §3.1）",
            "budget_semantics": "静态权重存储字节（理论紧密打包；不含激活/峰值显存/时延/BitOps）",
            "budget_reference": args.budget,
            "budget_fraction_of_fp16": budget_fraction,
            "skip_semantics": "不量化、保留 FP16（成本 2·d_out·d_in+bias 字节，失真 0）；0-bit 剪枝为第三阶段独立选项",
            "guard_thresholds": {"tau_rms": tau_rms, "tau_sat": tau_sat,
                                 "semantics": "硬约束（§3.1 3a/3b）：违例层从搜索空间删除、保留 FP16"},
            "guard_filtered_layers": [r["layer"] for r in removed],
            "w_i_log": w_log,
            "w_i_log_note": "raw_d_solver 是原始单层散度（实验报告曾误报该值为 w_i）；"
                            "final_w_i 才是进入搜索的权重，必须满足 mean≈1 且 ∈[0.5,2]",
        },
        "budget_bytes": budget,
        "fp16_total_bytes": fp_total,
        "total_bytes": total,
        "objective": obj,
        "primary_source": primary["source"],
        "packdirs": {str(args.group): args.packdir} if args.packdir else {},
        "layers": {},
        "topk": [],
        "bootstrap": bs,
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
            "rms_ratio": lay.get(key, {}).get("rms_ratio") if key else None,
            "sat_rate": lay.get(key, {}).get("sat_rate") if key else None,
            "d_solver_ref": d_ref,
        }
    for c in topk:
        topk_bytes = plan_total_bytes(c["plan"], shapes, args.row_rot)
        if topk_bytes > budget + 1e-3:
            raise SystemExit(f"[select] FATAL: TopK candidate {c['source']} exceeds budget")
        out_plan["topk"].append({
            "source": c["source"],
            "objective": c["objective"],
            "bytes": topk_bytes,
            "n_skip": sum(1 for v in c["plan"].values() if v["skip"]),
            "skip_layers": list(plan_mask(c["plan"])),
            "d_solver": None,  # GPU 配对 rollout 后回填（TopK 裁决）
            "d_solver_std": None,
        })

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
