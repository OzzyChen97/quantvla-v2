#!/usr/bin/env python3
"""GR00T v2 metric validity audit (v1.3 gate 0, design doc §6.6.1).

Answers three pre-questions BEFORE any LIBERO run:
  1. Can the scores distinguish bit widths?  (bit monotonicity + W2-vs-W8 separation)
  2. Does the layer ranking correlate with the action-side d_solver?  (Spearman)
  3. Is the ranking stable across synthetic-calibration seeds?  (seed stability)

Protocol: 20–30 representative layers (LLM early/mid/late, q/k/v/o, MLP, DiT
early/mid/late, action-head neighbors) × bits {2,4,6,8} (W2 is a STRESS probe
only — it never enters the search space) × metrics:
  1−CKA, CS (cross term reported separately), D_rms, D_sat, output NMSE,
  single-layer d_solver (intervention vs the weight_bits=0 reference).

Output JSON (default checkpoints/packs/gr00t/metric_audit_libero_<suite>.json):
  {"meta": {...}, "stats": {...}, "seeds": [ {seed, layers, stats} ]}

Usage (groot_test env, one idle GPU):

    python scripts/tools/gr00t_metric_audit.py --suite spatial \
        --layers-subset 30 --bits 2,4,6,8 --n-seeds 3 --n-obs 8
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "tools"))

from gr00t.model.policy import COMPUTE_DTYPE  # noqa: E402
from gr00t_v2_common import (  # noqa: E402
    PACKDIR_TEMPLATE,
    SUITE_DIRS,
    ensure_flash_attn_rpath,
    fixed_calibration_buffer,
    load_policy,
    make_obs,
    resolve_data_config,
    restore_quant_env,
    set_quant_env,
    strip_quant_env,
)
from gr00t_sensitivity_probe import (  # noqa: E402
    REF_BITS,
    _LayerCollector,
    discover_targets,
    guard_metrics,
    run_activations,
    run_rollouts,
    set_all_bits,
    set_single_layer_bits,
    solver_divergence,
    subset_names,
)

AUDIT_BITS_DEFAULT = [2, 4, 6, 8]  # W2 = stress probe only


# --------------------------------------------------------------------------- #
# Stats (pure, offline-testable)
# --------------------------------------------------------------------------- #
def _nmse(fp: torch.Tensor, q: torch.Tensor) -> float:
    fp = fp.detach().to(torch.float32)
    q = q.detach().to(torch.float32)
    den = (fp * fp).mean()
    if float(den) <= 0:
        return 0.0
    return float(((q - fp) ** 2).mean() / den)


def _spearman(a: List[float], b: List[float]) -> float:
    n = len(a)
    if n < 2:
        return 0.0

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
    ma, mb = float(np.mean(ra)), float(np.mean(rb))
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den > 1e-12 else 1.0


def compute_audit_stats(layers: Dict[str, Dict[str, Any]], bits: List[int]) -> Dict[str, Any]:
    """Per-bit Spearman(metric, d_solver) + bit monotonicity + W2-vs-W8 separation.

    layers: {name: {"b4": {metric: value, ...}, "d_solver_b4": float, ...}}
    """
    metrics = ["cka_loss", "cs", "cs_cross", "rms_ratio", "sat_rate", "nmse"]
    keys = {"cka_loss": "cka", "cs": "cs", "cs_cross": "cs_cross",
            "rms_ratio": "rms_ratio", "sat_rate": "sat_rate", "nmse": "nmse"}
    stats: Dict[str, Any] = {"spearman": {}, "monotonicity": {}, "w2_vs_w8": {}}

    for b in bits:
        xs: Dict[str, List[float]] = {m: [] for m in metrics}
        ys: List[float] = []
        names: List[str] = []
        for n, entry in layers.items():
            d = entry.get(f"d_solver_b{b}")
            if d is None:
                continue
            e = entry.get(f"b{b}", {})
            ok = True
            row: Dict[str, float] = {}
            for m in metrics:
                v = e.get(keys[m])
                if m == "cka_loss" and e.get("cka") is not None:
                    v = 1.0 - float(e["cka"])
                if v is None:
                    ok = False
                    break
                row[m] = float(v)
            if not ok:
                continue
            for m in metrics:
                xs[m].append(row[m])
            ys.append(float(d))
            names.append(n)
        for m in metrics:
            stats["spearman"].setdefault(f"b{b}", {})[m] = (
                _spearman(xs[m], ys) if len(xs[m]) >= 2 else None
            )

        # bit monotonicity: for each layer, metric(W8) ≤ metric(W6) ≤ metric(W4)
        ordered = [b for b in sorted(bits, reverse=True) if b >= 4]
        stats["monotonicity"].setdefault(f"b{b}", {})
        if b in ordered:
            for m in metrics:
                good = bad = 0
                for n, entry in layers.items():
                    vals = []
                    for bb in ordered:
                        e = entry.get(f"b{bb}", {})
                        v = e.get(keys[m])
                        if m == "cka_loss" and e.get("cka") is not None:
                            v = 1.0 - float(e["cka"])
                        if v is None:
                            break
                        vals.append(float(v))
                    if len(vals) != len(ordered):
                        continue
                    if all(vals[i] <= vals[i + 1] + 1e-9 for i in range(len(vals) - 1)):
                        good += 1
                    else:
                        bad += 1
                stats["monotonicity"][f"b{b}"][m] = {"monotonic": good, "violated": bad}

    # W2 vs W8 separation (the known-bad-case probe): if many layers show
    # metric(W2) ≈ metric(W8), the collection pipeline cannot be trusted for
    # mixed-bit search (keep binary + min_bits=4, design §6.6.1).
    # hard-gate inputs (review round 3, item 6): finite metric rate, and the
    # W2 stress guard fire rate (fraction of layers whose W2 guard values exceed
    # τ = P99(W4 candidates) × 1.5 — the known-bad case must fire).
    finite = tot = 0
    for entry in layers.values():
        for b in bits:
            e = entry.get(f"b{b}", {})
            for k in ("cka", "cs", "cs_cross", "rms_ratio", "sat_rate", "nmse"):
                v = e.get(k)
                if v is not None:
                    tot += 1
                    if math.isfinite(float(v)):
                        finite += 1
    stats["finite_rate"] = finite / tot if tot else 1.0

    if 2 in bits:
        rms4 = [float(entry["b4"]["rms_ratio"]) for entry in layers.values()
                if entry.get("b4", {}).get("rms_ratio") is not None]
        sat4 = [float(entry["b4"]["sat_rate"]) for entry in layers.values()
                if entry.get("b4", {}).get("sat_rate") is not None]
        if rms4 and sat4:
            def _p99(vals, margin=1.5):
                srt = sorted(vals)
                idx = max(0, min(len(srt) - 1, math.ceil(0.99 * len(srt)) - 1))
                return srt[idx] * margin
            tau_r, tau_s = _p99(rms4), _p99(sat4)
            fires = tot2 = 0
            for entry in layers.values():
                b2 = entry.get("b2", {})
                if b2.get("rms_ratio") is not None and b2.get("sat_rate") is not None:
                    tot2 += 1
                    if float(b2["rms_ratio"]) > tau_r or float(b2["sat_rate"]) > tau_s:
                        fires += 1
            stats["guard_fire_w2"] = {
                "rate": fires / tot2 if tot2 else None, "n": tot2,
                "tau_rms": tau_r, "tau_sat": tau_s,
            }
        else:
            stats["guard_fire_w2"] = None

    if 2 in bits and 8 in bits:
        for m in metrics:
            ratios = []
            for n, entry in layers.items():
                v2 = entry.get("b2", {}).get(keys[m])
                v8 = entry.get("b8", {}).get(keys[m])
                if m == "cka_loss":
                    v2 = (1.0 - float(entry["b2"]["cka"])) if entry.get("b2", {}).get("cka") is not None else None
                    v8 = (1.0 - float(entry["b8"]["cka"])) if entry.get("b8", {}).get("cka") is not None else None
                if v2 is not None and v8 is not None and v8 > 0:
                    ratios.append(float(v2) / float(v8))
            if ratios:
                arr = np.array(ratios)
                stats["w2_vs_w8"][m] = {
                    "median_ratio": float(np.median(arr)),
                    "p05": float(np.percentile(arr, 5)),
                    "p95": float(np.percentile(arr, 95)),
                    "n_layers": len(ratios),
                }
            else:
                stats["w2_vs_w8"][m] = None
    return stats


def topk_mask(layers: Dict[str, Dict[str, Any]], metric_key: str, bit: int, k: int) -> Tuple[str, ...]:
    """The k layers with the SMALLEST metric value (proxy: 'least damaged') — the
    seed-stability unit is this FP16-vs-W4 split derived per metric."""
    vals = []
    for n, entry in layers.items():
        e = entry.get(f"b{bit}", {})
        if metric_key == "cka_loss":
            v = (1.0 - float(e["cka"])) if e.get("cka") is not None else None
        else:
            v = e.get(metric_key)
        if v is not None:
            vals.append((float(v), n))
    vals.sort()
    return tuple(sorted(n for _, n in vals[:k]))


def seed_stability(seed_layers: List[Dict[str, Dict[str, Any]]], bit: int = 4, k: int = 60) -> Dict[str, Any]:
    """Cross-seed agreement of metric rankings and derived W4 masks (design §6.6.1)."""
    metrics = ["cka_loss", "cs", "rms_ratio"]
    out: Dict[str, Any] = {}
    for m in metrics:
        masks = [topk_mask(ls, m, bit, k) for ls in seed_layers]
        pairs = [(a, b) for a in range(len(masks)) for b in range(a + 1, len(masks))]
        jaccards = []
        for a, b in pairs:
            sa, sb = set(masks[a]), set(masks[b])
            jaccards.append(len(sa & sb) / len(sa | sb) if sa or sb else 1.0)
        ranks = []
        for a, b in pairs:
            common = sorted(set(seed_layers[a]) & set(seed_layers[b]))
            va, vb = [], []
            for n in common:
                ea, eb = seed_layers[a][n].get(f"b{bit}", {}), seed_layers[b][n].get(f"b{bit}", {})
                if m == "cka_loss":
                    x = (1.0 - float(ea["cka"])) if ea.get("cka") is not None else None
                    y = (1.0 - float(eb["cka"])) if eb.get("cka") is not None else None
                else:
                    x, y = ea.get(m), eb.get(m)
                if x is not None and y is not None:
                    va.append(float(x))
                    vb.append(float(y))
            if len(va) >= 2:
                ranks.append(_spearman(va, vb))
        out[m] = {
            "mask_jaccard_mean": float(np.mean(jaccards)) if jaccards else None,
            "ranking_spearman_mean": float(np.mean(ranks)) if ranks else None,
        }
    return out


# --------------------------------------------------------------------------- #
# Collection (GPU; reuses the probe's protocol machinery)
# --------------------------------------------------------------------------- #
def collect_one_seed(
    model_fp, policy_fp, model_q, policy_q,
    obs_list, noises, batch_size, subset, bits, n_rollout_obs, gamma, max_tokens,
) -> Dict[str, Dict[str, Any]]:
    from gr00t.quantization.kernel_scores import LayerScoreBank

    n_obs = len(obs_list)
    # FP16 pass: trajectory (global ref) + per-layer outputs (guards/NMSE ref)
    fp_traj = run_rollouts(model_fp, policy_fp, obs_list, noises, batch_size)
    fp_col = _LayerCollector(subset, mode="q", banks={}, max_tokens=max_tokens)
    fp_col.install(model_fp)
    run_activations(model_fp, policy_fp, obs_list, noises, batch_size)
    fp_out = {n: fp_col.pooled(n) for n in subset}
    fp_col.remove()

    # reference R pass (wrapped pipeline, all-0) -> banks + ref trajectory
    banks: Dict[str, Any] = {n: LayerScoreBank(n, max_tokens=max_tokens) for n in subset}
    set_all_bits(model_q, REF_BITS)
    lin_ref = _LayerCollector(subset, mode="ref", banks=banks, max_tokens=max_tokens)
    lin_ref.install(model_q)
    ref_traj = run_rollouts(model_q, policy_q, obs_list, noises, batch_size)
    lin_ref.remove()
    for b in banks.values():
        b.finalize_ref()

    layers: Dict[str, Dict[str, Any]] = {}
    for name in subset:
        for bit in bits:
            set_all_bits(model_q, REF_BITS)
            set_single_layer_bits(model_q, name, bit)
            col = _LayerCollector([name], mode="q", banks=banks, max_tokens=max_tokens)
            col.install(model_q)
            run_activations(model_q, policy_q, obs_list, noises, batch_size)
            pooled = col.pooled(name)
            col.remove()
            s = banks[name].evaluate(pooled) if pooled is not None else {"cka": None, "cs": None, "cs_cross": None}
            s.update(guard_metrics(fp_out.get(name), pooled))
            s["nmse"] = _nmse(fp_out[name], pooled) if fp_out.get(name) is not None and pooled is not None else None

            # single-layer d_solver at THIS bit (intervention vs R).
            # P0-4 (correctness review): the target bit must stay active while
            # the q_traj rollout runs — the old code reset the layer to REF_BITS
            # first, so d_solver measured D(R, R) ≈ 0.
            q_traj = run_rollouts(model_q, policy_q, obs_list[:n_rollout_obs], noises[:n_rollout_obs], batch_size)
            ref_sub = ref_traj[:, :n_rollout_obs]
            mean_div, _ = solver_divergence(ref_sub, q_traj, gamma)
            set_single_layer_bits(model_q, name, REF_BITS)
            layers.setdefault(name, {})[f"b{bit}"] = s
            layers[name][f"d_solver_b{bit}"] = mean_div
            del q_traj
            gc.collect()
    return layers


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GR00T v2 metric validity audit (v1.3 gate 0)")
    p.add_argument("--suite", default="spatial", choices=["spatial", "goal", "object", "90", "10"])
    p.add_argument("--model-path", default=None)
    p.add_argument("--data-config", default=None,
                   help="Default: resolved per suite via SUITE_DATA_CONFIG (goal -> MeanStd).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--denoising-steps", type=int, default=8)
    p.add_argument("--n-obs", type=int, default=8, help="Synthetic obs per seed (activations + fp traj).")
    p.add_argument("--n-rollout-obs", type=int, default=4, help="Obs for per-layer d_solver (audit scale).")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--bits", default="2,4,6,8", help="Audit bits (W2 = stress probe only).")
    p.add_argument("--layers-subset", type=int, default=30, help="Stride subset of representative layers.")
    p.add_argument("--n-seeds", type=int, default=3, help="Synthetic calibration seeds.")
    p.add_argument("--group", type=int, default=64)
    p.add_argument("--ls", type=float, default=0.15)
    p.add_argument("--act-pct", type=float, default=99.9)
    p.add_argument("--row-rot", default="restore")
    p.add_argument("--calib-steps", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--gamma", type=float, default=1.2)
    p.add_argument("--packdir", default=None)
    p.add_argument("--include", default=None, help="Target include regex (default: probe default).")
    p.add_argument("--exclude", default=None, help="Target exclude regex (default: probe default).")
    p.add_argument("--out", default=None)
    p.add_argument("--selftest", action="store_true", help="Offline stats selftest and exit.")
    return p.parse_args()


def _selftest() -> None:
    rng = np.random.default_rng(0)
    bits = [2, 4, 6, 8]
    names = [f"L{i}" for i in range(30)]
    layers: Dict[str, Dict[str, Any]] = {}
    for n in names:
        base = rng.random()
        layers[n] = {}
        for b in bits:
            noise = rng.normal(0, 0.02)
            scale = (8.0 - b) / 4.0  # lower bit -> worse
            layers[n][f"b{b}"] = {
                "cka": 1.0 - 0.02 * scale - abs(noise),
                "cs": 0.1 * scale + abs(noise) * 0.1,
                "cs_cross": 0.1 * scale + abs(noise) * 0.1,
                "rms_ratio": 0.05 * scale + max(abs(noise), 0.005) * 0.05,
                "sat_rate": 1e-4 * scale + max(abs(noise), 0.005) * 1e-4,
                "nmse": 0.01 * scale + abs(noise) * 0.01,
            }
            # d_solver strongly driven by the same |noise| that drives cka_loss,
            # plus a bit-level term -> Spearman(metric, d_solver) must be high.
            layers[n][f"d_solver_b{b}"] = 0.01 * scale + 0.03 * base + 3.0 * abs(noise)
    stats = compute_audit_stats(layers, bits)
    sp = stats["spearman"]["b4"]["cka_loss"]
    assert sp is not None and sp > 0.3, f"spearman b4 cka_loss = {sp}"
    mono = stats["monotonicity"]["b8"]["rms_ratio"]
    assert mono["violated"] <= mono["monotonic"] / 2, mono
    w2 = stats["w2_vs_w8"]["rms_ratio"]
    assert w2["median_ratio"] > 2.0, w2  # synthetic data: W2 clearly worse than W8
    # seed stability: two seeds share the same underlying layer structure plus
    # small independent noise -> rankings must largely agree.
    base_vec = {n: float(v) for n, v in zip(names, np.linspace(0.002, 0.022, len(names)))}

    def draw(seed):
        r = np.random.default_rng(seed)
        ls = {}
        for n in names:
            noise = r.normal(0, 0.002)
            ls[n] = {f"b4": {"cka": 1.0 - (base_vec[n] + noise),
                             "cs": 0.1 + base_vec[n] * 10.0 + noise * 10.0,
                             "rms_ratio": 0.05 + base_vec[n] + noise}}
        return ls
    st = seed_stability([draw(0), draw(1)], bit=4, k=15)
    assert st["cka_loss"]["ranking_spearman_mean"] > 0.5, st
    assert st["cka_loss"]["mask_jaccard_mean"] > 0.5, st
    print("[metric_audit] selftest OK (offline stats)")
    print(f"  spearman(1-CKA, d_solver)@b4 = {sp:.3f}")
    print(f"  rms_ratio monotonicity b8 = {mono}")
    print(f"  W2/W8 rms_ratio median = {w2['median_ratio']:.2f}")
    print(f"  seed stability = {st}")


def main() -> None:
    args = parse_args()
    if args.selftest:
        _selftest()
        return

    bits = [int(x) for x in args.bits.split(",") if x.strip()]
    args.data_config = resolve_data_config(args.suite, args.data_config)
    suite_dir = SUITE_DIRS[args.suite]
    if args.model_path is None:
        args.model_path = str(REPO_ROOT / "checkpoints" / "gr00t" / suite_dir)
    if args.packdir is None:
        args.packdir = str(
            REPO_ROOT / "checkpoints/packs/gr00t"
            / PACKDIR_TEMPLATE.format(suite=args.suite, g=args.group, calib=args.calib_steps, ls=str(args.ls).replace(".", ""))
        )
    if args.out is None:
        args.out = str(REPO_ROOT / "checkpoints/packs/gr00t" / f"metric_audit_libero_{args.suite}.json")

    ensure_flash_attn_rpath()

    if args.include is None or args.exclude is None:
        from gr00t_v2_common import DEFAULT_EXCLUDE, DEFAULT_INCLUDE

        args.include = args.include or DEFAULT_INCLUDE
        args.exclude = args.exclude or DEFAULT_EXCLUDE

    # ---- target discovery (FP16, no quant env) ----
    saved_env = strip_quant_env()
    policy_fp = load_policy(args.model_path, data_config=args.data_config, denoising_steps=args.denoising_steps, device=args.device)
    model_fp = policy_fp.model
    target_names = discover_targets(model_fp, args)
    subset = subset_names(target_names, args.layers_subset)
    print(f"[metric_audit] targets {len(target_names)} -> audit subset {len(subset)} layers, bits {bits}, seeds {args.n_seeds}")
    horizon = int(model_fp.action_head.config.action_horizon)
    action_dim = int(model_fp.action_head.config.action_dim)
    del model_fp, policy_fp
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    restore_quant_env(saved_env)

    seed_results: List[Dict[str, Any]] = []
    for seed in range(args.n_seeds):
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        obs_list = [make_obs(rng, "libero") for _ in range(args.n_obs)]
        noises = [torch.randn(horizon, action_dim) for _ in obs_list]

        # FP16 model (quant env stripped) + quant model (quant env set) — both live
        saved_env2 = strip_quant_env()
        policy_fp = load_policy(args.model_path, data_config=args.data_config, denoising_steps=args.denoising_steps, device=args.device)
        model_fp = policy_fp.model
        set_quant_env(
            args.include, args.exclude, args.packdir, bits_default=4, group=args.group, ls=args.ls,
            act_pct=args.act_pct, calib_steps=args.calib_steps, row_rot=args.row_rot, act_dynamic=False,
        )
        policy_q = load_policy(args.model_path, data_config=args.data_config, denoising_steps=args.denoising_steps, device=args.device)
        model_q = policy_q.model
        set_all_bits(model_q, 4)

        # P0-2: static A8 calibration must reach cfg.calib_batches batches in
        # the all-zero reference state before any measurement.
        from gr00t.quantization.duquant_layers import all_calibrated, calibration_progress

        set_all_bits(model_q, REF_BITS)
        n_warm_obs = args.calib_steps * args.batch_size
        warm_obs, warm_noises, _ = fixed_calibration_buffer(
            seed, n_warm_obs, horizon, action_dim, fmt="libero"
        )
        run_rollouts(model_q, policy_q, warm_obs, warm_noises, args.batch_size, return_trajectory=False)
        full, total = calibration_progress(model_q)
        if not all_calibrated(model_q):
            raise SystemExit(f"[metric_audit] A8 calibration incomplete: {full}/{total}")

        t0 = time.time()
        layers = collect_one_seed(
            model_fp, policy_fp, model_q, policy_q,
            obs_list, noises, args.batch_size, subset, bits,
            args.n_rollout_obs, args.gamma, args.max_tokens,
        )
        stats = compute_audit_stats(layers, bits)
        seed_results.append({"seed": seed, "layers": layers, "stats": stats})
        print(f"[metric_audit] seed {seed} done in {time.time() - t0:.1f}s; "
              f"spearman(1-CKA,d_solver)@b4 = {stats['spearman'].get('b4', {}).get('cka_loss')}")
        restore_quant_env(saved_env2)
        del model_fp, policy_fp, model_q, policy_q
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    stability = seed_stability([s["layers"] for s in seed_results], bit=4, k=len(subset) // 2)
    out = {
        "meta": {
            "suite": args.suite, "model_path": args.model_path, "bits": bits,
            "layers_subset": args.layers_subset, "subset": subset,
            "n_seeds": args.n_seeds, "n_obs": args.n_obs, "n_rollout_obs": args.n_rollout_obs,
            "note": "gate 0 audit; W2 is a STRESS probe and never enters the search space",
            "stability": stability,
        },
        "seeds": seed_results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[metric_audit] saved -> {args.out}")
    print("[metric_audit] gate-0 reading: monotonicity + W2/W8 separation + seed stability (§6.6.1)")


if __name__ == "__main__":
    main()
