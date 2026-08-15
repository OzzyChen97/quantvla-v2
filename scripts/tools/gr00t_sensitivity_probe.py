#!/usr/bin/env python3
"""GR00T v2 sensitivity probe (P0-G measurement layer, v1.2 reference + v1.3 guards).

Scores (all computed as REFERENCE-vs-intervention differences on identical
synthetic inputs — data-free, the only external reference is the model itself):

  - CKA      : centered linear kernel alignment (ICLR 2025 hidden-state matching)
  - CS       : uncentered Cauchy-Schwarz divergence (CS-Aligner, ICLR 2026)
  - D_solver : paired-noise DENOISING-trajectory divergence — the flow-matching
               trajectory inside ONE action chunk (x_0 noise ... x_T final
               action, T+1 states). Solver-level metric, NOT an environment
               long-horizon rollout; the long-horizon link is validated by
               LIBERO evaluation, not measured here.

v1.3 additions (design doc §6.2 / §5.1.2–5.1.3):
  - feasibility guards per layer per bit, vs the PURE FP16 model (deployment
    reference): D_rms (median |log rms ratio| over channels), D_sat (fraction
    of outputs exceeding the downstream static-A8 range, proxied from the FP16
    output P99.9), amax ratio. Guard thresholds τ = P99(W4 candidates)×1.5 are
    written into meta for the selector's hard filtering.
  - main probe scans W4 only (--bits default "4"); --layers-subset N enables
    the audit mode on a deterministic stride-N layer subset.
  - w_i sampling norm: --n-rollout-obs 8 obs × 2 paired noises per obs
    (median aggregation, std recorded).
  - CS in-situ scaling check (--cs-in-situ-check): the cross term must respond
    monotonically when a real layer output is scaled by 2/4/8.

v1.2 reference protocol (fixes the set_all_bits(16) ambiguity):
  - REFERENCE R = the quantized pipeline with EVERY target layer at
    weight_bits = 0 (weights unquantized; rotations/permutation/A8 activation
    quantization stay active). weight_bits=0 hits the full-precision path in
    DuQuantLinear.forward — verified in code.
  - Single-layer intervention = only layer i at bit b, everything else at 0.
    Both sides share the wrapper and upstream behavior, so CKA/CS attribute
    layer i's WEIGHT quantization only (no upstream drift, no wrapper confound).
  - w_i (importance weights) = per-layer d_solver at one probing bit (default
    4) measured intervention-vs-R.
  - Global D_solver = FP16 model vs full config (all layers at b). This is the
    DEPLOY-relevant pairing and is a config-level scalar — never summed.
  - Base quantization mode of the probe: ATM/OHB OFF, per-step OFF, STATIC
    activation scale (GR00T_DUQUANT_ACT_DYNAMIC=0). ATM/OHB/dynamic-act are
    deployment-time corrections; calibrations for them must use the same act
    mode as deployment (see calibrate_atm_perstep_gr00t.py --act-dynamic).

Trajectory indexing: return_trajectory now yields T+1 states (index 0 = initial
noise, index T = final action) — no off-by-one in per-step weighting:
    div_k = ||x_k^R − x_k^q||² / (||x_k^R||² + ε),  w_k ∝ γ^{k+1} (normalized).

Paired noise gives pointwise pairs; conditional action-DISTRIBUTION divergence
is NOT estimated here (needs multiple noise samples per obs — third-stage ext).

Design doc: docs/quantvla_v2_design.md §6.2 (schema in §6.2.6).

Usage (groot_test env, one idle GPU):

    cd /home1/gyy/vla/QuantVLA
    export PYTHONPATH=/home1/gyy/vla/QuantVLA/code:$PYTHONPATH
    python scripts/tools/gr00t_sensitivity_probe.py --suite spatial \
        --n-obs 16 --bits 4 --group 64 --out <json>
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "tools"))

from gr00t.model.policy import COMPUTE_DTYPE  # noqa: E402
from gr00t_v2_common import (  # noqa: E402
    DEFAULT_EXCLUDE,
    DEFAULT_INCLUDE,
    PACKDIR_TEMPLATE,
    SUITE_DIRS,
    chunked,
    ensure_flash_attn_rpath,
    load_policy,
    make_l1_obs,
    make_obs,
    resolve_data_config,
    restore_quant_env,
    set_quant_env,
    stack_obs,
    strip_quant_env,
)

# Reference weight_bits for the v1.2 protocol: 0 = full-precision weight path
# inside the wrapped pipeline (rotations + A8 still active).
REF_BITS = 0


def discover_targets(model: torch.nn.Module, args: argparse.Namespace) -> List[str]:
    """Linear-layer names matching the include/exclude regexes (the v2 candidate set)."""
    from gr00t.quantization.duquant_layers import select_targets

    targets = select_targets(
        model,
        include_regex=args.include,
        exclude_regex=args.exclude,
        scope_prefix=None,
        whitelist=None,
        blacklist=None,
    )
    return [n for n, _ in targets]


def discover_attention_names(model: torch.nn.Module) -> List[str]:
    """DiT attention module names (action_head.model.transformer_blocks.*.attn1)."""
    from gr00t.atm.dit_atm import _is_dit_attention

    return [name for name, mod in model.named_modules() if _is_dit_attention(name, mod, scope="dit")]


# --------------------------------------------------------------------------- #
# Collectors (token-capped at hook time to keep memory bounded)
# --------------------------------------------------------------------------- #
class _LayerCollector:
    """Collects outputs of named linear layers (ref -> banks, q -> capped pools)."""

    def __init__(self, names: List[str], mode: str, banks: Dict[str, Any], max_tokens: int):
        self.names = set(names)
        self.mode = mode  # "ref" or "q"
        self.banks = banks
        self.max_tokens = max_tokens
        self.q_pools: Dict[str, List[torch.Tensor]] = {}
        self.handles: List[Any] = []
        self.hit_count: Dict[str, int] = {}

    def _hook_fn(self, name: str):
        def fn(module, args, output):
            from gr00t.quantization.kernel_scores import pool_samples

            if self.mode == "ref":
                self.banks[name].accumulate_ref(output)
            else:
                pooled = pool_samples(output, self.max_tokens)
                if pooled is not None:
                    self.q_pools.setdefault(name, []).append(pooled)
            self.hit_count[name] = self.hit_count.get(name, 0) + 1

        return fn

    def install(self, model: torch.nn.Module) -> None:
        self.remove()
        for name, mod in model.named_modules():
            if name in self.names:
                self.handles.append(mod.register_forward_hook(self._hook_fn(name)))

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []

    def pooled(self, name: str) -> Optional[torch.Tensor]:
        from gr00t.quantization.kernel_scores import pool_samples

        chunks = self.q_pools.pop(name, [])
        if not chunks:
            return None
        cat = torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]
        return pool_samples(cat, self.max_tokens)


class _AttentionCollector:
    """Collects DiT attention outputs via register_output_capture (step aware)."""

    def __init__(self, mode: str, banks: Dict[str, Any], max_tokens: int):
        self.mode = mode
        self.banks = banks
        self.max_tokens = max_tokens
        self.q_pools: Dict[str, List[torch.Tensor]] = {}
        self.hit_count: Dict[str, int] = {}

    def __call__(self, layer_name: str, tensor: torch.Tensor, step: Optional[int]) -> None:
        from gr00t.quantization.kernel_scores import pool_samples

        key = f"attn:{layer_name}"
        if self.mode == "ref":
            self.banks[key].accumulate_ref(tensor)
        else:
            pooled = pool_samples(tensor, self.max_tokens)
            if pooled is not None:
                self.q_pools.setdefault(key, []).append(pooled)
        self.hit_count[key] = self.hit_count.get(key, 0) + 1

    def pooled(self, key: str) -> Optional[torch.Tensor]:
        from gr00t.quantization.kernel_scores import pool_samples

        chunks = self.q_pools.pop(key, [])
        if not chunks:
            return None
        cat = torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]
        return pool_samples(cat, self.max_tokens)


# --------------------------------------------------------------------------- #
# Bit switching (in-process; weight cache auto-recomputes)
# --------------------------------------------------------------------------- #
def set_all_bits(model: torch.nn.Module, bits: int) -> int:
    from gr00t.quantization.duquant_layers import DuQuantLinear

    count = 0
    for _, m in model.named_modules():
        if isinstance(m, DuQuantLinear):
            m.weight_bits = int(bits)
            count += 1
    return count


def set_single_layer_bits(model: torch.nn.Module, name: str, bits: int) -> bool:
    from gr00t.quantization.duquant_layers import DuQuantLinear

    for n, m in model.named_modules():
        if n == name and isinstance(m, DuQuantLinear):
            m.weight_bits = int(bits)
            return True
    return False


# --------------------------------------------------------------------------- #
# Forward passes
# --------------------------------------------------------------------------- #
def run_rollouts(
    model: torch.nn.Module,
    policy: Any,
    obs_list: List[Dict[str, Any]],
    noises: List[torch.Tensor],
    batch_size: int,
    return_trajectory: bool = True,
) -> Optional[torch.Tensor]:
    """Paired-noise rollouts over batched chunks; returns (T+1, B_total, H, D).

    Index 0 = initial noise, index T = final action (see module docstring).
    """
    trajs: List[torch.Tensor] = []
    use_autocast = str(policy.device).startswith("cuda")
    for batched_obs, batched_noise in chunked(obs_list, noises, batch_size):
        norm = policy.apply_transforms(batched_obs)
        with torch.inference_mode():
            if use_autocast:
                with torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
                    out = model.get_action(norm, action_noise=batched_noise, return_trajectory=return_trajectory)
            else:
                out = model.get_action(norm, action_noise=batched_noise, return_trajectory=return_trajectory)
        if return_trajectory:
            trajs.append(out["_trajectory"])  # (T+1, B, H, D) cpu
    if not return_trajectory:
        return None
    return torch.cat(trajs, dim=1)  # (T+1, B_total, H, D)


def run_activations(
    model: torch.nn.Module,
    policy: Any,
    obs_list: List[Dict[str, Any]],
    noises: List[torch.Tensor],
    batch_size: int,
) -> None:
    """Forward-only pass with FIXED paired noises.

    v1.2.1 修正：所有 pass（参照与全部干预）使用同一组配对噪声，去噪轨迹
    完全一致——DiT 层输出不再因内部随机噪声漂移，参照池也不会混入多条轨迹。
    """
    use_autocast = str(policy.device).startswith("cuda")
    for batched_obs, batched_noise in chunked(obs_list, noises, batch_size):
        norm = policy.apply_transforms(batched_obs)
        with torch.inference_mode():
            if use_autocast:
                with torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
                    model.get_action(norm, action_noise=batched_noise)
            else:
                model.get_action(norm, action_noise=batched_noise)


def solver_divergence(
    ref_traj: torch.Tensor,
    q_traj: torch.Tensor,
    gamma: float,
) -> tuple:
    """Paired per-step normalized denoising-trajectory divergence.

    Trajectories are (T+1, B, H, D): index 0 = noise, index T = final action.
    Weights w_k ∝ γ^{k+1} are normalized over the T+1 states, so the final
    action step gets the largest weight.

    P0-3 (correctness review): the weighted per-step divergences are SUMMED
    (Σ_k w_k·div_k, Σw = 1), not averaged. With q = 2·ref every step's relative
    error is exactly 1, so the result must be exactly 1 — the previous
    mean(dim=0) returned 1/(T+1).

    Returns (mean_over_obs, per_obs_list).
    """
    ref = ref_traj.float()[:, : q_traj.shape[1]]
    q = q_traj.float()
    num = ((ref - q) ** 2).sum(dim=(-1, -2))  # (T+1, B)
    den = (ref**2).sum(dim=(-1, -2)).clamp_min(1e-8)  # ε 防分母爆炸
    rel = num / den
    k_steps = rel.shape[0]  # T+1
    weights = torch.tensor([gamma ** (k + 1) for k in range(k_steps)])
    weights = weights / weights.sum()  # 归一化：跨 T 近似可比（不同 T 仍是不同积分网格，见设计文档）
    per_obs = (rel * weights[:, None]).sum(dim=0)  # Σ_k w_k·div_k（P0-3：不是平均）
    return float(per_obs.mean()), [float(v) for v in per_obs]


def subset_names(names: List[str], k: Optional[int]) -> List[str]:
    """Deterministic stride subset (v1.3 --layers-subset / audit mode)."""
    if not k or k <= 0 or k >= len(names):
        return list(names)
    idx = np.unique(np.linspace(0, len(names) - 1, k).astype(int))
    return [names[i] for i in idx]


def guard_metrics(fp_out: Optional[torch.Tensor], q_out: Optional[torch.Tensor]) -> Dict[str, Optional[float]]:
    """v1.3 feasibility guards vs the PURE FP16 reference (deployment pairing)."""
    from gr00t.quantization.kernel_scores import amax_ratio, rms_ratio_median, sat_rate

    if fp_out is None or q_out is None:
        return {"rms_ratio": None, "amax_ratio": None, "sat_rate": None}
    return {
        "rms_ratio": rms_ratio_median(fp_out, q_out),
        "amax_ratio": amax_ratio(fp_out, q_out),
        "sat_rate": sat_rate(fp_out, q_out),
    }


# --------------------------------------------------------------------------- #
# Main probe
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GR00T v2 sensitivity probe (P0-G, v1.2 reference protocol)")
    p.add_argument("--suite", default="spatial", choices=["spatial", "goal", "object", "90", "10"])
    p.add_argument(
        "--model-path",
        default=None,
        help="Model checkpoint dir (default: checkpoints/gr00t/libero-<suite>).",
    )
    p.add_argument("--data-config", default=None,
                   help="Default: resolved per suite via SUITE_DATA_CONFIG (goal -> MeanStd).")
    p.add_argument("--obs-format", default="libero", choices=["libero", "gr1"],
                   help="合成 obs 格式：libero（默认）或 gr1（fourier_gr1_arms_waist）")
    p.add_argument("--embodiment-tag", default="new_embodiment",
                   help="模型 embodiment tag（GR1 tabletop 用 gr1）")
    p.add_argument("--device", default="cuda")
    p.add_argument("--denoising-steps", type=int, default=8)
    p.add_argument("--n-obs", type=int, default=16, help="Synthetic observations (CKA/CS + global D_solver).")
    p.add_argument("--batch-size", type=int, default=8, help="Obs batch size for one forward pass.")
    p.add_argument("--bits", default="4", help="Weight bit widths to scan (v1.3 main probe: 4; audit mode: 2,4,6,8). Reference is weight_bits=0.")
    p.add_argument("--group", type=int, default=64, help="DuQuant rotation block size (fixed for the whole scan).")
    p.add_argument("--ls", type=float, default=0.15)
    p.add_argument("--act-pct", type=float, default=99.9)
    p.add_argument("--row-rot", default="restore")
    p.add_argument("--calib-steps", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=1024, help="Token cap per layer for CKA/CS pools.")
    p.add_argument("--gamma", type=float, default=1.2, help="Late-denoising-step weight for solver divergence.")
    p.add_argument("--per-layer-bits", default="4", help="Probing bits for per-layer solver-divergence importance weights.")
    p.add_argument("--n-rollout-obs", type=int, default=8,
                   help="Obs count for per-layer rollouts (v1.3: 8–16; 2 paired noises per obs).")
    p.add_argument("--n-noises-per-obs", type=int, default=2,
                   help="v1.3: paired-noise repeats per obs for w_i (median aggregation).")
    p.add_argument("--layers-subset", type=int, default=0,
                   help="v1.3 audit mode: restrict attribution + importance rollouts to a "
                        "deterministic stride subset of N layers (0 = all).")
    p.add_argument("--cs-in-situ-check", action="store_true",
                   help="v1.3: verify the CS cross term responds monotonically when a real "
                        "layer output is scaled by 2/4/8 (closing criterion, §5.1.2).")
    p.add_argument("--guard-margin", type=float, default=1.5,
                   help="v1.3: τ = P99(W4 guard candidates) × margin, written into meta.")
    p.add_argument("--include", default=DEFAULT_INCLUDE)
    p.add_argument("--exclude", default=DEFAULT_EXCLUDE)
    p.add_argument("--packdir", default=None, help="DuQuant pack dir (default derives from suite/group/calib/ls).")
    p.add_argument("--out", default=None, help="Output JSON path.")
    p.add_argument("--skip-per-layer", action="store_true", help="Skip per-layer CKA/CS attribution passes.")
    p.add_argument("--skip-layer-rollouts", action="store_true", help="Skip per-layer D_solver importance rollouts.")
    p.add_argument("--dry-run", action="store_true", help="Load FP model, list targets, exit.")
    p.add_argument("--selftest", action="store_true", help="Run kernel_scores.selftest() and exit.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.selftest:
        from gr00t.quantization.kernel_scores import selftest

        selftest()
        # v1.3 offline helpers
        names = [f"L{i}" for i in range(100)]
        sub = subset_names(names, 20)
        assert len(sub) == 20 and sub[0] == names[0] and sub[-1] == names[-1], sub
        assert subset_names(names, 0) == names and subset_names(names, 500) == names
        x = torch.randn(256, 64)
        g = guard_metrics(x, x * 8.0)
        assert abs(g["rms_ratio"] - 2.0794) < 1e-3 and g["amax_ratio"] > 7.0 and g["sat_rate"] > 0.5, g
        g_none = guard_metrics(None, x)
        assert g_none == {"rms_ratio": None, "amax_ratio": None, "sat_rate": None}
        # solver_divergence: identical trajectories -> 0
        t = torch.randn(9, 4, 16, 6)  # T+1=9, B=4, H=16, D=6
        m, per = solver_divergence(t, t, gamma=1.2)
        assert m == 0.0 and all(p == 0.0 for p in per)
        m2, per2 = solver_divergence(t, t * 2.0, gamma=1.2)
        assert m2 > 0.0 and len(per2) == 4
        # P0-3: q = 2·ref -> every step's relative error is exactly 1 ->
        # weighted SUM (weights normalized) is exactly 1, NOT 1/(T+1).
        assert abs(m2 - 1.0) < 1e-6, f"P0-3: D_solver(2·ref) = {m2}, expected 1.0"
        print("[probe] selftest OK (offline helpers + P0-3 sum fix)")
        return

    args.bits = [int(x) for x in args.bits.split(",") if x.strip()]
    args.per_layer_bits = [int(x) for x in args.per_layer_bits.split(",") if x.strip()]
    args.data_config = resolve_data_config(args.suite, args.data_config)
    suite_dir = SUITE_DIRS[args.suite]
    if args.model_path is None:
        args.model_path = str(REPO_ROOT / "checkpoints" / "gr00t" / suite_dir)
    if args.packdir is None:
        args.packdir = str(
            REPO_ROOT
            / "checkpoints/packs/gr00t"
            / PACKDIR_TEMPLATE.format(suite=args.suite, g=args.group, calib=args.calib_steps, ls=str(args.ls).replace(".", ""))
        )
    if args.out is None:
        args.out = str(
            REPO_ROOT
            / "checkpoints/packs/gr00t"
            / f"sensitivity_libero_{args.suite}_g{args.group}_b{'_'.join(map(str, args.bits))}.json"
        )

    ensure_flash_attn_rpath()
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    print("=" * 100)
    print("[probe] GR00T v2 sensitivity probe (P0-G, v1.2 reference protocol)")
    print(f"[probe] model={args.model_path}")
    print(f"[probe] bits={args.bits} group={args.group} n_obs={args.n_obs} batch={args.batch_size}")
    print(f"[probe] base mode: ATM/OHB OFF, per-step OFF, static act scale; reference = weight_bits=0 pipeline")
    print(f"[probe] out={args.out}")
    print("=" * 100)

    # ---------------------------------------------------------------- FP16 pass
    # Used ONLY for the deploy-relevant global D_solver pairing (FP16 vs config).
    saved_env = strip_quant_env()
    policy_fp = load_policy(args.model_path, data_config=args.data_config, denoising_steps=args.denoising_steps, device=args.device, embodiment_tag=args.embodiment_tag)
    model_fp = policy_fp.model

    target_names = discover_targets(model_fp, args)
    attn_names = discover_attention_names(model_fp)
    print(f"[probe] target linear layers: {len(target_names)}")
    print(f"[probe] DiT attention modules: {len(attn_names)}")
    if args.dry_run:
        for n in target_names:
            print(f"  - {n}")
        print("[probe] dry-run done.")
        return

    horizon = int(model_fp.action_head.config.action_horizon)
    action_dim = int(model_fp.action_head.config.action_dim)

    n_total = max(args.n_obs, args.n_rollout_obs)
    obs_list = [make_obs(rng, args.obs_format) for _ in range(n_total)]
    # 每个 obs 一个 2D 噪声 (H, D)：chunked 打包时叠加 batch 维 → (B, H, D)
    noises = [torch.randn(horizon, action_dim) for _ in obs_list]
    # v1.3: 每个 obs 2 个配对噪声（w_i 中位数聚合，降低单噪声方差）
    if args.n_noises_per_obs >= 2:
        noises_b = [torch.randn(horizon, action_dim) for _ in obs_list]
    else:
        noises_b = None

    print("[probe] FP16 pass: paired trajectories for global D_solver (+ guard reference outputs) ...")
    t0 = time.time()
    fp_traj = run_rollouts(model_fp, policy_fp, obs_list[: args.n_obs], noises[: args.n_obs], args.batch_size)
    # v1.3: also collect per-layer FP16 outputs (guards are measured vs pure FP16)
    fp_col = _LayerCollector(target_names, mode="q", banks={}, max_tokens=args.max_tokens)
    fp_col.install(model_fp)
    run_activations(model_fp, policy_fp, obs_list[: args.n_obs], noises[: args.n_obs], args.batch_size)
    fp_out = {n: fp_col.pooled(n) for n in target_names}
    fp_col.remove()
    n_fp_out = sum(1 for v in fp_out.values() if v is not None)
    print(f"[probe] FP16 pass done in {time.time() - t0:.1f}s; fp_traj {tuple(fp_traj.shape)} "
          f"(T+1 states); guard refs collected {n_fp_out}/{len(target_names)}")

    del model_fp, policy_fp
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    restore_quant_env(saved_env)

    # ---------------------------------------------------------------- quant load
    set_quant_env(
        args.include,
        args.exclude,
        args.packdir,
        bits_default=args.bits[0],
        group=args.group,
        ls=args.ls,
        act_pct=args.act_pct,
        calib_steps=args.calib_steps,
        row_rot=args.row_rot,
        act_dynamic=False,  # base mode: STATIC act scale
    )
    policy_q = load_policy(args.model_path, data_config=args.data_config, denoising_steps=args.denoising_steps, device=args.device, embodiment_tag=args.embodiment_tag)
    model_q = policy_q.model
    n_wrapped = set_all_bits(model_q, args.bits[0])
    print(f"[probe] quant model loaded, DuQuantLinear wrapped: {n_wrapped}")

    # P0-2 (correctness review): the static A8 calibration must accumulate
    # cfg.calib_batches REAL batches (each forward pass = one observe per
    # layer) IN THE ALL-ZERO REFERENCE STATE, and only then freeze. The old
    # code froze on the first batch AND calibrated under bits[0] (W4) — both
    # wrong for the reference protocol.
    set_all_bits(model_q, REF_BITS)
    from gr00t.quantization.duquant_layers import all_calibrated, calibration_progress

    n_warm_batches = args.calib_steps  # GR00T_DUQUANT_CALIB_STEPS = batch count
    n_warm_obs = n_warm_batches * args.batch_size
    print(f"[probe] A8 calibration: {n_warm_obs} obs = {n_warm_batches} batches "
          f"(state = weight_bits=0 reference) ...")
    t0 = time.time()
    warm_obs = [make_obs(rng, args.obs_format) for _ in range(n_warm_obs)]
    warm_noises = [torch.randn(horizon, action_dim) for _ in warm_obs]
    run_rollouts(model_q, policy_q, warm_obs, warm_noises, args.batch_size, return_trajectory=False)
    full, total = calibration_progress(model_q)
    if not all_calibrated(model_q):
        raise SystemExit(
            f"[probe] A8 calibration incomplete: {full}/{total} layers full after "
            f"{n_warm_batches} batches — calibration state invalid for measurement"
        )
    print(f"[probe] A8 calibration done in {time.time() - t0:.1f}s ({full}/{total} layers frozen)")

    from gr00t.quantization.kernel_scores import LayerScoreBank
    from gr00t.atm import clear_atm_capture, register_output_capture

    banks: Dict[str, LayerScoreBank] = {}
    for n in target_names:
        banks[n] = LayerScoreBank(n, max_tokens=args.max_tokens)
    for n in attn_names:
        banks[f"attn:{n}"] = LayerScoreBank(f"attn:{n}", max_tokens=args.max_tokens)

    results: Dict[str, Any] = {
        "meta": {
            "suite": args.suite,
            "model_path": args.model_path,
            "bits": args.bits,
            "group": args.group,
            "n_obs": args.n_obs,
            "n_rollout_obs": args.n_rollout_obs,
            "n_noises_per_obs": args.n_noises_per_obs,
            "layers_subset": args.layers_subset,
            "batch_size": args.batch_size,
            "max_tokens": args.max_tokens,
            "gamma": args.gamma,
            "denoising_steps": args.denoising_steps,
            "calib_steps": args.calib_steps,
            "per_layer_bits": args.per_layer_bits,
            "obs_source": "L1 synthetic (data-free)",
            "token_mix": "all_tokens_flattened_stride_subsampled_cap1024 (padding NOT masked)",
            "reference_protocol": "wrapped pipeline, all target layers weight_bits=0, A8 static act, rotations active",
            "base_mode": "ATM OFF, OHB OFF, per-step OFF, static activation scale",
            "global_dsolver_pairing": "pure FP16 model vs full config",
            "guard_reference": "pure FP16 model (deployment pairing); D_sat proxied by P99.9(|fp_out|)/127",
            "guard_thresholds": None,  # filled at the end: τ = P99(W4 candidates) × margin
        },
        "layers": {n: {} for n in banks},
        "global": {},
    }

    def save_incremental() -> None:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[probe] saved -> {args.out}")

    # ---- REFERENCE pass (all-0): 单次配对噪声前向，激活参照 + 参照轨迹同源 ----
    print("[probe] REFERENCE pass (all layers weight_bits=0, paired noise) ...")
    t0 = time.time()
    set_all_bits(model_q, REF_BITS)
    lin_ref = _LayerCollector(target_names, mode="ref", banks=banks, max_tokens=args.max_tokens)
    attn_ref = _AttentionCollector(mode="ref", banks=banks, max_tokens=args.max_tokens)
    lin_ref.install(model_q)
    register_output_capture(model_q, attn_ref, scope="dit")
    # 只用配对噪声跑一次：激活参照与 ref_traj 来自同一条轨迹
    ref_traj = run_rollouts(model_q, policy_q, obs_list[: args.n_obs], noises[: args.n_obs], args.batch_size)
    lin_ref.remove()
    clear_atm_capture(model_q)
    for bank in banks.values():
        bank.finalize_ref()
    n_ready = sum(1 for b in banks.values() if b.ready)
    print(f"[probe] REFERENCE pass done in {time.time() - t0:.1f}s; banks ready {n_ready}/{len(banks)}")

    # P0-3: the second paired-noise set needs its OWN reference trajectory
    # (D(x^R(ε_b), x^Q(ε_b))), computed while every layer is still at
    # weight_bits=0 and the collectors are removed (banks already finalized).
    ref_traj_b = None
    if noises_b is not None:
        ref_traj_b = run_rollouts(
            model_q, policy_q, obs_list[: args.n_rollout_obs],
            noises_b[: args.n_rollout_obs], args.batch_size
        )
        print(f"[probe] reference trajectory for noise set B: {tuple(ref_traj_b.shape)}")

    # ---- v1.3 CS in-situ scaling check (closing criterion, §5.1.2) ----
    if args.cs_in_situ_check:
        check_bank = next((b for b in banks.values() if b.ready and b._fp_raw is not None), None)
        if check_bank is None:
            print("[probe] CS in-situ check: no ready bank, skipped")
            results["meta"]["cs_in_situ_check"] = {"status": "skipped"}
        else:
            vals = {}
            base = check_bank.evaluate(check_bank._fp_raw)
            for c in (2.0, 4.0, 8.0):
                r = check_bank.evaluate(check_bank._fp_raw * c)
                vals[str(c)] = {"cs_cross": r.get("cs_cross"), "cs": r.get("cs")}
            crosses = [vals[str(c)]["cs_cross"] for c in (2.0, 4.0, 8.0) if vals[str(c)]["cs_cross"] is not None]
            monotonic = all(a < b for a, b in zip(crosses, crosses[1:]))
            results["meta"]["cs_in_situ_check"] = {
                "layer": check_bank.name, "base_cs_cross": base.get("cs_cross"),
                "scaled": vals, "cross_monotonic": monotonic,
            }
            print(f"[probe] CS in-situ check ({check_bank.name}): cross monotonic = {monotonic}, {vals}")
            if not monotonic:
                print("[probe] WARNING: CS cross term does not respond to output scaling — "
                      "selector should disable CS (λ_cs=0) and report the negative result (§5.1.2)")

    # ---- per-layer CKA/CS + guards attribution (single-layer intervention vs R) ----
    if not args.skip_per_layer:
        attr_names = subset_names(target_names, args.layers_subset)
        if len(attr_names) != len(target_names):
            print(f"[probe] --layers-subset {args.layers_subset}: attribution on "
                  f"{len(attr_names)}/{len(target_names)} layers")
        for b in args.bits:
            t0 = time.time()
            set_all_bits(model_q, REF_BITS)
            for name in attr_names:
                set_single_layer_bits(model_q, name, b)
                col = _LayerCollector([name], mode="q", banks=banks, max_tokens=args.max_tokens)
                col.install(model_q)
                run_activations(model_q, policy_q, obs_list[: args.n_obs], noises[: args.n_obs], args.batch_size)
                set_single_layer_bits(model_q, name, REF_BITS)
                pooled = col.pooled(name)
                col.remove()
                scores = banks[name].evaluate(pooled) if pooled is not None else {"cka": None, "cs": None, "cs_cross": None}
                # v1.3 feasibility guards vs the pure FP16 reference
                guards = guard_metrics(fp_out.get(name), pooled)
                scores.update(guards)
                results["layers"][name][f"b{b}"] = scores
            save_incremental()
            print(f"[probe] per-layer attribution b={b} done in {time.time() - t0:.1f}s")

    # ---- attention-level CKA/CS under global configs (vs R) ----
    for b in args.bits:
        t0 = time.time()
        set_all_bits(model_q, b)
        attn_col = _AttentionCollector(mode="q", banks=banks, max_tokens=args.max_tokens)
        register_output_capture(model_q, attn_col, scope="dit")
        run_activations(model_q, policy_q, obs_list[: args.n_obs], noises[: args.n_obs], args.batch_size)
        for aname in attn_names:
            key = f"attn:{aname}"
            pooled = attn_col.pooled(key)
            results["layers"][key][f"b{b}"] = (
                banks[key].evaluate(pooled) if pooled is not None else {"cka": None, "cs": None}
            )
        clear_atm_capture(model_q)
        save_incremental()
        print(f"[probe] attention b={b} done in {time.time() - t0:.1f}s")

    # ---- global D_solver (FP16 vs full config; config-level, NOT additive) ----
    for b in args.bits:
        t0 = time.time()
        set_all_bits(model_q, b)
        q_traj = run_rollouts(model_q, policy_q, obs_list[: args.n_obs], noises[: args.n_obs], args.batch_size)
        mean_div, per_obs = solver_divergence(fp_traj, q_traj, args.gamma)
        results["global"][f"b{b}"] = {
            "d_solver": mean_div,
            "d_solver_std": float(np.std(per_obs)) if len(per_obs) > 1 else 0.0,
            "per_obs": per_obs,
        }
        del q_traj
        gc.collect()
        save_incremental()
        print(f"[probe] global D_solver b={b} done in {time.time() - t0:.1f}s")

    # ---- per-layer D_solver importance (single-layer intervention vs R;
    #      v1.3: n_rollout_obs obs × n_noises_per_obs paired noises, median) ----
    if not args.skip_layer_rollouts:
        w_names = subset_names(target_names, args.layers_subset)
        if len(w_names) != len(target_names):
            print(f"[probe] --layers-subset {args.layers_subset}: importance rollouts on "
                  f"{len(w_names)}/{len(target_names)} layers")
        for b in args.per_layer_bits:
            for name in w_names:
                set_all_bits(model_q, REF_BITS)
                if not set_single_layer_bits(model_q, name, b):
                    continue
                all_per_obs: List[float] = []
                # noise set A: paired against the main reference trajectory
                ref_sub_a = ref_traj[:, : args.n_rollout_obs]
                q_traj = run_rollouts(
                    model_q, policy_q, obs_list[: args.n_rollout_obs],
                    noises[: args.n_rollout_obs], args.batch_size
                )
                _, per_obs_a = solver_divergence(ref_sub_a, q_traj, args.gamma)
                all_per_obs.extend(per_obs_a)
                del q_traj
                # noise set B: paired against ITS OWN reference trajectory
                # (P0-3: D(x^R(ε_b), x^Q(ε_b)) — the old code compared ε_b
                # quantized rollouts against the ε_a reference, which measured
                # noise mismatch, not quantization)
                if noises_b is not None and ref_traj_b is not None:
                    q_traj_b = run_rollouts(
                        model_q, policy_q, obs_list[: args.n_rollout_obs],
                        noises_b[: args.n_rollout_obs], args.batch_size
                    )
                    _, per_obs_b = solver_divergence(ref_traj_b, q_traj_b, args.gamma)
                    all_per_obs.extend(per_obs_b)
                    del q_traj_b
                gc.collect()
                # median over (obs, noise) — robust aggregation, std recorded
                mean_div = float(np.median(all_per_obs))
                std_div = float(np.std(all_per_obs)) if len(all_per_obs) > 1 else 0.0
                results["layers"][name][f"d_solver_b{b}"] = mean_div
                results["layers"][name][f"d_solver_b{b}_std"] = std_div
            save_incremental()
            print(f"[probe] per-layer importance rollouts b={b} done")

    # ---- v1.3 guard thresholds: τ = P99 of the measured W4 candidates × margin ----
    if not args.skip_per_layer:
        guard_bit = 4 if 4 in args.bits else args.bits[0]
        rms_vals: List[float] = []
        sat_vals: List[float] = []
        for n in target_names:
            entry = results["layers"].get(n, {}).get(f"b{guard_bit}", {})
            for key, lst in (("rms_ratio", rms_vals), ("sat_rate", sat_vals)):
                v = entry.get(key)
                if v is not None:
                    lst.append(float(v))

        def _p99(vals: List[float]) -> Optional[float]:
            if not vals:
                return None
            s = sorted(vals)
            idx = max(0, min(len(s) - 1, int(np.ceil(0.99 * len(s))) - 1))
            return s[idx] * args.guard_margin

        results["meta"]["guard_thresholds"] = {
            "tau_rms": _p99(rms_vals),
            "tau_sat": _p99(sat_vals),
            "guard_bit": guard_bit,
            "estimation": f"P99 of W4 candidates × {args.guard_margin}",
            "n_candidates": len(rms_vals),
        }
        print(f"[probe] guard thresholds: {results['meta']['guard_thresholds']}")

    save_incremental()
    print("[probe] done.")


if __name__ == "__main__":
    main()
