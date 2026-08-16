#!/usr/bin/env python3
"""A/B test: Triton fused W4-dequant matmul vs eager fake-quant path (v1.4 probe).

Question: does the fused fast path change RESULTS? Loads a real deployment
plan (GR00T_DUQUANT_PLAN), runs the SAME paired obs/noises through the eager
and fused paths, and reports:

  * per-layer output divergence (max rel diff),
  * final-action divergence (max rel diff over the action chunk),
  * D_solver(fused vs eager) — the config-level functional metric,
  * per-rollout wall latency for both paths.

Usage (groot_test env, one idle GPU):
    python scripts/tools/ab_duquant_fused.py \
        --ckpt checkpoints/gr00t/libero-spatial --suite spatial \
        --plan checkpoints/packs/gr00t/gr00t_quant_plan_libero_spatial_adjudicated.final_plan.json \
        --packdir checkpoints/packs/gr00t/duquant_packed_libero_spatial_w4a8_b64c32ls015 \
        --n-obs 16
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
    chunked,
    ensure_a8_calibrated,
    ensure_flash_attn_rpath,
    fixed_calibration_buffer,
    load_policy,
    make_obs,
    resolve_data_config,
    set_quant_env,
    strip_quant_env,
)
from gr00t_sensitivity_probe import discover_targets, run_rollouts, solver_divergence  # noqa: E402


class _Hook:
    def __init__(self, names: List[str]):
        self.names = set(names)
        self.out: Dict[str, torch.Tensor] = {}
        self.handles = []

    def _fn(self, name):
        def hook(module, args, output):
            if isinstance(output, tuple):
                output = output[0]
            self.out[name] = output.detach().float().cpu()
        return hook

    def install(self, model):
        self.remove()
        for name, mod in model.named_modules():
            if name in self.names:
                self.handles.append(mod.register_forward_hook(self._fn(name)))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def run_path(ckpt: str, suite: str, plan: str, packdir: str, obs_list, noises,
             batch: int, fused: bool, names: List[str], device: str):
    import os

    strip_quant_env()
    os.environ["GR00T_DUQUANT_FUSED"] = "1" if fused else "0"
    set_quant_env(DEFAULT_INCLUDE, DEFAULT_EXCLUDE, packdir, bits_default=4,
                  group=64, ls=0.15, act_pct=99.9, calib_steps=32,
                  row_rot="restore", act_dynamic=False)
    os.environ["GR00T_DUQUANT_PLAN"] = plan
    policy = load_policy(ckpt, data_config=resolve_data_config(suite, None),
                         denoising_steps=8, device=device)
    horizon = int(policy.model.action_head.config.action_horizon)
    action_dim = int(policy.model.action_head.config.action_dim)
    warm_obs, warm_noises, warm_sha = fixed_calibration_buffer(
        0, 32 * batch, horizon, action_dim, fmt="libero")
    ensure_a8_calibrated(policy, warm_obs, warm_noises, batch, act_dynamic=False,
                         expected_wrapped=None, act_scale_path=None,
                         act_scale_meta={"buffer_sha256": warm_sha})
    hook = _Hook(names)
    hook.install(policy.model)
    t0 = time.time()
    traj = run_rollouts(policy.model, policy, obs_list, noises, batch)
    dt = time.time() - t0
    hook.remove()
    # fp16 reference for D_solver (same obs/noises)
    strip_quant_env()
    policy_fp = load_policy(ckpt, data_config=resolve_data_config(suite, None),
                            denoising_steps=8, device=device)
    fp_traj = run_rollouts(policy_fp.model, policy_fp, obs_list, noises, batch)
    d_mean, _ = solver_divergence(fp_traj, traj, gamma=1.2)
    n_fused = sum(1 for m in policy.model.modules()
                  if getattr(m, "_fused_ready", False))
    out = {"traj": traj, "layers": hook.out, "d_solver": d_mean, "sec": dt,
           "n_fused_ready": n_fused}
    del policy, policy_fp, fp_traj
    gc.collect()
    torch.cuda.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--suite", default="spatial")
    ap.add_argument("--plan", required=True)
    ap.add_argument("--packdir", required=True)
    ap.add_argument("--n-obs", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ensure_flash_attn_rpath()
    rng = np.random.default_rng(0)
    obs_list = [make_obs(rng, "libero") for _ in range(args.n_obs)]
    # paired noises (same for both paths)
    policy_fp = load_policy(args.ckpt, data_config=resolve_data_config(args.suite, None),
                            device=args.device)
    horizon = int(policy_fp.model.action_head.config.action_horizon)
    action_dim = int(policy_fp.model.action_head.config.action_dim)
    del policy_fp
    gc.collect()
    torch.cuda.empty_cache()
    noises = [torch.randn(horizon, action_dim) for _ in obs_list]

    names = discover_targets_pure(args.ckpt, args.suite, args.device)
    print(f"[ab-fused] {len(names)} target layers; plan={args.plan}")
    eager = run_path(args.ckpt, args.suite, args.plan, args.packdir, obs_list, noises,
                     args.batch_size, fused=False, names=names, device=args.device)
    fused = run_path(args.ckpt, args.suite, args.plan, args.packdir, obs_list, noises,
                     args.batch_size, fused=True, names=names, device=args.device)

    # per-layer divergence — ENERGY-weighted (RMS): the elementwise max-rel
    # metric explodes on near-zero entries and misrepresents a bf16-accumulation
    # difference as catastrophic; RMS rel error is the honest metric.
    layer_rms = {}
    layer_max = {}
    for n in names:
        a = eager["layers"].get(n)
        b = fused["layers"].get(n)
        if a is None or b is None:
            continue
        num = ((a - b) ** 2).sum()
        den = (a ** 2).sum()
        layer_rms[n] = float((num / den.clamp_min(1e-12)) ** 0.5)
        layer_max[n] = float(((a - b).abs() / (a.abs() + 1e-2)).max())
    rms = sorted(layer_rms.values(), reverse=True)
    print(f"[ab-fused] layers compared: {len(rms)}")
    print(f"[ab-fused] per-layer RMS rel diff: max={rms[0]:.3e} "
          f"p90={np.percentile(rms, 90):.3e} median={np.percentile(rms, 50):.3e}")
    print(f"[ab-fused] per-layer max-rel diff (near-zero inflated): "
          f"max={max(layer_max.values()):.3e} median={np.percentile(sorted(layer_max.values()), 50):.3e}")
    # final action divergence — energy-weighted
    ta, tb = eager["traj"][-1].float(), fused["traj"][-1].float()
    act_rms = float((((ta - tb) ** 2).sum() / (ta ** 2).sum().clamp_min(1e-12)) ** 0.5)
    act_max = float(((ta - tb).abs() / (ta.abs() + 1e-3)).max())
    print(f"[ab-fused] final-action RMS rel diff: {act_rms:.3e} "
          f"(max-rel {act_max:.3e}, near-zero inflated)")
    print(f"[ab-fused] D_solver eager={eager['d_solver']:.5f} fused={fused['d_solver']:.5f} "
          f"(delta {fused['d_solver'] - eager['d_solver']:+.2e})")
    print(f"[ab-fused] fused_ready layers: eager={eager.get('n_fused_ready')} "
          f"fused={fused.get('n_fused_ready')}")
    print(f"[ab-fused] wall per rollout: eager {eager['sec']:.1f}s fused {fused['sec']:.1f}s "
          f"({eager['sec'] / fused['sec']:.2f}x)")
    report = {
        "plan": args.plan, "n_obs": args.n_obs,
        "per_layer_rms_rel_diff": {"max": rms[0] if rms else None,
                                   "p90": float(np.percentile(rms, 90)) if rms else None,
                                   "median": float(np.percentile(rms, 50)) if rms else None},
        "final_action_rms_rel_diff": act_rms,
        "d_solver": {"eager": eager["d_solver"], "fused": fused["d_solver"]},
        "wall_sec": {"eager": eager["sec"], "fused": fused["sec"]},
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"[ab-fused] saved -> {args.out}")
    print("[ab-fused] done.")


def discover_targets_pure(ckpt: str, suite: str, device: str) -> List[str]:
    """Discover target layers on the pure FP16 model (same as the probe)."""
    strip_quant_env()
    policy_fp = load_policy(ckpt, data_config=resolve_data_config(suite, None), device=device)
    names = discover_targets(policy_fp.model,
                             argparse.Namespace(include=DEFAULT_INCLUDE, exclude=DEFAULT_EXCLUDE))
    del policy_fp
    gc.collect()
    torch.cuda.empty_cache()
    return names


if __name__ == "__main__":
    main()
