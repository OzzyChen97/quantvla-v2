#!/usr/bin/env python3
"""P2-G: data-free per-step ATM/OHB calibration for GR00T (QuantVLA v2).

Collects FP16 vs quantized per-head attention-logit std and per-head output RMS
grouped by denoising step (key = t_discretized, same synthetic L1 obs and the
SAME paired noise in both passes), computes per-step α/β and writes the v2 JSON
schema consumed by dit_atm.py with GR00T_ATM_PER_STEP=1:

    {"layer": {
        "all": [per-head α pooled across steps],          # v1 static fallback
        "beta_perhead": [per-head β pooled across steps],
        "steps": {
            "0":   {"all": [...], "beta_perhead": [...]},
            "125": {"all": [...], "beta_perhead": [...]}, ...}}}

Equivalence property (regression guard): if every "steps" entry equals the
pooled "all"/"beta_perhead", the runtime output is bit-identical to v1 static
ATM/OHB (missing-step lookup falls back to the static value).

Usage (groot_test env, one idle GPU):

    cd /home1/gyy/vla/QuantVLA
    export PYTHONPATH=/home1/gyy/vla/QuantVLA/code:$PYTHONPATH
    python scripts/tools/calibrate_atm_perstep_gr00t.py --suite spatial

Then serve with:

    export GR00T_DUQUANT_PACKDIR=checkpoints/packs/gr00t/duquant_packed_libero_spatial_w4a8_b64c32ls015
    export GR00T_ATM_ENABLE=1 GR00T_ATM_ALPHA_PATH=checkpoints/packs/gr00t/atm_alpha_beta_perstep_spatial.json
    export GR00T_ATM_PER_STEP=1
    export GR00T_OHB_ENABLE=1
    ./scripts/run_quantvla.sh libero_spatial
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
    PACKDIR_TEMPLATE,
    SUITE_DIRS,
    chunked,
    ensure_flash_attn_rpath,
    load_policy,
    make_l1_obs,
    make_noises,
    set_quant_env,
    strip_quant_env,
)


# --------------------------------------------------------------------------- #
# Step-aware statistics collectors
# --------------------------------------------------------------------------- #
class StepStdCollector:
    """Per-layer per-step per-head logits-std aggregator (FP or quant pass)."""

    def __init__(self) -> None:
        self.by_step: Dict[str, Dict[int, torch.Tensor]] = {}
        self.counts: Dict[str, Dict[int, int]] = {}
        self.pooled: Dict[str, torch.Tensor] = {}
        self.pooled_counts: Dict[str, int] = {}

    def __call__(self, layer: str, std: torch.Tensor, step: Optional[int]) -> None:
        if step is None:
            return
        std = std.mean(dim=0).detach().to(torch.float32).cpu()
        s = self.by_step.setdefault(layer, {})
        c = self.counts.setdefault(layer, {})
        s[step] = s.get(step, torch.zeros_like(std)) + std
        c[step] = c.get(step, 0) + 1
        p = self.pooled.setdefault(layer, torch.zeros_like(std))
        p += std
        self.pooled_counts[layer] = self.pooled_counts.get(layer, 0) + 1

    def finalize(self) -> Dict[str, Dict[int, torch.Tensor]]:
        out = {}
        for layer, s in self.by_step.items():
            out[layer] = {t: s[t] / max(self.counts[layer][t], 1) for t in s}
        return out

    def finalize_pooled(self) -> Dict[str, torch.Tensor]:
        return {layer: self.pooled[layer] / max(self.pooled_counts[layer], 1) for layer in self.pooled}


class StepRmsCollector:
    """Per-layer per-step per-head output-RMS aggregator (FP or quant pass)."""

    def __init__(self) -> None:
        self.by_step: Dict[str, Dict[int, torch.Tensor]] = {}
        self.counts: Dict[str, Dict[int, int]] = {}
        self.pooled: Dict[str, torch.Tensor] = {}
        self.pooled_counts: Dict[str, int] = {}

    def __call__(self, layer: str, rms: torch.Tensor, step: Optional[int]) -> None:
        if step is None:
            return
        rms = rms.detach().to(torch.float32).cpu()
        s = self.by_step.setdefault(layer, {})
        c = self.counts.setdefault(layer, {})
        s[step] = s.get(step, torch.zeros_like(rms)) + rms
        c[step] = c.get(step, 0) + 1
        p = self.pooled.setdefault(layer, torch.zeros_like(rms))
        p += rms
        self.pooled_counts[layer] = self.pooled_counts.get(layer, 0) + 1

    def finalize(self) -> Dict[str, Dict[int, torch.Tensor]]:
        out = {}
        for layer, s in self.by_step.items():
            out[layer] = {t: s[t] / max(self.counts[layer][t], 1) for t in s}
        return out

    def finalize_pooled(self) -> Dict[str, torch.Tensor]:
        return {layer: self.pooled[layer] / max(self.pooled_counts[layer], 1) for layer in self.pooled}


# --------------------------------------------------------------------------- #
# Collection passes
# --------------------------------------------------------------------------- #
def run_collection_pass(
    policy: Any,
    obs_list: List[Dict[str, Any]],
    noises: List[torch.Tensor],
    batch_size: int,
    use_autocast: bool,
) -> tuple:
    from gr00t.atm import register_atm_capture_step, register_ohb_perhead_capture_step

    model = policy.model
    std_col = StepStdCollector()
    rms_col = StepRmsCollector()
    register_atm_capture_step(model, std_col, scope="dit")
    register_ohb_perhead_capture_step(model, rms_col, scope="dit")

    for batched_obs, batched_noise in chunked(obs_list, noises, batch_size):
        norm = policy.apply_transforms(batched_obs)
        with torch.inference_mode():
            if use_autocast:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    model.get_action(norm, action_noise=batched_noise)
            else:
                model.get_action(norm, action_noise=batched_noise)
    return std_col, rms_col


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P2-G data-free per-step ATM/OHB calibration (GR00T)")
    p.add_argument("--suite", default="spatial", choices=["spatial", "goal", "object", "90", "10"])
    p.add_argument("--model-path", default=None)
    p.add_argument("--data-config", default="examples.Libero.custom_data_config:LiberoDataConfig")
    p.add_argument("--device", default="cuda")
    p.add_argument("--denoising-steps", type=int, default=8)
    p.add_argument("--n-obs", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--group", type=int, default=64)
    p.add_argument("--ls", type=float, default=0.15)
    p.add_argument("--act-pct", type=float, default=99.9)
    p.add_argument("--act-dynamic", action="store_true",
                   help="Calibrate with the SAME dynamic activation mode as deployment "
                        "(GR00T_DUQUANT_ACT_DYNAMIC=1). MUST match the deployment flag, "
                        "otherwise alpha/beta are calibrated on a different act mode.")
    p.add_argument("--row-rot", default="restore")
    p.add_argument("--calib-steps", type=int, default=32)
    p.add_argument("--include", default=DEFAULT_INCLUDE)
    p.add_argument("--exclude", default=DEFAULT_EXCLUDE)
    p.add_argument("--packdir", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--alpha-min", type=float, default=0.7)
    p.add_argument("--alpha-max", type=float, default=1.4)
    p.add_argument("--alpha-neutral", type=float, default=0.02)
    p.add_argument("--beta-log-clamp", type=float, default=0.30)
    p.add_argument("--beta-neutral", type=float, default=0.03)
    p.add_argument("--scope", default="dit")
    p.add_argument("--selftest", action="store_true", help="Unit-test the per-step builders and exit.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.selftest:
        from gr00t.atm import compute_per_step_alpha, compute_per_step_beta

        # identical teacher/quant -> alpha=1, beta=1 everywhere
        t = {0: torch.tensor([1.0, 2.0]), 125: torch.tensor([2.0, 1.0])}
        q = {k: v.clone() for k, v in t.items()}
        a = compute_per_step_alpha(t, q)
        b = compute_per_step_beta(t, q)
        assert all(abs(x - 1.0) < 1e-6 for x in a["all"]), a
        for st in a["steps"].values():
            assert all(abs(x - 1.0) < 1e-6 for x in st["all"]), st
        assert all(abs(x - 1.0) < 1e-6 for x in b["beta_perhead"]), b
        # divergent teacher -> alpha != 1, per-step entries differ from pooled
        t2 = {0: torch.tensor([2.0]), 125: torch.tensor([1.0])}
        q2 = {0: torch.tensor([1.0]), 125: torch.tensor([1.0])}
        a2 = compute_per_step_alpha(t2, q2)
        assert a2["steps"]["0"]["all"] != a2["steps"]["125"]["all"], a2
        assert abs(a2["steps"]["0"]["all"][0] - 1.4) < 1e-5  # clamped
        print("[calibrate-perstep] selftest OK")
        return

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
        args.out = str(REPO_ROOT / "checkpoints/packs/gr00t" / f"atm_alpha_beta_perstep_{args.suite}.json")

    ensure_flash_attn_rpath()
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    print("=" * 100)
    print("[calibrate-perstep] P2-G per-step ATM/OHB calibration (data-free, FP16 reference only)")
    print(f"[calibrate-perstep] model={args.model_path} n_obs={args.n_obs} batch={args.batch_size}")
    print(f"[calibrate-perstep] out={args.out}")
    print("=" * 100)

    obs_list = [make_l1_obs(rng) for _ in range(args.n_obs)]

    # ---------------- teacher (FP16) ----------------
    saved_env = strip_quant_env()
    policy_fp = load_policy(args.model_path, data_config=args.data_config, denoising_steps=args.denoising_steps, device=args.device)
    noises = make_noises(policy_fp.model, args.n_obs, seed=0)
    use_autocast = str(policy_fp.device).startswith("cuda")

    print("[calibrate-perstep] FP16 pass ...")
    t0 = time.time()
    std_fp, rms_fp = run_collection_pass(policy_fp, obs_list, noises, args.batch_size, use_autocast)
    print(f"[calibrate-perstep] FP16 pass done in {time.time() - t0:.1f}s; "
          f"{len(std_fp.by_step)} attention layers captured")
    del policy_fp
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---------------- quantized ----------------
    # v1.2: the quant pass must run in the SAME act mode as deployment so that
    # alpha/beta are calibrated against the right activation statistics.
    set_quant_env(
        args.include, args.exclude, args.packdir,
        bits_default=4, group=args.group, ls=args.ls,
        act_pct=args.act_pct, calib_steps=args.calib_steps, row_rot=args.row_rot,
        act_dynamic=args.act_dynamic,
    )
    policy_q = load_policy(args.model_path, data_config=args.data_config, denoising_steps=args.denoising_steps, device=args.device)
    use_autocast = str(policy_q.device).startswith("cuda")

    print(f"[calibrate-perstep] warmup {args.calib_steps} forwards for activation calibration ...")
    warm_obs = [make_l1_obs(rng) for _ in range(args.calib_steps)]
    warm_noises = make_noises(policy_q.model, args.calib_steps, seed=1)
    for batched_obs, batched_noise in chunked(warm_obs, warm_noises, args.batch_size):
        norm = policy_q.apply_transforms(batched_obs)
        with torch.inference_mode():
            if use_autocast:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    policy_q.model.get_action(norm, action_noise=batched_noise)
            else:
                policy_q.model.get_action(norm, action_noise=batched_noise)

    print("[calibrate-perstep] quant pass ...")
    t0 = time.time()
    std_q, rms_q = run_collection_pass(policy_q, obs_list, noises, args.batch_size, use_autocast)
    print(f"[calibrate-perstep] quant pass done in {time.time() - t0:.1f}s")

    # ---------------- per-step alpha/beta ----------------
    from gr00t.atm import compute_per_step_alpha, compute_per_step_beta

    std_fp_bs = std_fp.finalize()
    std_q_bs = std_q.finalize()
    rms_fp_bs = rms_fp.finalize()
    rms_q_bs = rms_q.finalize()

    data: Dict[str, Dict[str, Any]] = {}
    for layer in sorted(set(std_fp_bs) & set(std_q_bs)):
        alpha_part = compute_per_step_alpha(
            std_fp_bs[layer], std_q_bs[layer],
            min_alpha=args.alpha_min, max_alpha=args.alpha_max, neutral_threshold=args.alpha_neutral,
        )
        beta_part = compute_per_step_beta(
            rms_fp_bs.get(layer, {}), rms_q_bs.get(layer, {}),
            log_clamp=args.beta_log_clamp, neutral=args.beta_neutral,
        )
        steps: Dict[str, Dict[str, Any]] = {}
        for t in sorted(set(alpha_part["steps"]) | set(beta_part["steps"])):
            steps[t] = {
                "all": alpha_part["steps"].get(t, {}).get("all"),
                "beta_perhead": beta_part["steps"].get(t, {}).get("beta_perhead"),
            }
        data[layer] = {
            "all": alpha_part["all"],
            "beta_perhead": beta_part["beta_perhead"],
            "steps": steps,
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    n_steps = {len(entry["steps"]) for entry in data.values()}
    print(f"[calibrate-perstep] saved {len(data)} layers -> {args.out} (steps per layer: {n_steps})")
    print("[calibrate-perstep] serve with:")
    print(f"  export GR00T_ATM_ENABLE=1 GR00T_ATM_ALPHA_PATH={args.out}")
    print("  export GR00T_ATM_PER_STEP=1")
    print("  export GR00T_OHB_ENABLE=1")
    print("[calibrate-perstep] done.")


if __name__ == "__main__":
    main()
