#!/usr/bin/env python3
"""Shared utilities for the GR00T QuantVLA v2 tools (P0-G probe / P2-G calibrator).

Moved here from gr00t_sensitivity_probe.py so the per-step ATM/OHB calibrator
(calibrate_atm_perstep_gr00t.py) can reuse the exact same synthetic-obs format
and quant env handling. Nothing here touches pi0.5.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "code") not in os.environ.get("PYTHONPATH", ""):
    os.environ["PYTHONPATH"] = str(REPO_ROOT / "code") + os.pathsep + os.environ.get("PYTHONPATH", "")

# v1 eval defaults (run_quantvla.sh): LLM all projections + DiT ff only = 116 layers
DEFAULT_INCLUDE = (
    r".*(backbone\.eagle_model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj"
    r"|gate_proj|up_proj|down_proj)"
    r"|action_head\.model\.transformer_blocks\.\d+\.ff\.net\.(0\.proj|2)).*"
)
DEFAULT_EXCLUDE = r"(?:^|\.)(vision|radio|norm|ln|layernorm|embed|lm_head|attn1)(?:\.|$)"

PACKDIR_TEMPLATE = "duquant_packed_libero_{suite}_w4a8_b{g}c{calib}ls{ls}"

LANG_INSTRUCTION = "pick up the black bowl between the plate and the ramekin and place it on the plate"

SUITE_DIRS = {
    "spatial": "libero-spatial",
    "goal": "libero-goal",
    "object": "libero-object",
    "90": "libero-90",
    "10": "libero-long",
    # v1.4 Stage D: RoboCasa365 GR00T checkpoints (HF robocasa/robocasa365_checkpoints)
    "robocasa365_atomic": "../robocasa365/gr00t_n1-5/foundation_model_learning/target_posttraining/atomic_seen/checkpoint-60000",
}


# --------------------------------------------------------------------------- #
# Synthetic observations (data-free L1: only reference is the FP16 model)
# --------------------------------------------------------------------------- #
def make_l1_obs(rng: np.random.Generator) -> Dict[str, Any]:
    """Random-image + random-state observation in LIBERO GR00T format (per-obs, unbatched)."""
    return {
        # videos: (T, H, W, C) per obs (batched stack adds the B dim)
        "video.image": rng.integers(0, 256, (1, 256, 256, 3), dtype=np.uint8),
        "video.wrist_image": rng.integers(0, 256, (1, 256, 256, 3), dtype=np.uint8),
        # states: (T, D) per obs (batched stack adds the B dim)
        "state.x": rng.uniform(-0.35, 0.35, (1, 1)).astype(np.float32),
        "state.y": rng.uniform(-0.35, 0.35, (1, 1)).astype(np.float32),
        "state.z": rng.uniform(0.6, 1.2, (1, 1)).astype(np.float32),
        "state.roll": rng.uniform(-3.14, 3.14, (1, 1)).astype(np.float32),
        "state.pitch": rng.uniform(-3.14, 3.14, (1, 1)).astype(np.float32),
        "state.yaw": rng.uniform(-3.14, 3.14, (1, 1)).astype(np.float32),
        "state.gripper": rng.uniform(0.0, 1.0, (1, 2)).astype(np.float32),
        "annotation.human.action.task_description": [LANG_INSTRUCTION],
    }


GR1_LANG = "place the cup into the drawer and close it"


ROBOCASA365_LANG = "add ice cubes to the blender"


def make_robocasa365_obs(rng: np.random.Generator) -> Dict[str, Any]:
    """Random RoboCasa365 observation in GR00T format (per-obs, unbatched).

    Mirrors the wrapper obs layout consumed by RoboCasa365DataConfig: three
    256x256 cameras (T,H,W,C uint8) + the five state groups (T,D float32) from
    the checkpoint's own statistics.
    """
    return {
        "video.robot0_agentview_left": rng.integers(0, 256, (1, 256, 256, 3), dtype=np.uint8),
        "video.robot0_agentview_right": rng.integers(0, 256, (1, 256, 256, 3), dtype=np.uint8),
        "video.robot0_eye_in_hand": rng.integers(0, 256, (1, 256, 256, 3), dtype=np.uint8),
        "state.base_position": rng.uniform(-1.0, 1.0, (1, 3)).astype(np.float32),
        "state.base_rotation": rng.uniform(-1.0, 1.0, (1, 4)).astype(np.float32),
        "state.end_effector_position_relative": rng.uniform(-1.0, 1.0, (1, 3)).astype(np.float32),
        "state.end_effector_rotation_relative": rng.uniform(-1.0, 1.0, (1, 4)).astype(np.float32),
        "state.gripper_qpos": rng.uniform(0.0, 1.0, (1, 2)).astype(np.float32),
        "annotation.human.action.task_description": [ROBOCASA365_LANG],
    }


def make_gr1_obs(rng: np.random.Generator) -> Dict[str, Any]:
    """Random synthetic obs in GR1 (fourier_gr1_arms_waist) GR00T format.

    keys: video.ego_view (1 view) + 5 组 state + coarse_action 语言。
    state 维度: left/right arm 7, left/right hand 6, waist 3。
    """
    return {
        "video.ego_view": rng.integers(0, 256, (1, 256, 256, 3), dtype=np.uint8),
        "state.left_arm": rng.uniform(-0.5, 0.5, (1, 7)).astype(np.float32),
        "state.right_arm": rng.uniform(-0.5, 0.5, (1, 7)).astype(np.float32),
        "state.left_hand": rng.uniform(0.0, 1.0, (1, 6)).astype(np.float32),
        "state.right_hand": rng.uniform(0.0, 1.0, (1, 6)).astype(np.float32),
        "state.waist": rng.uniform(-0.3, 0.3, (1, 3)).astype(np.float32),
        "annotation.human.coarse_action": [GR1_LANG],
    }


def make_obs(rng: np.random.Generator, fmt: str = "libero") -> Dict[str, Any]:
    """Synthetic obs dispatcher: 'libero' (default), 'gr1' or 'robocasa365'."""
    if fmt == "gr1":
        return make_gr1_obs(rng)
    if fmt == "robocasa365":
        return make_robocasa365_obs(rng)
    return make_l1_obs(rng)


def stack_obs(obs_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Stack B single-obs dicts into one batched obs dict.

    Mirrors Gr00tPolicy.get_action's batched path: videos become (B,T,H,W,C),
    states (B,T,D), language becomes a np.array of B strings.
    """
    out: Dict[str, Any] = {}
    for key in obs_list[0].keys():
        vals = [o[key] for o in obs_list]
        if isinstance(vals[0], np.ndarray):
            out[key] = np.stack(vals, axis=0)
        else:
            out[key] = np.array([v[0] if isinstance(v, (list, tuple)) else v for v in vals])
    return out


def chunked(obs_list: List[Dict[str, Any]], noises: List[torch.Tensor], batch_size: int):
    """Yield (batched_obs, batched_noise) chunks."""
    for i in range(0, len(obs_list), batch_size):
        obs_chunk = obs_list[i : i + batch_size]
        noise_chunk = noises[i : i + batch_size]
        if not obs_chunk:
            continue
        batched_noise = torch.stack(noise_chunk, dim=0)  # (B, H, D)
        yield stack_obs(obs_chunk), batched_noise


# --------------------------------------------------------------------------- #
# Quant env handling (mirrors the v1 calibrator helpers)
# --------------------------------------------------------------------------- #
def strip_quant_env() -> Dict[str, str]:
    backup = {}
    for key in list(os.environ.keys()):
        if key.startswith(("GR00T_DUQUANT_", "GR00T_ATM_", "GR00T_OHB_")):
            backup[key] = os.environ.pop(key)
    return backup


def restore_quant_env(env_map: Dict[str, str]) -> None:
    for key, value in env_map.items():
        os.environ[key] = value


def set_quant_env(
    include: str,
    exclude: str,
    packdir: str,
    bits_default: int = 4,
    group: int = 64,
    ls: float = 0.15,
    act_pct: float = 99.9,
    calib_steps: int = 32,
    row_rot: str = "restore",
    act_dynamic: bool = False,
) -> None:
    os.environ["GR00T_DUQUANT_SCOPE"] = ""
    os.environ["GR00T_DUQUANT_INCLUDE"] = include
    os.environ["GR00T_DUQUANT_EXCLUDE"] = exclude
    os.environ["GR00T_DUQUANT_WBITS_DEFAULT"] = str(bits_default)
    os.environ["GR00T_DUQUANT_ABITS"] = "8"
    os.environ["GR00T_DUQUANT_BLOCK"] = str(group)
    os.environ["GR00T_DUQUANT_PERMUTE"] = "0"
    os.environ["GR00T_DUQUANT_ROW_ROT"] = row_rot
    os.environ["GR00T_DUQUANT_ACT_PCT"] = str(act_pct)
    os.environ["GR00T_DUQUANT_CALIB_STEPS"] = str(calib_steps)
    os.environ["GR00T_DUQUANT_LS"] = str(ls)
    os.environ["GR00T_DUQUANT_PACKDIR"] = packdir
    os.environ["GR00T_DUQUANT_ACT_DYNAMIC"] = "1" if act_dynamic else "0"
    os.environ["GR00T_DUQUANT_DEBUG"] = "0"
    # Base-mode consistency (v1.2): measurement passes must not inherit ambient
    # ATM/OHB/per-step scaling — those are deployment-time corrections whose
    # calibration must match the deployment act mode.
    for key in list(os.environ.keys()):
        if key.startswith(("GR00T_ATM_", "GR00T_OHB_")):
            os.environ.pop(key, None)


def ensure_flash_attn_rpath() -> None:
    """Same LD_LIBRARY_PATH helper as run_quantvla.sh (idempotent)."""
    if os.environ.get("LD_LIBRARY_PATH"):
        return
    import sys

    torch_lib = Path(sys.executable).parent / "lib" / "python3.10" / "site-packages" / "torch" / "lib"
    if torch_lib.exists():
        os.environ["LD_LIBRARY_PATH"] = str(torch_lib)


# --------------------------------------------------------------------------- #
# Policy loading
# --------------------------------------------------------------------------- #
def load_policy(
    model_path: str,
    data_config: str = "examples.Libero.custom_data_config:LiberoDataConfig",
    denoising_steps: int = 8,
    device: str = "cuda",
    embodiment_tag: str = "new_embodiment",
) -> Any:
    from gr00t.experiment.data_config import load_data_config
    from gr00t.model.policy import Gr00tPolicy

    cfg = load_data_config(data_config)
    policy = Gr00tPolicy(
        model_path=model_path,
        embodiment_tag=embodiment_tag,
        modality_config=cfg.modality_config(),
        modality_transform=cfg.transform(),
        denoising_steps=denoising_steps,
        device=device,
    )
    return policy


def action_shape(model: Any) -> tuple:
    """(action_horizon, action_dim) for paired-noise generation."""
    horizon = int(model.action_head.config.action_horizon)
    action_dim = int(model.action_head.config.action_dim)
    return horizon, action_dim


def make_noises(model: Any, n: int, seed: int = 0) -> List[torch.Tensor]:
    horizon, action_dim = action_shape(model)
    gen = torch.Generator().manual_seed(seed)
    # 每个 obs 一个 2D 噪声 (H, D)；chunked 打包时叠加 batch 维 → (B, H, D)
    return [torch.randn(horizon, action_dim, generator=gen) for _ in range(n)]


# --------------------------------------------------------------------------- #
# Suite -> data-config mapping (review round 2, item 6)
# --------------------------------------------------------------------------- #
# The LIBERO service (run_quantvla.sh) serves the GOAL suite with
# LiberoDataConfigMeanStd; every measurement tool must use the SAME transforms
# or probe/TopK/baseline scores live in a different coordinate system than the
# deployed policy.
SUITE_DATA_CONFIG = {
    "spatial": "examples.Libero.custom_data_config:LiberoDataConfig",
    "goal": "examples.Libero.custom_data_config:LiberoDataConfigMeanStd",
    "object": "examples.Libero.custom_data_config:LiberoDataConfig",
    "10": "examples.Libero.custom_data_config:LiberoDataConfig",
    "90": "examples.Libero.custom_data_config:LiberoDataConfig",
    "robocasa365_atomic": "examples.RoboCasa365.custom_data_config:RoboCasa365DataConfig",
}


def resolve_data_config(suite: str, arg: str | None) -> str:
    if arg:
        return arg
    if suite not in SUITE_DATA_CONFIG:
        raise SystemExit(f"unknown suite {suite!r} for data-config resolution")
    return SUITE_DATA_CONFIG[suite]


# --------------------------------------------------------------------------- #
# A8 calibration closure (review round 2, items 1/3/4/5)
# --------------------------------------------------------------------------- #
def _hash_obs_dict(obs: Dict[str, Any], h: Any) -> None:
    for key in sorted(obs):
        v = obs[key]
        h.update(key.encode())
        if isinstance(v, np.ndarray):
            h.update(v.tobytes())
        elif isinstance(v, (list, tuple)):
            h.update(repr(v).encode())
        else:
            h.update(str(v).encode())


def fixed_calibration_buffer(
    seed: int,
    n_obs: int,
    horizon: int,
    action_dim: int,
    fmt: str = "libero",
) -> tuple:
    """Deterministic, SELF-CONTAINED synthetic calibration buffer + sha256.

    Review round 3, item 2: the seed fully determines the buffer — no caller
    RNG state, no process-global torch RNG (a local torch.Generator is used),
    so the TOPK scorer, the baselines, the calibrator and the inference server
    all reproduce the IDENTICAL obs+noise data from the same seed. The
    fingerprint covers the observations (images/state/instruction), the noises
    and the shape/format metadata — not just the noises.
    """
    import hashlib

    np_rng = np.random.default_rng(seed)
    torch_gen = torch.Generator(device="cpu").manual_seed(seed)
    obs_list = [make_obs(np_rng, fmt) for _ in range(n_obs)]
    noises = [torch.randn(horizon, action_dim, generator=torch_gen) for _ in range(n_obs)]
    h = hashlib.sha256()
    h.update(f"{fmt}|{n_obs}|{horizon}|{action_dim}".encode())
    for ob in obs_list:
        _hash_obs_dict(ob, h)
    for nz in noises:
        h.update(nz.numpy().tobytes())
    return obs_list, noises, h.hexdigest()


def warmup_forward(policy: Any, obs_list: list, noises: list, batch_size: int) -> None:
    """Forward-only pass with the given paired noises (autocast-aware)."""
    from gr00t.model.policy import COMPUTE_DTYPE

    use_autocast = str(policy.device).startswith("cuda")
    for batched_obs, batched_noise in chunked(obs_list, noises, batch_size):
        norm = policy.apply_transforms(batched_obs)
        with torch.inference_mode():
            if use_autocast:
                with torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
                    policy.model.get_action(norm, action_noise=batched_noise)
            else:
                policy.model.get_action(norm, action_noise=batched_noise)


def count_wrapped_layers(model: Any) -> int:
    from gr00t.quantization.duquant_layers import DuQuantLinear

    return sum(1 for m in model.modules() if isinstance(m, DuQuantLinear))


def ensure_a8_calibrated(
    policy: Any,
    warm_obs: list,
    warm_noises: list,
    batch_size: int,
    act_dynamic: bool = False,
    expected_wrapped: int | None = None,
    act_scale_path: str | None = None,
    act_scale_meta: Dict[str, Any] | None = None,
) -> None:
    """Close the static-A8 calibration loop before any measurement/serving.

    Review round 2 (items 1/3/4):
      - dynamic-act mode has no static calibrators (0/0 must NOT abort): only
        the wrapped-layer count is verified;
      - static mode runs the FIXED buffer until all_calibrated() and raises if
        the calibration does not complete;
      - optional persistence: load frozen scales from act_scale_path when they
        exist, save them after the first successful warmup.
    """
    from gr00t.quantization.duquant_layers import (
        all_calibrated,
        load_act_scales,
        save_act_scales,
        static_calibrators_required,
        static_scales_ready,
    )

    model = policy.model
    if expected_wrapped is not None:
        n = count_wrapped_layers(model)
        if n != expected_wrapped:
            raise SystemExit(
                f"[a8-calib] wrap mismatch: {n} wrapped layers, expected {expected_wrapped}"
            )
    if act_dynamic or not static_calibrators_required(model):
        return  # dynamic act (or no static-A8 layers): nothing to freeze

    if act_scale_path and Path(act_scale_path).exists():
        # review round 3, item 1: judge the LOADED state by static_scales_ready
        # (frozen scales installed), not by the calibrator counter.
        load_act_scales(model, act_scale_path, require=act_scale_meta)
        if static_scales_ready(model) and all_calibrated(model):
            print(f"[a8-calib] loaded frozen A8 scales from {act_scale_path}", flush=True)
            return
        raise SystemExit(f"[a8-calib] {act_scale_path} does not cover all static layers")

    if static_scales_ready(model):
        return
    warmup_forward(policy, warm_obs, warm_noises, batch_size)
    if not all_calibrated(model):
        from gr00t.quantization.duquant_layers import calibration_progress

        full, total = calibration_progress(model)
        raise SystemExit(
            f"[a8-calib] static A8 calibration incomplete: {full}/{total} after "
            f"{len(warm_obs)} obs / {len(warm_obs) // max(batch_size, 1)} batches"
        )
    if act_scale_path:
        save_act_scales(model, act_scale_path, meta=act_scale_meta)
        print(f"[a8-calib] frozen A8 scales saved -> {act_scale_path}", flush=True)
    print(f"[a8-calib] static A8 calibration complete ({count_wrapped_layers(model)} wrapped layers)", flush=True)
