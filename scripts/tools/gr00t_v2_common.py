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
    """Synthetic obs dispatcher: 'libero' (default) or 'gr1'."""
    if fmt == "gr1":
        return make_gr1_obs(rng)
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
