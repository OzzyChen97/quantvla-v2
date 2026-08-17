# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
GR00T Inference Service

This script provides both ZMQ and HTTP server/client implementations for deploying GR00T models.
The HTTP server exposes a REST API for easy integration with web applications and other services.

1. Default is zmq server.

Run server: python scripts/inference_service.py --server
Run client: python scripts/inference_service.py --client

2. Run as Http Server:

Dependencies for `http_server` mode:
    => Server (runs GR00T model on GPU): `pip install uvicorn fastapi json-numpy`
    => Client: `pip install requests json-numpy`

HTTP Server Usage:
    python scripts/inference_service.py --server --http-server --port 8000

HTTP Client Usage (assuming a server running on 0.0.0.0:8000):
    python scripts/inference_service.py --client --http-server --host 0.0.0.0 --port 8000

You can use bore to forward the port to your client: `159.223.171.199` is bore.pub.
    bore local 8000 --to 159.223.171.199
"""

import os
import sys
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import tyro

from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING
from gr00t.eval.robot import RobotInferenceClient, RobotInferenceServer
from gr00t.experiment.data_config import load_data_config
from gr00t.model.policy import Gr00tPolicy


@dataclass
class ArgsConfig:
    """Command line arguments for the inference service."""

    model_path: str = "nvidia/GR00T-N1.5-3B"
    """Path to the model checkpoint directory."""

    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "gr1"
    """The embodiment tag for the model."""

    data_config: str = "fourier_gr1_arms_waist"
    """
    The name of the data config to use, e.g. so100, fourier_gr1_arms_only, unitree_g1, etc.

    Or a path to a custom data config file. e.g. "module:ClassName" format.
    See gr00t/experiment/data_config.py for more details.
    """

    port: int = 5555
    """The port number for the server."""

    host: str = "localhost"
    """The host address for the server."""

    server: bool = False
    """Whether to run the server."""

    client: bool = False
    """Whether to run the client."""

    denoising_steps: int = 4
    """The number of denoising steps to use."""

    api_token: str = None
    """API token for authentication. If not provided, authentication is disabled."""

    http_server: bool = False
    """Whether to run it as HTTP server. Default is ZMQ server."""


#####################################################################################


def _example_zmq_client_call(obs: dict, host: str, port: int, api_token: str):
    """
    Example ZMQ client call to the server.
    """
    # Original ZMQ client mode
    # Create a policy wrapper
    policy_client = RobotInferenceClient(host=host, port=port, api_token=api_token)

    print("Available modality config available:")
    modality_configs = policy_client.get_modality_config()
    print(modality_configs.keys())

    time_start = time.time()
    action = policy_client.get_action(obs)
    print(f"Total time taken to get action from server: {time.time() - time_start} seconds")
    return action


def _example_http_client_call(obs: dict, host: str, port: int, api_token: str):
    """
    Example HTTP client call to the server.
    """
    import json_numpy

    json_numpy.patch()
    import requests

    # Send request to HTTP server
    print("Testing HTTP server...")

    time_start = time.time()
    response = requests.post(f"http://{host}:{port}/act", json={"observation": obs})
    print(f"Total time taken to get action from HTTP server: {time.time() - time_start} seconds")

    if response.status_code == 200:
        action = response.json()
        return action
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return {}


def _maybe_close_a8_calibration(
    policy, data_config: str = "", model_path: str = ""
) -> None:
    """Static-A8 calibration closure before serving (review round 2, item 3).

    No-op for FP16 models (no GR00T_DUQUANT_* env) and for dynamic-act models
    (no static calibrators). For static-A8 quantized models: load persisted
    frozen scales when GR00T_DUQUANT_ACT_SCALE_PATH exists, otherwise run the
    fixed synthetic buffer (calib_steps × batch_size obs, seed 0) and save.
    """
    if not any(
        k.startswith("GR00T_DUQUANT_") and k != "GR00T_DUQUANT_PACKDIR"
        for k in os.environ
    ):
        return
    from pathlib import Path as _Path

    _here = _Path(__file__).resolve().parents[1]
    if str(_here / "scripts" / "tools") not in sys.path:
        sys.path.insert(0, str(_here / "scripts" / "tools"))

    from gr00t.quantization.duquant_layers import static_calibrators_required
    from gr00t_v2_common import (
        count_wrapped_layers,
        ensure_a8_calibrated,
        fixed_calibration_buffer,
    )

    act_dynamic = os.environ.get("GR00T_DUQUANT_ACT_DYNAMIC", "0") not in ("0", "false", "False")
    if act_dynamic or not static_calibrators_required(policy.model):
        return
    scale_path = os.environ.get("GR00T_DUQUANT_ACT_SCALE_PATH") or None
    calib_steps = int(os.environ.get("GR00T_DUQUANT_CALIB_STEPS", "32"))
    batch_size = 8
    horizon = int(policy.model.action_head.config.action_horizon)
    action_dim = int(policy.model.action_head.config.action_dim)
    # review round 3: self-contained seed-based buffer — identical data every
    # start, so the sha256 + sidecar prove the scales match the experiment.
    # v1.4 Stage D: GR00T_OBS_FORMAT=robocasa365 for the RoboCasa365 checkpoints
    fmt = os.environ.get("GR00T_OBS_FORMAT", "libero")
    warm_obs, warm_noises, sha = fixed_calibration_buffer(
        0, calib_steps * batch_size, horizon, action_dim, fmt=fmt
    )
    import hashlib as _hl

    plan_path = os.environ.get("GR00T_DUQUANT_PLAN")
    plan_sha = None
    if plan_path and os.path.exists(plan_path):
        with open(plan_path, "rb") as f:
            plan_sha = _hl.sha256(f.read()).hexdigest()
    checkpoint_path = str(_Path(model_path).resolve()) if model_path else None
    act_meta = {
        "buffer_sha256": sha,
        "calibration_seed": 0,
        "data_config": data_config,
        "obs_format": fmt,
        "act_percentile": float(os.environ.get("GR00T_DUQUANT_ACT_PCT", "99.9")),
        "calib_batches": calib_steps,
        "denoising_steps": int(
            os.environ.get("GR00T_DENOISING_STEPS", str(policy.denoising_steps))
        ),
        "plan_sha256": plan_sha,
        "checkpoint_path": checkpoint_path,
        "wrapped_layers": count_wrapped_layers(policy.model),
    }
    print(f"[inference] static A8 calibration warmup: {calib_steps * batch_size} "
          f"synthetic obs (buffer sha256 {sha[:16]}...; plan {plan_sha})", flush=True)
    t0 = time.time()
    ensure_a8_calibrated(
        policy, warm_obs, warm_noises, batch_size,
        act_dynamic=False, act_scale_path=scale_path, act_scale_meta=act_meta,
    )
    print(f"[inference] static A8 calibration complete in {time.time() - t0:.1f}s", flush=True)


def _seeded_action_handler(policy, payload: dict) -> dict:
    """Evaluate one observation with deterministic, request-local FM noise.

    The seed is derived by the RoboCasa client from
    (task, environment seed, replan index).  Creating the noise with a local
    CPU generator makes requests independent of server RNG state and client
    interleaving while preserving the standard-normal distribution.
    """
    if not isinstance(payload, dict) or "observations" not in payload:
        raise ValueError("get_action_seeded requires {observations, action_seed}")
    if "action_seed" not in payload:
        raise ValueError("get_action_seeded payload is missing action_seed")

    import torch

    seed = int(payload["action_seed"])
    if seed < 0 or seed >= 2**63:
        raise ValueError(f"action_seed must be in [0, 2**63), got {seed}")
    horizon = int(policy.model.action_head.config.action_horizon)
    action_dim = int(policy.model.action_head.config.action_dim)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    action_noise = torch.randn(
        (horizon, action_dim), generator=generator, dtype=torch.float32
    )
    return policy.get_action(payload["observations"], action_noise=action_noise)


def _runtime_info(policy) -> dict:
    from gr00t_v2_common import count_wrapped_layers

    modules = list(policy.model.modules())
    return {
        "wrapped_layers": count_wrapped_layers(policy.model),
        "plan": os.environ.get("GR00T_DUQUANT_PLAN"),
        "act_scale_path": os.environ.get("GR00T_DUQUANT_ACT_SCALE_PATH"),
        "atm_enabled": os.environ.get("GR00T_ATM_ENABLE", "0") == "1",
        "ohb_enabled": os.environ.get("GR00T_OHB_ENABLE", "0") == "1",
        "atm_path": os.environ.get("GR00T_ATM_ALPHA_PATH"),
        "atm_layers": sum(hasattr(m, "_atm_alpha_all") for m in modules),
        "ohb_layers": sum(
            hasattr(m, "_ohb_beta_perhead") or hasattr(m, "_ohb_beta_scalar")
            for m in modules
        ),
        "denoising_steps": int(policy.denoising_steps),
    }


def main(args: ArgsConfig):
    if args.server:
        # Create a policy
        # The `Gr00tPolicy` class is being used to create a policy object that encapsulates
        # the model path, transform name, embodiment tag, and denoising steps for the robot
        # inference system. This policy object is then utilized in the server mode to start
        # the Robot Inference Server for making predictions based on the specified model and
        # configuration.

        # we will use an existing data config to create the modality config and transform
        # if a new data config is specified, this expect user to
        # construct your own modality config and transform
        # see gr00t/utils/data.py for more details
        data_config = load_data_config(args.data_config)
        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()

        policy = Gr00tPolicy(
            model_path=args.model_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=args.embodiment_tag,
            denoising_steps=args.denoising_steps,
        )

        # Review round 2, item 3: close the static-A8 calibration loop BEFORE
        # the server accepts any request. Without this, the first LIBERO
        # get_action calls would online-calibrate the frozen scales on TEST
        # observations (model state changing during evaluation). A fixed
        # synthetic buffer (same sha256 across starts) completes the
        # calibration; GR00T_DUQUANT_ACT_SCALE_PATH persists it across runs.
        _maybe_close_a8_calibration(
            policy, data_config=args.data_config, model_path=args.model_path
        )

        # Start the server
        if args.http_server:
            from gr00t.eval.http_server import HTTPInferenceServer  # noqa: F401

            server = HTTPInferenceServer(
                policy, port=args.port, host=args.host, api_token=args.api_token
            )
            server.run()
        else:
            server = RobotInferenceServer(policy, port=args.port, api_token=args.api_token)
            server.register_endpoint(
                "get_action_seeded",
                lambda payload: _seeded_action_handler(policy, payload),
            )
            server.register_endpoint(
                "get_runtime_info", lambda: _runtime_info(policy), requires_input=False
            )
            server.run()

    # Here is mainly a testing code
    elif args.client:
        # In this mode, we will send a random observation to the server and get an action back
        # This is useful for testing the server and client connection

        # Making prediction...
        # - obs: video.ego_view: (1, 256, 256, 3)
        # - obs: state.left_arm: (1, 7)
        # - obs: state.right_arm: (1, 7)
        # - obs: state.left_hand: (1, 6)
        # - obs: state.right_hand: (1, 6)
        # - obs: state.waist: (1, 3)

        # - action: action.left_arm: (16, 7)
        # - action: action.right_arm: (16, 7)
        # - action: action.left_hand: (16, 6)
        # - action: action.right_hand: (16, 6)
        # - action: action.waist: (16, 3)
        obs = {
            "video.ego_view": np.random.randint(0, 256, (1, 256, 256, 3), dtype=np.uint8),
            "state.left_arm": np.random.rand(1, 7),
            "state.right_arm": np.random.rand(1, 7),
            "state.left_hand": np.random.rand(1, 6),
            "state.right_hand": np.random.rand(1, 6),
            "state.waist": np.random.rand(1, 3),
            "annotation.human.action.task_description": ["do your thing!"],
        }

        if args.http_server:
            action = _example_http_client_call(obs, args.host, args.port, args.api_token)
        else:
            action = _example_zmq_client_call(obs, args.host, args.port, args.api_token)

        for key, value in action.items():
            print(f"Action: {key}: {value.shape}")
    else:
        raise ValueError("Please specify either --server or --client")


if __name__ == "__main__":
    # SIGUSR1 dumps all python thread stacks to the log — the runtime watchdog
    # uses it to diagnose hung servers (review round 5, item 3)
    import faulthandler
    import signal as _signal

    faulthandler.register(_signal.SIGUSR1)

    config = tyro.cli(ArgsConfig)
    main(config)
