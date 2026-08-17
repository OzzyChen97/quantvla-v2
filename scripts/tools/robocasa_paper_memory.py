#!/usr/bin/env python3
"""QuantVLA paper-style LLM+DiT component-memory accounting.

This deliberately does *not* report live CUDA memory.  It mirrors the scope
used by Tables 1/2 of the QuantVLA paper:

* all Linear weights/biases in the GR00T language model and DiT blocks form
  the FP16 reference scope (180 Linear layers for GR00T N1.5);
* a plan-selected weight is stored at its declared bit width with ideal tight
  packing;
* FP16 per-output-channel weight scales and FP32 DuQuant block rotations are
  included;
* unselected weights and all biases remain FP16;
* activations, CUDA workspaces, vision modules, fake-quant caches, and the
  simulator are excluded;
* ATM/OHB are treated as folded into existing scales, as stated in the paper.

The result is a deployment-storage estimate, not a claim about the current
fake-quant runtime's allocation or latency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tempfile
from pathlib import Path
from typing import Any


LLM_LINEAR = re.compile(
    r"backbone\.eagle_model\.language_model\..*\."
    r"(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.weight"
)
DIT_LINEAR = re.compile(
    r"action_head\.model\.transformer_blocks\.\d+\."
    r"(?:attn1\.(?:to_q|to_k|to_v|to_out\.0)|ff\.net\.(?:0\.proj|2))\.weight"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reference_path(value: Any) -> Path | None:
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("path")
    if not value:
        return None
    return Path(str(value)).expanduser().resolve()


def _checkpoint_dir(value: Any) -> Path:
    path = _reference_path(value)
    if path is None:
        raise ValueError("manifest is missing checkpoint.path")
    if path.is_file():
        path = path.parent
    if not path.is_dir():
        raise ValueError(f"checkpoint directory does not exist: {path}")
    return path


def read_tensor_shapes(checkpoint: Path) -> dict[str, tuple[int, ...]]:
    """Read tensor shapes from a single- or multi-file safetensors checkpoint."""
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - environment error is explicit
        raise RuntimeError("safetensors is required for paper-style memory accounting") from exc

    if checkpoint.is_file():
        files = [checkpoint]
    else:
        single = checkpoint / "model.safetensors"
        files = [single] if single.is_file() else sorted(checkpoint.glob("model-*.safetensors"))
    if not files:
        raise ValueError(f"no model safetensors found under {checkpoint}")

    shapes: dict[str, tuple[int, ...]] = {}
    for path in files:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in shapes:
                    raise ValueError(f"duplicate checkpoint tensor {key} in {path}")
                shapes[key] = tuple(int(value) for value in handle.get_slice(key).get_shape())
    return shapes


def paper_scope(shapes: dict[str, tuple[int, ...]]) -> list[str]:
    layers = sorted(
        name for name, shape in shapes.items()
        if len(shape) == 2 and (LLM_LINEAR.fullmatch(name) or DIT_LINEAR.fullmatch(name))
    )
    if not layers:
        raise ValueError("checkpoint contains no QuantVLA paper-scope LLM+DiT Linear layers")
    return layers


def _numel(shape: tuple[int, ...]) -> int:
    return math.prod(shape)


def _base_bytes(
    shapes: dict[str, tuple[int, ...]], scope_layers: list[str]
) -> tuple[int, int, int]:
    weight_bytes = sum(_numel(shapes[name]) * 2 for name in scope_layers)
    bias_bytes = 0
    for weight_name in scope_layers:
        bias_name = weight_name[:-7] + ".bias"
        if bias_name in shapes:
            bias_bytes += _numel(shapes[bias_name]) * 2
    return weight_bytes + bias_bytes, weight_bytes, bias_bytes


def _plan_memory(
    shapes: dict[str, tuple[int, ...]],
    scope_layers: list[str],
    baseline_bytes: int,
    plan_path: Path | None,
) -> dict[str, Any]:
    if plan_path is None:
        return {
            "plan_sha256": None,
            "quantized_layers": 0,
            "fp16_layers": len(scope_layers),
            "planned_fp16_skips": 0,
            "quantized_parameters": 0,
            "component_bytes": baseline_bytes,
            "component_mib": baseline_bytes / 2**20,
            "component_gib": baseline_bytes / 2**30,
            "relative_savings": 0.0,
            "compression_ratio": 1.0,
            "breakdown_mib": {
                "packed_weights": 0.0,
                "weight_scales": 0.0,
                "rotations": 0.0,
                "permutation_indices": 0.0,
            },
        }
    if not plan_path.is_file():
        raise ValueError(f"quantization plan does not exist: {plan_path}")
    plan = json.loads(plan_path.read_text())
    entries = plan.get("layers")
    if not isinstance(entries, dict) or not entries:
        raise ValueError(f"plan has no layer map: {plan_path}")

    scope = set(scope_layers)
    total = float(baseline_bytes)
    packed_weights = weight_scales = rotations = permutation_indices = 0.0
    quantized_parameters = quantized_layers = planned_skips = 0
    for layer_name, config in entries.items():
        weight_name = f"{layer_name}.weight"
        if weight_name not in scope:
            raise ValueError(f"plan layer is outside/missing from paper scope: {weight_name}")
        if bool(config.get("skip", False)):
            planned_skips += 1
            continue
        bits = int(config.get("bits", 4))
        group = int(config.get("group", 64))
        if bits <= 0 or group <= 0:
            raise ValueError(f"invalid bits/group for {layer_name}: {bits}/{group}")
        out_features, in_features = shapes[weight_name]
        parameters = out_features * in_features
        fp16_weight = parameters * 2.0
        packed = parameters * bits / 8.0
        scales = out_features * 2.0
        rotation = (
            math.ceil(in_features / group) + math.ceil(out_features / group)
        ) * group * group * 4.0
        # Current official artifacts use enable_permute=false.  Honor an
        # explicit future plan setting without charging absent permutations.
        permutation = in_features * 4.0 if bool(config.get("permute", False)) else 0.0
        total += packed + scales + rotation + permutation - fp16_weight
        packed_weights += packed
        weight_scales += scales
        rotations += rotation
        permutation_indices += permutation
        quantized_parameters += parameters
        quantized_layers += 1

    component_bytes = int(round(total))
    return {
        "plan_sha256": _sha256(plan_path),
        "quantized_layers": quantized_layers,
        "fp16_layers": len(scope_layers) - quantized_layers,
        "planned_fp16_skips": planned_skips,
        "quantized_parameters": quantized_parameters,
        "component_bytes": component_bytes,
        "component_mib": component_bytes / 2**20,
        "component_gib": component_bytes / 2**30,
        "relative_savings": 1.0 - component_bytes / baseline_bytes,
        "compression_ratio": baseline_bytes / component_bytes,
        "breakdown_mib": {
            "packed_weights": packed_weights / 2**20,
            "weight_scales": weight_scales / 2**20,
            "rotations": rotations / 2**20,
            "permutation_indices": permutation_indices / 2**20,
        },
    }


def calculate_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    checkpoint = _checkpoint_dir(manifest.get("checkpoint"))
    shapes = read_tensor_shapes(checkpoint)
    scope_layers = paper_scope(shapes)
    baseline, weight_bytes, bias_bytes = _base_bytes(shapes, scope_layers)
    configs = {}
    for config in manifest.get("configs", []):
        config_id = str(config["id"])
        configs[config_id] = _plan_memory(
            shapes, scope_layers, baseline, _reference_path(config.get("plan"))
        )
    if not configs:
        raise ValueError(f"manifest contains no configs: {manifest_path}")
    return {
        "scope": "LLM+DiT Linear weights/biases (QuantVLA Tables 1/2)",
        "unit_note": "GiB (2^30 bytes), displayed as GB in the paper",
        "estimate_kind": "theoretical tightly-packed deployment component storage",
        "excluded": [
            "vision modules",
            "activation peaks",
            "CUDA/runtime workspaces",
            "fake-quant caches",
            "simulator",
        ],
        "atm_ohb": "folded into existing scales; zero additional operators/buffers",
        "checkpoint": str(checkpoint),
        "scope_linear_layers": len(scope_layers),
        "fp16_weight_bytes": weight_bytes,
        "fp16_bias_bytes": bias_bytes,
        "fp16_component_bytes": baseline,
        "fp16_component_gib": baseline / 2**30,
        "configs": configs,
    }


def selftest() -> None:
    shapes = {
        "backbone.eagle_model.language_model.model.layers.0.self_attn.q_proj.weight": (8, 8),
        "backbone.eagle_model.language_model.model.layers.0.self_attn.q_proj.bias": (8,),
        "action_head.model.transformer_blocks.0.attn1.to_q.weight": (8, 8),
        "vision.unrelated.weight": (100, 100),
    }
    scope = paper_scope(shapes)
    assert len(scope) == 2
    baseline, _, _ = _base_bytes(shapes, scope)
    assert baseline == 2 * 8 * 8 * 2 + 8 * 2
    with tempfile.TemporaryDirectory() as tmp:
        plan = Path(tmp) / "plan.json"
        plan.write_text(json.dumps({
            "layers": {
                "backbone.eagle_model.language_model.model.layers.0.self_attn.q_proj": {
                    "bits": 4, "group": 8, "skip": False,
                }
            }
        }))
        result = _plan_memory(shapes, scope, baseline, plan)
        assert result["quantized_layers"] == 1
        assert result["fp16_layers"] == 1
        assert result["component_bytes"] < baseline + 2 * 8 * 8 * 4
    print("[paper-memory] selftest OK (scope + packed weights + metadata)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest")
    parser.add_argument("--selftest", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.selftest:
        selftest()
        return
    if not args.manifest:
        raise SystemExit("--manifest is required unless --selftest is used")
    print(json.dumps(calculate_manifest(Path(args.manifest)), indent=2))


if __name__ == "__main__":
    main()
