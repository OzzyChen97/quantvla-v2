import json
import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import torch

try:
    from diffusers.models.attention import Attention
    from diffusers.models.attention_processor import AttnProcessor2_0
except ModuleNotFoundError as exc:  # pragma: no cover - handled at runtime
    raise RuntimeError(
        "diffusers is required for GR00T ATM support. "
        "Please ensure the gr00t environment is activated."
    ) from exc

ATM_ENABLE_ENV = "GR00T_ATM_ENABLE"
ATM_ALPHA_ENV = "GR00T_ATM_ALPHA_PATH"
ATM_SCOPE_ENV = "GR00T_ATM_SCOPE"
ATM_PER_STEP_ENV = "GR00T_ATM_PER_STEP"

OHB_ENABLE_ENV = "GR00T_OHB_ENABLE"
OHB_SCOPE_ENV = "GR00T_OHB_SCOPE"
OHB_ONLY_DIT_ENV = "GR00T_OHB_ONLY_DIT"
OHB_FALLBACK_ENV = "GR00T_OHB_FALLBACK"

_ATM_PATCH_FLAG = "_gr00t_atm_processor_patched"


def _is_dit_attention(name: str, module: Attention, scope: str = "dit") -> bool:
    if not isinstance(module, Attention):
        return False
    if scope == "dit":
        return "action_head.model.transformer_blocks" in name
    if scope == "all":
        return True
    return scope in name


class _ATMProcessor(AttnProcessor2_0):
    """Attention processor with per-head statistics capture and optional scaling."""

    def __init__(self):
        super().__init__()

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Capture logits statistics before ATM scaling
        capture_cb = getattr(attn, "_atm_capture_callback", None)
        capture_step_cb = getattr(attn, "_atm_capture_step_callback", None)
        logits_capture_cb = getattr(attn, "_atm_logits_capture_callback", None)
        if capture_cb is not None or capture_step_cb is not None or logits_capture_cb is not None:
            std, logits_tensor = _compute_logits_std(
                query, key, attention_mask, head_dim, return_logits=True
            )
            if capture_cb is not None:
                capture_cb(attn, std)
            if capture_step_cb is not None:
                capture_step_cb(attn, std, _get_step(attn))
            if logits_capture_cb is not None:
                logits_capture_cb(attn, logits_tensor)

        # Apply ATM scaling if provided (QuantVLA v2: per-step table takes
        # precedence, missing steps fall back to the v1 static "all" alpha)
        alpha = getattr(attn, "_atm_alpha_all", None)
        alpha_by_step = getattr(attn, "_atm_alpha_by_step", None)
        if alpha_by_step is not None:
            alpha = _resolve_step_value(alpha_by_step, alpha, _get_step(attn))
        if alpha is not None:
            alpha = alpha.to(dtype=query.dtype, device=query.device).view(1, -1, 1, 1)
            query = query * alpha

        hidden_states = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        # Capture per-head RMS for per-head OHB calibration (BEFORE reshape)
        # hidden_states shape here: (batch, heads, seq, head_dim)
        ohb_perhead_capture_cb = getattr(attn, "_atm_ohb_perhead_capture_callback", None)
        ohb_perhead_capture_step_cb = getattr(attn, "_atm_ohb_perhead_capture_step_callback", None)
        if ohb_perhead_capture_cb is not None or ohb_perhead_capture_step_cb is not None:
            rms_per_head = _compute_rms_per_head(hidden_states)
            if ohb_perhead_capture_cb is not None:
                ohb_perhead_capture_cb(attn, rms_per_head)
            if ohb_perhead_capture_step_cb is not None:
                ohb_perhead_capture_step_cb(attn, rms_per_head, _get_step(attn))

        # Apply per-head OHB beta scaling (BEFORE reshape)
        beta_perhead = getattr(attn, "_ohb_beta_perhead", None)
        beta_perhead_by_step = getattr(attn, "_ohb_beta_perhead_by_step", None)
        if beta_perhead_by_step is not None:
            beta_perhead = _resolve_step_value(beta_perhead_by_step, beta_perhead, _get_step(attn))
        if beta_perhead is not None:
            # beta_perhead shape: (heads,), expand to (1, heads, 1, 1)
            beta_perhead = beta_perhead.to(dtype=hidden_states.dtype, device=hidden_states.device)
            beta_perhead = beta_perhead.view(1, -1, 1, 1)
            hidden_states = hidden_states * beta_perhead

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        # Capture/output scale for OHB calibration
        ohb_capture_cb = getattr(attn, "_atm_ohb_capture_callback", None)
        if ohb_capture_cb is not None:
            ohb_capture_cb(attn, _compute_rms(hidden_states))

        beta = getattr(attn, "_ohb_beta_scalar", None)
        beta_by_step = getattr(attn, "_ohb_beta_by_step", None)
        if beta_by_step is not None:
            beta = _resolve_step_value(beta_by_step, beta, _get_step(attn))
        if beta is not None and beta != 1.0:
            hidden_states = hidden_states * beta

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        # QuantVLA v2: generic output-tensor capture for CKA/CS sensitivity
        # probing (scripts/tools/gr00t_sensitivity_probe.py). Zero overhead
        # when no callback is registered.
        out_cb = getattr(attn, "_atm_output_capture_callback", None)
        if out_cb is not None:
            step_getter = getattr(attn, "_atm_step_getter", None)
            step = step_getter() if step_getter is not None else None
            out_cb(attn, hidden_states.detach(), step)

        return hidden_states


def _compute_logits_std(
    query: torch.Tensor,
    key: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    head_dim: int,
    *,
    return_logits: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    dtype = torch.float32
    scale = 1.0 / math.sqrt(max(float(head_dim), 1.0))
    logits = torch.matmul(query.to(dtype), key.to(dtype).transpose(-1, -2)) * scale

    if attention_mask is not None:
        # attention mask is additive, with large negative entries for masked positions
        valid = attention_mask >= -1e4
    else:
        valid = torch.ones_like(logits, dtype=torch.bool)

    valid = valid.to(dtype)
    count = valid.sum(dim=(-1, -2)).clamp_min(1.0)
    mean = (logits * valid).sum(dim=(-1, -2)) / count
    mean = mean.unsqueeze(-1).unsqueeze(-1)
    var = ((logits - mean) ** 2 * valid).sum(dim=(-1, -2)) / count
    std = torch.sqrt(var.clamp_min(1e-12))
    std = std.detach()
    if return_logits:
        return std, logits.detach()
    return std


def _compute_rms(tensor: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(tensor.detach().to(torch.float32) ** 2) + 1e-12)


def _compute_rms_per_head(tensor: torch.Tensor) -> torch.Tensor:
    """Compute RMS per head for per-head OHB.

    Args:
        tensor: Shape (batch, heads, seq, head_dim) - attention output BEFORE reshape.

    Returns:
        Shape (heads,) - RMS value per head averaged over batch, seq, head_dim.
    """
    # tensor shape: (batch, heads, seq, head_dim)
    t = tensor.detach().to(torch.float32)
    # Compute RMS per head: sqrt(mean(x^2)) over (batch, seq, head_dim)
    rms_per_head = torch.sqrt(torch.mean(t ** 2, dim=(0, 2, 3)) + 1e-12)  # (heads,)
    return rms_per_head


def _get_step(attn: Attention) -> Optional[int]:
    """Current denoising step (t_discretized) or None (QuantVLA v2)."""
    getter = getattr(attn, "_atm_step_getter", None)
    return getter() if getter is not None else None


def _resolve_step_value(by_step: Dict[int, Any], fallback: Any, step: Optional[int]) -> Any:
    """Per-step table lookup with fallback (missing steps use the static value)."""
    if step is None:
        return fallback
    return by_step.get(int(step), fallback)


def compute_per_step_alpha(
    teacher_std_by_step: Dict[int, torch.Tensor],
    quant_std_by_step: Dict[int, torch.Tensor],
    min_alpha: float = 0.7,
    max_alpha: float = 1.4,
    neutral_threshold: float = 0.02,
) -> Dict[str, Any]:
    """Build the v2 per-step ATM alpha entry (QuantVLA v2 P2-G).

    Args:
        teacher_std_by_step: {t_discretized: (heads,) tensor} FP16 reference.
        quant_std_by_step: {t_discretized: (heads,) tensor} quantized model.

    Returns:
        {"all": [...], "steps": {"<t>": {"all": [...]}, ...}} where "all" is the
        step-pooled v1-style static alpha (mean of per-step values).
    """
    steps = sorted(set(teacher_std_by_step) & set(quant_std_by_step))

    def alpha_of(t_std: torch.Tensor, q_std: torch.Tensor) -> List[float]:
        a = torch.where(
            q_std > 0,
            t_std.to(torch.float32) / (q_std.to(torch.float32) + 1e-6),
            torch.ones_like(t_std),
        )
        a = a.clamp(min_alpha, max_alpha)
        a = torch.where((a - 1.0).abs() < neutral_threshold, torch.ones_like(a), a)
        return a.tolist()

    per_step: Dict[str, Dict[str, List[float]]] = {}
    pooled: List[torch.Tensor] = []
    for t in steps:
        entry = {"all": alpha_of(teacher_std_by_step[t], quant_std_by_step[t])}
        per_step[str(t)] = entry
        pooled.append(torch.tensor(entry["all"]))
    if pooled:
        all_alpha = torch.stack(pooled).mean(dim=0).tolist()
    elif teacher_std_by_step:
        first = next(iter(teacher_std_by_step.values()))
        all_alpha = [1.0] * int(first.shape[0])
    else:
        all_alpha = []
    return {"all": all_alpha, "steps": per_step}


def compute_per_step_beta(
    teacher_rms_by_step: Dict[int, torch.Tensor],
    quant_rms_by_step: Dict[int, torch.Tensor],
    log_clamp: float = 0.30,
    neutral: float = 0.03,
) -> Dict[str, Any]:
    """Build the v2 per-step OHB beta entry (per-head, QuantVLA v2 P2-G).

    Returns {"beta_perhead": [...], "steps": {"<t>": {"beta_perhead": [...]}, ...}}.
    """
    steps = sorted(set(teacher_rms_by_step) & set(quant_rms_by_step))

    def beta_of(t_rms: torch.Tensor, q_rms: torch.Tensor) -> List[float]:
        out = []
        for h in range(t_rms.shape[0]):
            t = max(float(t_rms[h]), 1e-8)
            q = max(float(q_rms[h]), 1e-8)
            rho = q / t
            log_beta = -math.log(max(rho, 1e-8))
            log_beta = max(-log_clamp, min(log_clamp, log_beta))
            out.append(1.0 if abs(log_beta) < neutral else math.exp(log_beta))
        return out

    per_step: Dict[str, Dict[str, List[float]]] = {}
    pooled: List[torch.Tensor] = []
    for t in steps:
        entry = {"beta_perhead": beta_of(teacher_rms_by_step[t], quant_rms_by_step[t])}
        per_step[str(t)] = entry
        pooled.append(torch.tensor(entry["beta_perhead"]))
    all_beta = torch.stack(pooled).mean(dim=0).tolist() if pooled else []
    return {"beta_perhead": all_beta, "steps": per_step}


def ensure_dit_attention_patch(model: torch.nn.Module, scope: str = "dit") -> None:
    """Replace attention processors for DiT attention layers with ATM-enabled processor.

    Also installs the denoising-step getter (QuantVLA v2) so per-step scaling
    and per-step statistics can read `action_head._current_denoise_step`.
    """
    action_head = getattr(model, "action_head", None)

    for name, module in model.named_modules():
        if _is_dit_attention(name, module, scope=scope):
            if not getattr(module, _ATM_PATCH_FLAG, False):
                module.set_processor(_ATMProcessor())
                setattr(module, _ATM_PATCH_FLAG, True)
            if action_head is not None:
                setattr(
                    module,
                    "_atm_step_getter",
                    lambda: getattr(action_head, "_current_denoise_step", None),
                )


def register_atm_capture(
    model: torch.nn.Module,
    callback: Callable[[Attention, torch.Tensor], None],
    scope: str = "dit",
) -> None:
    for name, module in model.named_modules():
        if _is_dit_attention(name, module, scope=scope):
            setattr(module, "_atm_capture_callback", lambda attn, std, layer=name: callback(layer, std))
            setattr(module, "_atm_capture_name", name)


def register_atm_capture_step(
    model: torch.nn.Module,
    callback: Callable[[str, torch.Tensor, Optional[int]], None],
    scope: str = "dit",
) -> None:
    """Step-aware per-head logits-std capture (QuantVLA v2 P2-G).

    callback receives (layer_name, std (heads,), denoise_step t_discretized).
    """
    for name, module in model.named_modules():
        if _is_dit_attention(name, module, scope=scope):
            setattr(
                module,
                "_atm_capture_step_callback",
                lambda attn, std, step, layer=name: callback(layer, std, step),
            )
            setattr(module, "_atm_capture_step_name", name)


def register_atm_logits_capture(
    model: torch.nn.Module,
    callback: Callable[[str, torch.Tensor], None],
    scope: str = "dit",
) -> None:
    for name, module in model.named_modules():
        if _is_dit_attention(name, module, scope=scope):
            setattr(
                module,
                "_atm_logits_capture_callback",
                lambda attn, logits, layer=name: callback(layer, logits),
            )
            setattr(module, "_atm_logits_capture_name", name)


def register_ohb_capture(
    model: torch.nn.Module,
    callback: Callable[[Attention, torch.Tensor], None],
    scope: str = "dit",
) -> None:
    for name, module in model.named_modules():
        if _is_dit_attention(name, module, scope=scope):
            setattr(module, "_atm_ohb_capture_callback", lambda attn, rms, layer=name: callback(layer, rms))
            setattr(module, "_atm_ohb_capture_name", name)


def register_ohb_perhead_capture(
    model: torch.nn.Module,
    callback: Callable[[str, torch.Tensor], None],
    scope: str = "dit",
) -> None:
    """Register per-head OHB capture callback.

    The callback receives (layer_name, rms_per_head) where rms_per_head has shape (heads,).
    """
    for name, module in model.named_modules():
        if _is_dit_attention(name, module, scope=scope):
            setattr(
                module,
                "_atm_ohb_perhead_capture_callback",
                lambda attn, rms, layer=name: callback(layer, rms),
            )
            setattr(module, "_atm_ohb_perhead_capture_name", name)


def register_ohb_perhead_capture_step(
    model: torch.nn.Module,
    callback: Callable[[str, torch.Tensor, Optional[int]], None],
    scope: str = "dit",
) -> None:
    """Step-aware per-head RMS capture (QuantVLA v2 P2-G).

    callback receives (layer_name, rms_per_head (heads,), denoise_step).
    """
    for name, module in model.named_modules():
        if _is_dit_attention(name, module, scope=scope):
            setattr(
                module,
                "_atm_ohb_perhead_capture_step_callback",
                lambda attn, rms, step, layer=name: callback(layer, rms, step),
            )
            setattr(module, "_atm_ohb_perhead_capture_step_name", name)


def register_output_capture(
    model: torch.nn.Module,
    callback: Callable[[str, torch.Tensor, Optional[int]], None],
    scope: str = "dit",
) -> None:
    """Register per-layer output-tensor capture (QuantVLA v2 CKA/CS probing).

    The callback receives (layer_name, output_tensor, denoise_step):
    - layer_name: qualified module name, e.g. "action_head.model.transformer_blocks.0.attn1"
    - output_tensor: the attention module's final output (B, seq, dim), detached
    - denoise_step: current t_discretized value, or None when the step getter is
      unavailable (e.g. outside the denoising loop).
    """
    action_head = getattr(model, "action_head", None)
    for name, module in model.named_modules():
        if _is_dit_attention(name, module, scope=scope):
            setattr(
                module,
                "_atm_output_capture_callback",
                lambda attn, out, step, layer=name: callback(layer, out, step),
            )
            setattr(module, "_atm_output_capture_name", name)
            if action_head is not None:
                setattr(
                    module,
                    "_atm_step_getter",
                    lambda: getattr(action_head, "_current_denoise_step", None),
                )


def clear_atm_capture(model: torch.nn.Module) -> None:
    for _, module in model.named_modules():
        if isinstance(module, Attention):
            if hasattr(module, "_atm_capture_callback"):
                delattr(module, "_atm_capture_callback")
            if hasattr(module, "_atm_capture_name"):
                delattr(module, "_atm_capture_name")
            if hasattr(module, "_atm_capture_step_callback"):
                delattr(module, "_atm_capture_step_callback")
            if hasattr(module, "_atm_capture_step_name"):
                delattr(module, "_atm_capture_step_name")
            if hasattr(module, "_atm_logits_capture_callback"):
                delattr(module, "_atm_logits_capture_callback")
            if hasattr(module, "_atm_logits_capture_name"):
                delattr(module, "_atm_logits_capture_name")
            if hasattr(module, "_atm_ohb_capture_callback"):
                delattr(module, "_atm_ohb_capture_callback")
            if hasattr(module, "_atm_ohb_capture_name"):
                delattr(module, "_atm_ohb_capture_name")
            if hasattr(module, "_atm_ohb_perhead_capture_callback"):
                delattr(module, "_atm_ohb_perhead_capture_callback")
            if hasattr(module, "_atm_ohb_perhead_capture_name"):
                delattr(module, "_atm_ohb_perhead_capture_name")
            if hasattr(module, "_atm_ohb_perhead_capture_step_callback"):
                delattr(module, "_atm_ohb_perhead_capture_step_callback")
            if hasattr(module, "_atm_ohb_perhead_capture_step_name"):
                delattr(module, "_atm_ohb_perhead_capture_step_name")
            if hasattr(module, "_atm_output_capture_callback"):
                delattr(module, "_atm_output_capture_callback")
            if hasattr(module, "_atm_output_capture_name"):
                delattr(module, "_atm_output_capture_name")
            if hasattr(module, "_atm_step_getter"):
                delattr(module, "_atm_step_getter")


@dataclass
class _AlphaSummary:
    matched_layers: int = 0
    total_heads: int = 0


def enable_dit_atm_if_configured(model: torch.nn.Module) -> None:
    atm_flag = os.environ.get(ATM_ENABLE_ENV, "0")
    ohb_flag = os.environ.get(OHB_ENABLE_ENV, "0")
    atm_enabled = atm_flag not in ("0", "false", "False", "")
    ohb_enabled = ohb_flag not in ("0", "false", "False", "")
    if not atm_enabled and not ohb_enabled:
        return

    alpha_path = os.environ.get(ATM_ALPHA_ENV)
    if not alpha_path:
        print("[GR00T-ATM] Scaling requested but GR00T_ATM_ALPHA_PATH not set; skipping.")
        return

    if not os.path.exists(alpha_path):
        print(f"[GR00T-ATM] Alpha JSON not found at {alpha_path}; skipping ATM.")
        return

    with open(alpha_path, "r", encoding="utf-8") as f:
        alpha_data = json.load(f)

    scope = os.environ.get(ATM_SCOPE_ENV, "dit")
    ohb_scope = os.environ.get(OHB_SCOPE_ENV, None)
    if ohb_scope is None:
        ohb_only_dit = os.environ.get(OHB_ONLY_DIT_ENV, "1") not in ("0", "false", "False")
        ohb_scope = "dit" if ohb_only_dit else scope
    summary = _AlphaSummary()
    ohb_layers = 0
    ohb_fallback = float(os.environ.get(OHB_FALLBACK_ENV, "1.0"))

    ensure_dit_attention_patch(model, scope=scope)

    for name, module in model.named_modules():
        if not _is_dit_attention(name, module, scope=scope):
            continue
        alpha_entry = alpha_data.get(name) or alpha_data.get(name.replace("model.", "model", 1))
        if not alpha_entry:
            beta_value = None
            alpha_values = None
        else:
            alpha_values = alpha_entry.get("all") or alpha_entry.get("alpha")
            beta_value = alpha_entry.get("beta")

        # ---- QuantVLA v2 P2-G: per-step tables (GR00T_ATM_PER_STEP=1) ----
        per_step = os.environ.get(ATM_PER_STEP_ENV, "0") not in ("0", "false", "False", "")
        if per_step:
            steps_entry = (alpha_entry or {}).get("steps") or {}
            if atm_enabled and steps_entry:
                alpha_by_step = {
                    int(t): torch.tensor(entry.get("all"), dtype=torch.float32)
                    for t, entry in steps_entry.items()
                    if entry.get("all")
                }
                if alpha_by_step:
                    setattr(module, "_atm_alpha_by_step", alpha_by_step)
            if ohb_enabled and steps_entry:
                beta_perhead_by_step = {
                    int(t): torch.tensor(entry.get("beta_perhead"), dtype=torch.float32)
                    for t, entry in steps_entry.items()
                    if entry.get("beta_perhead") is not None
                }
                if beta_perhead_by_step:
                    setattr(module, "_ohb_beta_perhead_by_step", beta_perhead_by_step)
                beta_by_step = {
                    int(t): float(entry.get("beta"))
                    for t, entry in steps_entry.items()
                    if entry.get("beta") is not None
                }
                if beta_by_step:
                    setattr(module, "_ohb_beta_by_step", beta_by_step)

        if atm_enabled and alpha_values:
            alpha_tensor = torch.tensor(alpha_values, dtype=torch.float32)
            setattr(module, "_atm_alpha_all", alpha_tensor)
            summary.matched_layers += 1
            summary.total_heads += len(alpha_values)

        if ohb_enabled and _is_dit_attention(name, module, scope=ohb_scope):
            # Check for per-head beta first
            beta_perhead_values = alpha_entry.get("beta_perhead") if alpha_entry else None
            if beta_perhead_values is not None:
                # Per-head OHB
                beta_tensor = torch.tensor(beta_perhead_values, dtype=torch.float32)
                setattr(module, "_ohb_beta_perhead", beta_tensor)
                ohb_layers += 1
            else:
                # Per-layer OHB (fallback)
                beta = float(beta_value) if beta_value is not None else ohb_fallback
                setattr(module, "_ohb_beta_scalar", beta)
                ohb_layers += 1

    if summary.matched_layers == 0 and atm_enabled:
        print(f"[GR00T-ATM] No attention layers matched alpha JSON ({alpha_path}).")
    elif atm_enabled:
        per_step_note = " [per-step]" if os.environ.get(ATM_PER_STEP_ENV, "0") not in ("0", "false", "False", "") else ""
        print(
            f"[GR00T-ATM] ATM enabled for {summary.matched_layers} layers "
            f"({summary.total_heads} heads) using {alpha_path}{per_step_note}"
        )

    if ohb_enabled:
        if ohb_layers == 0:
            print(f"[GR00T-ATM] OHB requested but no layers found (scope={ohb_scope}); fallback beta={ohb_fallback}")
        else:
            print(f"[GR00T-ATM] OHB enabled for {ohb_layers} layers using {alpha_path}")
