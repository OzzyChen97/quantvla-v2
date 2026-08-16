#!/usr/bin/env python3
"""Tail-aware functional metrics for QuantVLA v1.4 (D-020 route 3).

The v1.3 D_solver is a late-step-weighted MEAN relative error; the v1.3
LIBERO full test showed its ranking does not transfer consistently to
closed-loop success (v2 4.7-8.5x better on D_solver, +/-0.8-1.2 sigma on SR).
v1.4 replaces it with D_func, a tail-aware functional metric computed from the
SAME paired denoising trajectories (T+1, B, H, D):

  d_final    final action-chunk deviation (last denoising step, relative)
  d_kin      per-dim kinematic errors: translation (dims 0:3), rotation
             (3:6), gripper (6:6+g) — layout configurable per embodiment
  d_grip     gripper sign-mismatch rate: fraction of steps where the
             ref and quantized gripper deltas disagree in sign (binarization
             proxy for grasp/contact transitions)
  tail       p90 / p95 / CVaR0.9 of the per-obs divergences
  grasp-w    grasp-window weighting: steps with large gripper-state change
             (contact/grasp transition proxy) upweighted in the mean

  D_func = w_final*d_final + w_kin*d_kin + w_grip*d_grip + w_tail*CVaR0.9
  default weights 1/1/1/2 (tail emphasized); frozen for the experiment.

All components come from trajectory pairs the probe/scorer already collect —
no new rollouts. The end-effector-Jacobian weighting from the plan is replaced
by the grasp-window proxy here because the synthetic measurement protocol has
no EE Jacobian; that substitution is documented in the report.

Usage:
    python -m gr00t_func_metrics            # selftest
    from gr00t_func_metrics import d_func   # (ref_traj, q_traj, gamma) -> dict
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch

# LIBERO embodiment: [dx, dy, dz, droll, dpitch, dyaw, gripper(, gripper2)]
DEFAULT_LAYOUT = {"trans": (0, 3), "rot": (3, 6), "grip": (6, 8)}
DEFAULT_WEIGHTS = {"final": 1.0, "kin": 1.0, "grip": 1.0, "tail": 2.0}


def per_obs_divergences(ref: torch.Tensor, q: torch.Tensor, gamma: float) -> List[float]:
    """Late-step-weighted per-obs relative divergence (v1.3 D_solver inputs)."""
    ref = ref.float()[:, : q.shape[1]]
    q = q.float()
    num = ((ref - q) ** 2).sum(dim=(-1, -2))
    den = (ref**2).sum(dim=(-1, -2)).clamp_min(1e-8)
    rel = num / den
    k_steps = rel.shape[0]
    weights = torch.tensor([gamma ** (k + 1) for k in range(k_steps)])
    weights = weights / weights.sum()
    per_obs = (rel * weights[:, None]).sum(dim=0)
    return [float(v) for v in per_obs]


def tail_stats(per_obs: List[float]) -> Dict[str, float]:
    """p90 / p95 / CVaR0.9 of the per-obs divergence list."""
    a = np.asarray(per_obs, dtype=np.float64)
    if a.size == 0:
        return {"p90": 0.0, "p95": 0.0, "cvar90": 0.0, "n": 0}
    p90 = float(np.quantile(a, 0.90))
    p95 = float(np.quantile(a, 0.95))
    tail = a[a >= p90]
    cvar = float(tail.mean()) if tail.size else p90
    return {"p90": p90, "p95": p95, "cvar90": cvar, "n": int(a.size)}


def final_action_deviation(ref: torch.Tensor, q: torch.Tensor) -> float:
    """Relative error of the FINAL denoising step's action chunk."""
    r = ref.float()[-1]
    s = q.float()[-1]
    num = float(((r - s) ** 2).sum())
    den = float((r**2).sum().clamp_min(1e-8))
    return num / den


def per_dim_errors(ref: torch.Tensor, q: torch.Tensor, layout: Dict[str, tuple]) -> Dict[str, float]:
    """Per-dimension-group relative errors on the final action chunk."""
    r = ref.float()[-1]  # (B, H, D)
    s = q.float()[-1]
    out: Dict[str, float] = {}
    for grp, (lo, hi) in layout.items():
        if lo >= r.shape[-1]:
            out[grp] = 0.0
            continue
        hi = min(hi, r.shape[-1])
        num = float(((r[..., lo:hi] - s[..., lo:hi]) ** 2).sum())
        den = float((r[..., lo:hi] ** 2).sum().clamp_min(1e-8))
        out[grp] = num / den
    return out


def gripper_sign_mismatch(ref: torch.Tensor, q: torch.Tensor, grip_idx: tuple) -> Dict[str, float]:
    """Fraction of (obs, step) pairs whose gripper delta sign disagrees.

    Uses per-obs deltas over the trajectory (steps 1..T+1); a step counts when
    BOTH deltas exceed eps (active gripper motion) and their signs differ.
    """
    r = ref.float()
    s = q.float()
    lo, hi = grip_idx
    if lo >= r.shape[-1]:
        return {"rate": 0.0, "n_active": 0}
    hi = min(hi, r.shape[-1])
    dr = (r[1:, ..., lo:hi] - r[:-1, ..., lo:hi]).sum(dim=-1)  # (T, B)
    ds = (s[1:, ..., lo:hi] - s[:-1, ..., lo:hi]).sum(dim=-1)
    eps = 1e-4
    active = (dr.abs() > eps) & (ds.abs() > eps)
    mismatch = active & (dr.sign() != ds.sign())
    n_active = int(active.sum())
    n_mis = int(mismatch.sum())
    return {"rate": (n_mis / n_active) if n_active else 0.0, "n_active": n_active}


def grasp_window_weight(ref: torch.Tensor, grip_idx: tuple) -> torch.Tensor:
    """Per-(obs) grasp-window weight: 1 + |gripper delta| normalized (contact
    transition proxy). Shape (B,)."""
    r = ref.float()
    lo, hi = grip_idx
    if lo >= r.shape[-1]:
        return torch.ones(r.shape[1])
    hi = min(hi, r.shape[-1])
    d = (r[1:, ..., lo:hi] - r[:-1, ..., lo:hi]).abs().sum(dim=-1)  # (T, B, H)
    d = d.sum(dim=(0, 2))  # (B,) total gripper motion per obs
    mx = d.max().clamp_min(1e-8)
    return 1.0 + (d / mx)


def d_func(
    ref: torch.Tensor,
    q: torch.Tensor,
    gamma: float = 1.2,
    layout: Optional[Dict[str, tuple]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Combined tail-aware functional metric + all components."""
    layout = layout or DEFAULT_LAYOUT
    weights = weights or DEFAULT_WEIGHTS
    per_obs = per_obs_divergences(ref, q, gamma)
    tail = tail_stats(per_obs)
    d_fin = final_action_deviation(ref, q)
    dims = per_dim_errors(ref, q, layout)
    d_kin = (dims["trans"] + dims["rot"]) / 2.0
    d_grip = gripper_sign_mismatch(ref, q, layout["grip"])["rate"]
    # grasp-window weighted mean over per-obs (proxy for contact weighting)
    gw = grasp_window_weight(ref, layout["grip"])
    gw = gw / gw.mean()  # normalize to mean 1
    d_mean = float((torch.tensor(per_obs, dtype=torch.float32) * gw).mean())
    combined = (
        weights["final"] * d_fin
        + weights["kin"] * d_kin
        + weights["grip"] * d_grip
        + weights["tail"] * tail["cvar90"]
    )
    return {
        "d_func": combined,
        "d_final": d_fin,
        "d_kin": d_kin,
        "d_grip": d_grip,
        "d_mean": d_mean,
        "d_solver": float(np.mean(per_obs)),
        "d_solver_std": float(np.std(per_obs)) if len(per_obs) > 1 else 0.0,
        "tail": tail,
        "per_dim": dims,
        "weights": dict(weights),
        "per_obs": per_obs,
    }


def selftest() -> None:
    torch.manual_seed(0)
    t = torch.randn(9, 4, 16, 7)  # T+1=9, B=4, H=16, D=7 (libero layout)
    # identical -> everything 0
    d0 = d_func(t, t)
    assert d0["d_func"] == 0.0 and d0["d_final"] == 0.0 and d0["d_kin"] == 0.0
    assert d0["d_grip"] == 0.0 and d0["tail"]["cvar90"] == 0.0
    # uniform scaling -> final/deviation components exactly 1 (relative error)
    d2 = d_func(t, t * 2.0)
    assert abs(d2["d_final"] - 1.0) < 1e-5, d2["d_final"]
    assert abs(d2["d_solver"] - 1.0) < 1e-6, d2["d_solver"]  # P0-3 semantics
    # tail stats on a known list
    a = list(range(100))
    ts = tail_stats(a)
    assert abs(ts["p90"] - 89.1) < 1e-6 and abs(ts["p95"] - 94.05) < 1e-6 and abs(ts["cvar90"] - 94.5) < 1e-6, ts
    # gripper sign flip: q gripper = -ref gripper delta every step -> rate 1
    q = t.clone()
    q[1:, ..., 6] = -t[1:, ..., 6]  # deltas negate (ref delta d -> -d... set q step = -ref delta)
    gm = gripper_sign_mismatch(t, q, (6, 7))
    assert gm["n_active"] > 0 and gm["rate"] > 0.9, gm
    # per-dim: only translation corrupted -> trans error large, rot/grip 0
    q2 = t.clone()
    q2[-1, ..., 0:3] = t[-1, ..., 0:3] * 5.0
    pd = per_dim_errors(t, q2, DEFAULT_LAYOUT)
    assert pd["trans"] > 0.5 and pd["rot"] == 0.0 and pd["grip"] == 0.0, pd
    # combined D_func positive and >= mean component when tail dominates
    d3 = d_func(t, q2)
    assert d3["d_func"] > 0.0 and d3["per_dim"]["trans"] > 0.5
    print("[gr00t_func_metrics] selftest OK")
    print(f"  identical: d_func={d0['d_func']} d_solver={d0['d_solver']}")
    print(f"  2x scaled: d_final={d2['d_final']:.6f} d_solver={d2['d_solver']:.6f} (expect 1)")
    print(f"  trans-only corruption: d_func={d3['d_func']:.4f} trans={d3['per_dim']['trans']:.4f}")
    print(f"  gripper sign-flip rate: {gm['rate']:.4f} (n_active={gm['n_active']})")
    print(f"  tail(0..99): p90={ts['p90']} p95={ts['p95']} cvar90={ts['cvar90']}")


if __name__ == "__main__":
    selftest()
