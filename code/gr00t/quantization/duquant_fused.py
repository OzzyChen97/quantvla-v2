#!/usr/bin/env python3
"""Triton fused W4-dequant matmul for the GR00T DuQuant path (v1.4 fast-path probe).

The eager DuQuantLinear computes, every forward:

    y = linear(x_t, fake_quantize_sym(W_t, w_scales[:, None], 4))

i.e. weights are dequantized to float on the fly and multiplied in fp16/bf16
tensor cores. This module provides the SAME numerical semantics as a single
fused Triton kernel:

    * W_t is rounded/clamped ONCE into int8 per output row (identical
      rounding to torch.round in fake_quantize_sym),
    * the kernel dequantizes inside (w = w_q * scale) and runs one
      tl.dot accumulation,
    * activation A8 fake-quant, input rotations (R_in), output restore (R_out)
      and bias stay EXACTLY as the eager path — so the A/B difference isolates
      the matmul path only.

Accumulation is fp32 in the kernel (Triton dot semantics); the eager path uses
cuBLAS tensor-core accumulation, so bit-equality is not expected — the A/B
script measures the actual divergence on the real model.

Usage:
    python -m gr00t.quantization.duquant_fused            # selftest (CPU/GPU)
    GR00T_DUQUANT_FUSED=1 ...                              # enable in deployment
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _w4_dequant_matmul_kernel(
    A,  # (M, K) fp16/bf16 — already input-transformed + A8 fake-quantized
    WQ,  # (N, K) int8 quantized weights
    WS,  # (N,) per-output-row scales
    Y,  # (M, N) out
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_wn,
    stride_wk,
    stride_ym,
    stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    wq_ptrs = WQ + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - _ * BLOCK_K, other=0.0)
        wq = tl.load(wq_ptrs, mask=offs_k[None, :] < K - _ * BLOCK_K, other=0)
        ws = tl.load(WS + offs_n, mask=offs_n < N, other=1.0)
        w = wq.to(tl.float32) * ws[:, None]
        acc += tl.dot(a, tl.trans(w).to(a.dtype), out_dtype=tl.float32)
        a_ptrs += BLOCK_K * stride_ak
        wq_ptrs += BLOCK_K * stride_wk

    y_ptrs = Y + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc.to(Y.dtype.element_ty), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def pack_w4_int8(w_t: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Round+clamp W_t/scale to the same grid fake_quantize_sym uses, as int8.

    fake_quantize_sym clamps to [-max_q-1, max_q] with max_q = 7 for 4 bits,
    i.e. [-8, 7] — the asymmetric negative bucket is preserved exactly.
    """
    max_q = 7
    q = torch.clamp(torch.round(w_t / scales[:, None]), -max_q - 1, max_q)
    return q.to(torch.int8).contiguous()


def fused_linear_w4(
    x: torch.Tensor,
    w_q: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """x @ dequant(w_q, scales)^T — Triton fused, fp32 accumulation."""
    orig_dtype = x.dtype
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    wq2 = w_q.contiguous()
    m, k = x2.shape
    n = wq2.shape[0]
    y = torch.empty((m, n), dtype=orig_dtype, device=x.device)
    BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 64, 64, 64, 4
    if k % BLOCK_K != 0:
        # pad-free fallback: run the kernel with masked K (works but slower)
        pass
    grid = (triton.cdiv(m, BLOCK_M) * triton.cdiv(n, BLOCK_N),)
    _w4_dequant_matmul_kernel[grid](
        x2, wq2, scales, y, m, n, k,
        x2.stride(0), x2.stride(1), wq2.stride(0), wq2.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
    )
    return y.reshape(*x.shape[:-1], n)


def eager_linear_w4(x: torch.Tensor, w_t: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Reference eager path (same math as DuQuantLinear's weight_bits>0 branch)."""
    max_q = 7
    w_scaled = w_t / scales[:, None]
    w_deq = torch.clamp(torch.round(w_scaled), -max_q - 1, max_q) * scales[:, None]
    return torch.nn.functional.linear(x, w_deq.to(x.dtype), None)


def selftest() -> None:
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        print("[duquant_fused] no CUDA — selftest skipped (kernel needs GPU)")
        return
    for (m, n, k, dt) in ((256, 512, 1536, torch.float16), (256, 512, 1536, torch.bfloat16),
                          (37, 6144, 1536, torch.float16)):
        x = torch.randn(m, k, dtype=dt, device=dev) * 0.1
        w_t = torch.randn(n, k, dtype=torch.float32, device=dev) * 0.05
        scales = torch.rand(n, dtype=torch.float32, device=dev) * 0.1 + 0.01
        w_q = pack_w4_int8(w_t, scales)
        y_e = eager_linear_w4(x, w_t, scales)
        y_f = fused_linear_w4(x, w_q, scales)
        # fp32 reference: the exact dequant matmul in float32
        max_q = 7
        w_deq32 = torch.clamp(torch.round(w_t / scales[:, None]), -max_q - 1, max_q) * scales[:, None]
        y_ref = (x.float() @ w_deq32.T.float())
        denom = y_ref.abs() + 1e-2
        err_e = ((y_e.float() - y_ref).abs() / denom).max().item()
        err_f = ((y_f.float() - y_ref).abs() / denom).max().item()
        print(f"[duquant_fused] M={m} N={n} K={k} {dt}: eager-vs-fp32ref={err_e:.3e} "
              f"fused-vs-fp32ref={err_f:.3e}")
        # the meaningful gate: the fused path must be at least as accurate as
        # the eager tensor-core path (both vs the fp32 reference)
        assert err_f <= err_e * 1.2 + 1e-3, f"fused less accurate than eager: {err_f} vs {err_e}"
        assert err_f < 0.5, f"fused error grossly large: {err_f}"
    # timing
    x = torch.randn(256, 1536, dtype=torch.float16, device=dev) * 0.1
    w_t = torch.randn(512, 1536, dtype=torch.float32, device=dev) * 0.05
    scales = torch.rand(512, dtype=torch.float32, device=dev) * 0.1 + 0.01
    w_q = pack_w4_int8(w_t, scales)
    import time

    for _ in range(3):
        eager_linear_w4(x, w_t, scales)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(50):
        eager_linear_w4(x, w_t, scales)
    torch.cuda.synchronize()
    t_e = (time.time() - t0) / 50
    for _ in range(3):
        fused_linear_w4(x, w_q, scales)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(50):
        fused_linear_w4(x, w_q, scales)
    torch.cuda.synchronize()
    t_f = (time.time() - t0) / 50
    print(f"[duquant_fused] eager {t_e*1e3:.2f} ms vs fused {t_f*1e3:.2f} ms "
          f"({t_e/t_f:.2f}x speedup)")
    print("[duquant_fused] selftest OK")


if __name__ == "__main__":
    selftest()
