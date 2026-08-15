"""Kernel-based similarity scores for QuantVLA v2 (GR00T stack, P0-G measurement layer).

Implements the two "similarity instead of scale" metrics from the v2 design doc
(docs/quantvla_v2_design.md §5.1), both computed as FP16-reference vs quantized
differences on identical inputs:

1. Linear CKA (centered kernel alignment)
   Dasgupta & Cohn, ICLR 2025, Eq.(2):
       CKA(H_S, H_T) = ||Sigma_TS||_F^2 / ( ||Sigma_TT||_F * ||Sigma_SS||_F )
   Scale-invariant: multiplying either side by a scalar does not change the score,
   so it measures representation *shape* preservation, not scale matching.

2. CS divergence (uncentered Cauchy-Schwarz divergence, KDE estimator)
   Yin et al., ICLR 2026 (CS-Aligner), Eq.(7)/(8):
       D_CS(p; q) = log<(1/M^2) sum sum k(x_i, x_j)>
                    + log<(1/N^2) sum sum k(y_i, y_j)>
                    - 2 * log<(1/(M*N)) sum sum k(x_i, y_j)>
   with Gaussian kernel k(x, y) = exp(-||x - y||^2 / (2 sigma^2)),
   sigma fixed from the REFERENCE samples (median heuristic, floored at 1e-3).

Semantics (v1.2): the "reference" is NOT necessarily the pure FP16 model. In the
sensitivity probe the reference is the quantized pipeline with every target
layer at weight_bits=0 (weights unquantized; rotations / permutation / A8
activation quantization still active). This makes single-layer intervention
confound-free: both sides share the same wrapper and upstream behavior, and
only the target layer's weight quantization differs. The pure FP16 model is
used only for the deploy-relevant global D_solver pairing.

Precise statements:
- D_CS in [0, +inf] (extended reals): diverges when the cross term -> 0, i.e.
  it is NOT finite under zero overlap; its advantage is estimability under
  kernel smoothing, not finiteness.
- The FORMULA is symmetric; the ESTIMATOR is directed (reference-fixed
  bandwidth), so this is a reference-based evaluation, not a symmetric metric.
- Token/step samples are not i.i.d., so these are heuristic ranking scores,
  not the paper's KDE estimator in the strict sense.
- Linear CKA uses D×D covariances (hidden-dimension sized); the Gaussian-kernel
  CS term uses N×N Gram matrices (sample sized). They share pooling/interface
  code, NOT matrices.

Usage pattern (see scripts/tools/gr00t_sensitivity_probe.py):

    bank = LayerScoreBank(name)
    bank.accumulate_ref(ref_out)   # during the REFERENCE pass, once per batch
    bank.finalize_ref()            # computes reference self terms + bandwidth
    ...
    scores = bank.evaluate(q_out)  # per quant config -> {"cka": float, "cs": float}

The reference-side terms (Sigma_TT norm, kernel self term, bandwidth sigma) are
cached so that scanning many quant configs only costs the cross terms.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch


def _stride_subsample(t: torch.Tensor, max_tokens: int) -> torch.Tensor:
    """Deterministic uniform subsample of rows down to max_tokens."""
    n = t.shape[0]
    if n <= max_tokens:
        return t
    step = n // max_tokens + 1
    return t[::step][:max_tokens].contiguous()


def pool_samples(t: torch.Tensor, max_tokens: int) -> Optional[torch.Tensor]:
    """Flatten (..., D) -> (N, D) float32 CPU, cap N at max_tokens.

    Returns None when the tensor has fewer than 2 usable rows.
    """
    if t is None:
        return None
    t = t.detach().to(torch.float32).reshape(-1, t.shape[-1]).cpu()
    if t.shape[0] < 2:
        return None
    t = _stride_subsample(t, max_tokens)
    return t


def _center(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(dim=0, keepdim=True)


def _fro2(mat: torch.Tensor) -> float:
    """Squared Frobenius norm of a possibly-large matrix (float)."""
    return float(torch.linalg.matrix_norm(mat, ord="fro")) ** 2


def _log_mean_gaussian_kernel(
    x: torch.Tensor,
    y: Optional[torch.Tensor],
    sigma: float,
    block: int = 256,
) -> float:
    """log( (1/(M*N)) sum_i sum_j exp(-||x_i - y_j||^2 / (2 sigma^2)) ).

    Computed in log space with block-wise logsumexp so large M*N*D products
    never materialize. When y is None, x is compared with itself.
    """
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    xd = x.to(dev)
    yd = xd if y is None else y.to(dev)
    two_s2 = 2.0 * sigma * sigma
    total: Optional[torch.Tensor] = None
    for i in range(0, xd.shape[0], block):
        d2 = torch.cdist(xd[i : i + block], yd, p=2).square()
        vals = (-d2 / two_s2).logsumexp(dim=(0, 1))
        total = vals if total is None else torch.logaddexp(total, vals)
    assert total is not None
    log_mean = float(total) - math.log(x.shape[0] * yd.shape[0])
    return log_mean


class LayerScoreBank:
    """Per-layer reference score bank: linear CKA + CS divergence.

    "Reference" = the v1.2 reference protocol (wrapped pipeline with every
    target layer at weight_bits=0), see module docstring.
    """

    def __init__(self, name: str, max_tokens: int = 1024, eps: float = 1e-12):
        self.name = name
        self.max_tokens = max_tokens
        self.eps = eps

        # Reference-side state (accumulated during the reference pass)
        self._fp_chunks: List[torch.Tensor] = []
        self._fp_raw: Optional[torch.Tensor] = None  # (M, D), uncentered (for CS)
        self._fp_centered: Optional[torch.Tensor] = None  # (M, D), column-centered (for CKA)
        self._fp_self_norm: Optional[float] = None  # ||Sigma_TT||_F (for CKA)
        self._log_xx: Optional[float] = None  # log <kappa(x,x)> (for CS)
        self.sigma: Optional[float] = None  # Gaussian kernel bandwidth (fixed from FP)

    # ------------------------------------------------------------------ #
    # FP16 pass
    # ------------------------------------------------------------------ #
    def accumulate_ref(self, t: torch.Tensor) -> None:
        pooled = pool_samples(t, self.max_tokens)
        if pooled is not None:
            self._fp_chunks.append(pooled)

    def finalize_ref(self) -> None:
        if self._fp_centered is not None:
            return
        if not self._fp_chunks:
            return
        x = torch.cat(self._fp_chunks, dim=0)
        self._fp_chunks = []
        x = _stride_subsample(x, self.max_tokens)
        if x.shape[0] < 2:
            return

        # Bandwidth: median pairwise distance among (a subset of) FP samples.
        # Fixed from FP once, reused for every quant config so scores are comparable.
        self.sigma = self._estimate_sigma(x)

        xc = _center(x)
        self._fp_raw = x
        self._fp_centered = xc
        sss = xc.T @ xc
        self._fp_self_norm = float(torch.linalg.matrix_norm(sss, ord="fro"))

        self._log_xx = _log_mean_gaussian_kernel(x, None, self.sigma)

    def _estimate_sigma(self, x: torch.Tensor, n_pairs: int = 2048) -> float:
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        xd = x.to(dev)
        n = xd.shape[0]
        if n < 2:
            return 1.0
        gen = torch.Generator(device=dev).manual_seed(0)
        i = torch.randint(0, n, (n_pairs,), generator=gen, device=dev)
        j = torch.randint(0, n, (n_pairs,), generator=gen, device=dev)
        same = i == j
        j = torch.where(same, (j + 1) % n, j)
        d = torch.norm(xd[i] - xd[j], dim=1)
        sigma = float(d.median())
        return max(sigma, 1e-3)

    # ------------------------------------------------------------------ #
    # Quant pass
    # ------------------------------------------------------------------ #
    def evaluate(self, q_out: torch.Tensor) -> dict:
        """Score one quantized-config output tensor against the cached FP16 reference.

        Returns {"cka": float, "cs": float}; values are None when the layer had
        no usable FP samples.
        """
        result: dict = {"cka": None, "cs": None}
        if self._fp_centered is None or self.sigma is None:
            return result

        y = pool_samples(q_out, self.max_tokens)
        if y is None:
            return result

        # --- linear CKA ---
        yc = _center(y)
        sts = self._fp_centered.T @ yc  # Sigma_TS (transposed)
        stt_norm = self._fp_self_norm
        sss_norm = float(torch.linalg.matrix_norm(yc.T @ yc, ord="fro"))
        denom = stt_norm * sss_norm
        if denom > self.eps:
            cka = _fro2(sts) / denom
            result["cka"] = float(min(max(cka, 0.0), 1.0))

        # --- CS divergence ---
        log_yy = _log_mean_gaussian_kernel(y, None, self.sigma)
        log_xy = _log_mean_gaussian_kernel(self._fp_raw, y, self.sigma)
        cs = self._log_xx + log_yy - 2.0 * log_xy
        result["cs"] = float(max(cs, 0.0))

        return result

    @property
    def ready(self) -> bool:
        return self._fp_centered is not None and self.sigma is not None

    def state_dict(self) -> dict:
        """Compact diagnostics (for JSON export)."""
        return {
            "name": self.name,
            "n_fp_tokens": 0 if self._fp_centered is None else int(self._fp_centered.shape[0]),
            "sigma": self.sigma,
        }


def selftest() -> None:
    """Numerical sanity checks (run on CPU): python -m gr00t.quantization.kernel_scores"""
    torch.manual_seed(0)
    x = torch.randn(256, 64)

    bank = LayerScoreBank("t", max_tokens=512)
    bank.accumulate_ref(x)
    bank.finalize_ref()
    assert bank.ready, "bank failed to finalize"

    s_same = bank.evaluate(x)
    assert s_same["cka"] is not None and abs(s_same["cka"] - 1.0) < 1e-3, f"CKA(X,X)={s_same['cka']}"
    assert s_same["cs"] is not None and s_same["cs"] < 1e-3, f"CS(X,X)={s_same['cs']}"

    s_scaled = bank.evaluate(x * 3.7)  # isotropic scaling must not change CKA
    assert s_scaled["cka"] is not None and abs(s_scaled["cka"] - 1.0) < 1e-2, (
        f"CKA(X,3.7X)={s_scaled['cka']} (scale invariance violated)"
    )
    assert s_scaled["cs"] is not None and s_scaled["cs"] > 1e-2, f"CS(X,3.7X)={s_scaled['cs']}"

    y = torch.randn(256, 64) * 0.1  # shrunk, differently-shaped distribution
    s_other = bank.evaluate(y)
    assert s_other["cka"] is not None and s_other["cka"] < 0.99, f"CKA(X,0.1Y)={s_other['cka']}"
    assert s_other["cs"] is not None and s_other["cs"] > 0.0, f"CS(X,0.1Y)={s_other['cs']}"

    # CKA of a permuted/shuffled variant should be well below 1
    xp = x[torch.randperm(256)]
    s_perm = bank.evaluate(xp)
    assert s_perm["cka"] is not None and s_perm["cka"] < 0.9, f"CKA(X,perm(X))={s_perm['cka']}"

    print("[kernel_scores] selftest OK")
    print(f"  CKA(X,X)      = {s_same['cka']:.6f}")
    print(f"  CKA(X,3.7X)   = {s_scaled['cka']:.6f}   (scale-invariant, expect ~1)")
    print(f"  CKA(X,0.1Y)   = {s_other['cka']:.6f}")
    print(f"  CKA(X,permX)  = {s_perm['cka']:.6f}")
    print(f"  CS(X,X)       = {s_same['cs']:.6f}")
    print(f"  CS(X,3.7X)    = {s_scaled['cs']:.6f}")
    print(f"  CS(X,0.1Y)    = {s_other['cs']:.6f}")


if __name__ == "__main__":
    selftest()
