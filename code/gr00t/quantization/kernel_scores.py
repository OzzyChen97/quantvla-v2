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


def pool_samples_stratified(
    t: torch.Tensor,
    max_tokens: int,
    front_frac: float = 0.5,
) -> Optional[torch.Tensor]:
    """Position-stratified token pooling — POSITIONAL ONLY, no data-dependent
    row dropping (pairing-correctness; review round 4).

    GR00T's LLM/DiT token axis follows the vision-prefix layout: leading tokens
    are vision-dominant, trailing tokens are text/state-dominant. Flat pooling
    lets the (much larger) vision block drown out the action-relevant tokens in
    CKA/CS. This version splits each sequence at fraction `front_frac` and caps
    EACH block at max_tokens//2 with deterministic stride subsampling.

    IMPORTANT: the selection is purely positional so the REFERENCE and the
    quantized pools always keep the SAME token rows (CKA/CS require H_FP and
    H_q to be the same batch of tokens). Padding masking is handled by the
    bank's REF-shared row mask (accumulate_ref_blocks / evaluate_blocks), never
    by a per-tensor norm threshold — a data-dependent drop desyncs the row
    indices between ref and q (a near-zero padding row can drift above the
    threshold under quantization).

    Sequences of rank < 3 fall back to flat pooling.
    """
    front, back = split_blocks(t, max_tokens, front_frac)
    if front is None:
        return None
    out = torch.cat([front, back], dim=0) if back is not None else front
    if out.shape[0] < 2:
        return None
    return out


def split_blocks(
    t: torch.Tensor,
    max_tokens: int,
    front_frac: float = 0.5,
) -> tuple:
    """(front, back) block split with positional caps; (pooled, None) for rank<3.

    Deterministic and data-independent: the same input shapes always produce
    the same row counts, so ref and q pools stay aligned row-for-row.
    """
    if t is None:
        return None, None
    t = t.detach().to(torch.float32)
    if t.dim() < 3:
        return pool_samples(t, max_tokens), None
    t = t.reshape(-1, t.shape[-2], t.shape[-1]).cpu()  # (B, L, D)
    L = t.shape[1]
    n_front = max(1, round(L * front_frac))
    front = t[:, :n_front].reshape(-1, t.shape[-1])
    back = t[:, n_front:].reshape(-1, t.shape[-1])
    cap = max(2, max_tokens // 2)
    front = _stride_subsample(front, cap)
    back = _stride_subsample(back, cap)
    return front, back


def _center(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(dim=0, keepdim=True)


# --------------------------------------------------------------------------- #
# v1.3 feasibility guards (design doc §5.1.3) — amplitude / saturation
# --------------------------------------------------------------------------- #
def rms_ratio_median(fp: torch.Tensor, q: torch.Tensor, eps: float = 1e-6) -> float:
    """D_rms guard: median over output channels of |log(rms_q / rms_fp)|.

    Per-channel RMS is computed over the pooled tokens. A uniform scaling by c
    yields log(c) exactly. Reference semantics: pure FP16 (deployment pairing).
    """
    fp = fp.detach().to(torch.float32)
    q = q.detach().to(torch.float32)
    rms_fp = fp.square().mean(dim=0).sqrt()
    rms_q = q.square().mean(dim=0).sqrt()
    ratio = ((rms_q + eps) / (rms_fp + eps)).log().abs()
    return float(ratio.median())


def sat_rate(fp: torch.Tensor, q: torch.Tensor, act_pct: float = 99.9,
             qmax: float = 127.0, max_elem: int = 8_000_000) -> float:
    """D_sat guard: fraction of quantized outputs exceeding the next layer's
    static A8 range.

    The next layer's static activation scale is proxied from the FP16 output of
    THIS layer: s_next = P_act_pct(|fp|) / qmax (data-free, consistent with the
    static A8 calibration used in deployment). Elements of q with |x| > qmax·s
    would saturate the downstream A8 quantizer.

    Note: torch.quantile with a SCALAR q sorts the whole tensor and refuses
    inputs above 2^24 elements — large DiT layers exceed that after pooling,
    so the flattened values are deterministically stride-subsampled to
    max_elem before the percentile (the mean-based saturation rate over q is
    unaffected by size).
    """
    fp = fp.detach().to(torch.float32)
    q = q.detach().to(torch.float32)
    f = fp.flatten().abs()
    if f.numel() > max_elem:
        step = f.numel() // max_elem + 1
        f = f[::step]
    s = torch.quantile(f, act_pct / 100.0) / qmax
    if float(s) <= 0.0:
        return 0.0
    return float((q.abs() > qmax * s).to(torch.float32).mean())


def amax_ratio(fp: torch.Tensor, q: torch.Tensor, eps: float = 1e-6) -> float:
    """Diagnostic: |q|_max / |fp|_max (not a hard guard, logged alongside D_rms)."""
    return float(q.detach().to(torch.float32).abs().max() / (fp.detach().to(torch.float32).abs().max() + eps))


def _fro2(mat: torch.Tensor) -> float:
    """Squared Frobenius norm of a possibly-large matrix (float)."""
    return float(torch.linalg.matrix_norm(mat, ord="fro")) ** 2


# --------------------------------------------------------------------------- #
# CKA estimator battery (v1.4 forensic audit, D-020 §10 route 2)
# --------------------------------------------------------------------------- #
# The v1.3 score bank uses BIASED linear CKA (||Sigma_TS||_F^2 / denom). Under
# D >= N the biased HSIC has a known finite-sample inflation: even independent
# random representations can score high. The audit adds the unbiased HSIC
# estimator (Song et al. 2012) — for column-centered features the standard
# expression collapses to tr(K L)/(N(N-3)) because 1^T K 1 = 0 — plus the
# diagonal-adjusted RV coefficient (Smilde et al. 2009, "RV2"), and a control
# battery (row-shuffled / independent-random) to separate real similarity from
# dimension bias. All inputs must be row-aligned (same N), column-centered
# (N, D) float tensors.
def hsic_biased_linear(xc: torch.Tensor, yc: torch.Tensor) -> float:
    """Biased linear-kernel HSIC: ||Xc^T Yc||_F^2 / (N-1)^2."""
    n = xc.shape[0]
    assert n == yc.shape[0] and n >= 2, "row-aligned inputs required"
    return float(torch.linalg.matrix_norm(xc.T @ yc, ord="fro")) ** 2 / (n - 1) ** 2


def hsic_biased_from_gram(k: torch.Tensor, l: torch.Tensor, n: int) -> float:
    """Biased linear HSIC from precomputed Gram matrices K = Xc Xc^T, L = Yc Yc^T.

    Lets audit batteries precompute the reference Gram ONCE across shuffle
    seeds instead of recomputing the (N, D) @ (D, N) product per seed.
    """
    return float((k * l).sum()) / (n - 1) ** 2


def hsic_unbiased_from_gram(k: torch.Tensor, l: torch.Tensor, n: int) -> Optional[float]:
    """Unbiased linear HSIC from precomputed Grams (diagonal-free, Song et al.)."""
    if n <= 3:
        return None
    k0 = k - torch.diag(torch.diag(k))
    l0 = l - torch.diag(torch.diag(l))
    return float((k0 * l0).sum()) / (n * (n - 3))


def hsic_unbiased_linear(xc: torch.Tensor, yc: torch.Tensor) -> Optional[float]:
    """Unbiased linear-kernel HSIC (Song et al. 2012).

    For column-centered features the 1^T K 1 terms vanish and the U-statistic
    reduces to tr(K0 L0) / (N (N-3)) with DIAGONAL-FREE Gram matrices
    (K0_ii = L0_ii = 0); the diagonal terms ||x_i||^2 ||y_i||^2 would otherwise
    contribute a positive D-dependent bias even for independent variables.
    Requires N > 3.
    """
    n = xc.shape[0]
    assert n == yc.shape[0], "row-aligned inputs required"
    if n <= 3:
        return None
    k = xc @ xc.T
    l = yc @ yc.T
    k.fill_diagonal_(0.0)
    l.fill_diagonal_(0.0)
    return float((k * l).sum()) / (n * (n - 3))


def debiased_linear_cka(xc: torch.Tensor, yc: torch.Tensor) -> Optional[float]:
    """CKA assembled from UNBIASED HSIC estimates, clamped to [0, 1].

    hsic_u(X,Y) / sqrt(hsic_u(X,X) * hsic_u(Y,Y)); None when the denominator is
    non-positive (degenerate representation) or N <= 3.
    """
    v = debiased_linear_cka_raw(xc, yc)
    if v is None:
        return None
    return float(min(max(v, 0.0), 1.0))


def debiased_linear_cka_raw(xc: torch.Tensor, yc: torch.Tensor) -> Optional[float]:
    """UNCLAMPED unbiased-HSIC CKA (can be negative).

    Controls (shuffled/random) must be evaluated on the raw ratio: clamping to
    [0,1] folds the symmetric negative half of the null distribution into a
    positive offset (observed ~0.2 at N=64/D=1024), which would fake
    "similarity" in the D>=N regime. Audit gates use seed-averaged raw values.
    """
    xy = hsic_unbiased_linear(xc, yc)
    if xy is None:
        return None
    xx = hsic_unbiased_linear(xc, xc)
    yy = hsic_unbiased_linear(yc, yc)
    if xx is None or yy is None or xx <= 0.0 or yy <= 0.0:
        return None
    return xy / math.sqrt(xx * yy)


def rv2_adjusted(xc: torch.Tensor, yc: torch.Tensor) -> Optional[float]:
    """Diagonal-adjusted RV coefficient (RV2, Smilde et al. 2009).

    RV2 = ||S_XY - diag(S_XY)||_F^2 / (||S_XX - diag(S_XX)||_F * ||S_YY - diag(S_YY)||_F)
    with S_AB = A^T B / (N-1). Removes the self-similarity diagonal that
    inflates RV under high D/N; None on degenerate denominators.
    """
    n = xc.shape[0]
    assert n == yc.shape[0] and n >= 2, "row-aligned inputs required"
    sxy = xc.T @ yc / (n - 1)
    sxx = xc.T @ xc / (n - 1)
    syy = yc.T @ yc / (n - 1)
    num = _fro2(sxy) - float((sxy.diagonal() ** 2).sum())
    den1 = _fro2(sxx) - float((sxx.diagonal() ** 2).sum())
    den2 = _fro2(syy) - float((syy.diagonal() ** 2).sum())
    if den1 <= 0.0 or den2 <= 0.0:
        return None
    return float(min(max(num / math.sqrt(den1 * den2), 0.0), 1.0))


def cka_control_battery(
    xc: torch.Tensor,
    yc: torch.Tensor,
    seed: int = 0,
) -> dict:
    """Full estimator + control battery for one aligned (X, Y) pair.

    Returns {cka (biased, v1.3 value), debiased_cka, rv2, and the same three
    metrics under row-shuffled-Y and independent-random-Y controls}. The
    separation real >> shuffled/random is the Audit-1 pass condition.
    """
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(xc.shape[0], generator=g)
    y_perm = yc[perm]
    y_rand = torch.randn_like(yc)
    y_rand = y_rand - y_rand.mean(dim=0, keepdim=True)

    def trio(a: torch.Tensor, b: torch.Tensor) -> dict:
        return {
            "cka_biased": hsic_biased_linear(a, b)
            / math.sqrt(max(hsic_biased_linear(a, a) * hsic_biased_linear(b, b), 1e-30)),
            # RAW unbiased-HSIC CKA (controls gate on this; can be negative),
            "cka_debiased": debiased_linear_cka_raw(a, b),
            "rv2": rv2_adjusted(a, b),
        }

    out = {
        "real": trio(xc, yc),
        "shuffled": trio(xc, y_perm),
        "random": trio(xc, y_rand),
        "n": int(xc.shape[0]),
        "d": int(xc.shape[1]),
        "seed": seed,
    }
    return out


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
        self._fp_front_chunks: List[Optional[torch.Tensor]] = []
        self._fp_back_chunks: List[Optional[torch.Tensor]] = []
        self._fp_mask_chunks: List[Optional[torch.Tensor]] = []
        self._row_keep: Optional[torch.Tensor] = None  # bool over concatenated pooled rows
        self._fp_raw: Optional[torch.Tensor] = None  # (M, D), uncentered (for CS)
        self._fp_centered: Optional[torch.Tensor] = None  # (M, D), column-centered (for CKA)
        self._fp_self_norm: Optional[float] = None  # ||Sigma_TT||_F (for CKA)
        self._log_xx: Optional[float] = None  # log <kappa(x,x)> (for CS)
        self.sigma: Optional[float] = None  # Gaussian kernel bandwidth (fixed from FP)

    # ------------------------------------------------------------------ #
    # FP16 pass
    # ------------------------------------------------------------------ #
    def accumulate_ref(self, t: torch.Tensor) -> None:
        """Legacy flat-path accumulation (positional pooling, no padding mask)."""
        pooled = pool_samples(t, self.max_tokens)
        if pooled is not None:
            self._fp_chunks.append(pooled)

    def accumulate_ref_blocks(
        self,
        front: Optional[torch.Tensor],
        back: Optional[torch.Tensor],
        zero_mask: Optional[torch.Tensor],
    ) -> None:
        """Paired-blocks accumulation (review round 4): the padding zero-mask is
        computed ONCE from the REFERENCE rows and stored, so the quantized pool
        is masked with the SAME positional rows in evaluate_blocks()."""
        self._fp_front_chunks.append(front)
        self._fp_back_chunks.append(back)
        self._fp_mask_chunks.append(zero_mask)

    def finalize_ref(self) -> None:
        if self._fp_centered is not None:
            return
        if self._fp_front_chunks or self._fp_back_chunks:
            fronts = [f for f in self._fp_front_chunks if f is not None]
            backs = [b for b in self._fp_back_chunks if b is not None]
            masks = [m for m in self._fp_mask_chunks if m is not None]
            if not fronts and not backs:
                return
            front_cat = torch.cat(fronts, dim=0) if fronts else None
            back_cat = torch.cat(backs, dim=0) if backs else None
            parts = [p for p in (front_cat, back_cat) if p is not None]
            x_all = torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]
            # keep = drop the REF's zero rows (back-block padding heuristic),
            # then a deterministic stride cap; the same boolean over x_all is
            # replayed on the quantized pool -> row-for-row pairing.
            if masks and back_cat is not None and front_cat is not None:
                keep = torch.cat([torch.ones(front_cat.shape[0], dtype=torch.bool),
                                  torch.cat(masks, dim=0)])
            else:
                keep = torch.ones(x_all.shape[0], dtype=torch.bool)
            n = x_all.shape[0]
            if n > self.max_tokens:
                step = n // self.max_tokens + 1
                stride_sel = torch.zeros(n, dtype=torch.bool)
                stride_sel[torch.arange(0, n, step)[: self.max_tokens]] = True
                self._row_keep = keep & stride_sel
            else:
                self._row_keep = keep
            x = x_all[self._row_keep].contiguous()
            self._fp_front_chunks = []
            self._fp_back_chunks = []
            self._fp_mask_chunks = []
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
            return

        # legacy flat path
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
        """Score a quantized-config output tensor against the cached reference.

        The caller is responsible for ALIGNING the rows (legacy path: flat
        pooling). Returns {"cka", "cs", "cs_cross"}; values are None when the
        layer had no usable FP samples.
        """
        result: dict = {"cka": None, "cs": None, "cs_cross": None}
        if self._fp_centered is None or self.sigma is None:
            return result
        y = pool_samples(q_out, self.max_tokens)
        if y is None:
            return result
        return self._evaluate_rows(y)

    def evaluate_blocks(
        self,
        front: Optional[torch.Tensor],
        back: Optional[torch.Tensor],
    ) -> dict:
        """Score quantized (front, back) blocks against the reference.

        Review round 4 (pairing correctness): the quantized rows are masked
        with the SAME REF-shared row mask (`_row_keep`) — a near-zero padding
        row that drifted above the norm threshold under quantization is still
        dropped from BOTH sides, so H_FP and H_q remain the same token batch.
        """
        result: dict = {"cka": None, "cs": None, "cs_cross": None}
        if self._fp_centered is None or self.sigma is None or front is None:
            return result
        q_all = torch.cat([front, back], dim=0) if back is not None else front
        if self._row_keep is not None:
            if q_all.shape[0] != len(self._row_keep):
                raise RuntimeError(
                    f"[bank:{self.name}] row count mismatch: q has {q_all.shape[0]} "
                    f"rows, reference keep-mask has {len(self._row_keep)} — ref/q "
                    "pools are not aligned (check batch/shape consistency)"
                )
            q_all = q_all[self._row_keep].contiguous()
        return self._evaluate_rows(q_all)

    def _evaluate_rows(self, y: torch.Tensor) -> dict:
        result: dict = {"cka": None, "cs": None, "cs_cross": None}

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
        cs_cross = -2.0 * log_xy
        cs = self._log_xx + log_yy + cs_cross
        result["cs_cross"] = float(cs_cross)
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
    """Numerical sanity checks (run on CPU): python -m gr00t.quantization.kernel_scores

    v1.3 battery (design doc §5.1.2): the scaling-ladder asserts monotonicity of
    the CS CROSS term (not the full D_CS, whose within-set term collapses and
    can mask a scaling blow-up), plus mean-shift / rotation / outlier /
    covariance-change / permutation probes. Also checks the v1.3 feasibility
    guard functions (§5.1.3).
    """
    torch.manual_seed(0)
    x = torch.randn(256, 64)

    bank = LayerScoreBank("t", max_tokens=512)
    bank.accumulate_ref(x)
    bank.finalize_ref()
    assert bank.ready, "bank failed to finalize"

    s_same = bank.evaluate(x)
    assert s_same["cka"] is not None and abs(s_same["cka"] - 1.0) < 1e-3, f"CKA(X,X)={s_same['cka']}"
    assert s_same["cs"] is not None and s_same["cs"] < 1e-3, f"CS(X,X)={s_same['cs']}"
    assert s_same["cs_cross"] is not None

    s_scaled = bank.evaluate(x * 3.7)  # isotropic scaling must not change CKA
    assert s_scaled["cka"] is not None and abs(s_scaled["cka"] - 1.0) < 1e-2, (
        f"CKA(X,3.7X)={s_scaled['cka']} (scale invariance violated)"
    )
    assert s_scaled["cs"] is not None and s_scaled["cs"] > 1e-2, f"CS(X,3.7X)={s_scaled['cs']}"

    # v1.3 ladder: cs_cross must grow monotonically away from c=1 in BOTH
    # directions (0.25 < 0.5 < 1 < 2 < 4 < 8). Full D_CS is reported only.
    ladder = {}
    for c in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        r = bank.evaluate(x * c)
        assert r["cs_cross"] is not None, f"cross(X,{c}X) is None"
        ladder[c] = r["cs_cross"]
    for a, b in ((0.25, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0)):
        assert ladder[a] < ladder[b], f"cross ladder not monotonic: {ladder[a]:.4f} vs {ladder[b]:.4f}"

    # mean shift: cross term must respond
    s_shift = bank.evaluate(x + 2.0)
    assert s_shift["cs_cross"] is not None and s_shift["cs_cross"] > s_same["cs_cross"], (
        f"mean shift not detected: cross={s_shift['cs_cross']}"
    )

    # orthogonal rotation: CKA invariant, cross term responds. The response is
    # inherently small at large N (rotation only kills the diagonal spike), so
    # use a dedicated small bank (64 rows) and a modest margin.
    x64 = torch.randn(64, 64)
    bank64 = LayerScoreBank("rot", max_tokens=64)
    bank64.accumulate_ref(x64)
    bank64.finalize_ref()
    qq, _ = torch.linalg.qr(torch.randn(64, 64))
    s_rot_self = bank64.evaluate(x64)
    s_rot = bank64.evaluate(x64 @ qq)
    assert s_rot["cka"] is not None and abs(s_rot["cka"] - 1.0) < 1e-2, f"CKA(X,XR)={s_rot['cka']}"
    assert s_rot["cs_cross"] is not None and s_rot["cs_cross"] > s_rot_self["cs_cross"] + 1e-2, (
        f"rotation not detected by cross term: {s_rot_self['cs_cross']:.4f} vs {s_rot['cs_cross']:.4f}"
    )

    # few outliers: scores must stay finite (no inf/nan explosion)
    x_out = x.clone()
    x_out[:5] *= 100.0
    s_out = bank.evaluate(x_out)
    assert s_out["cka"] is not None and math.isfinite(s_out["cka"]), f"CKA outlier={s_out['cka']}"
    assert s_out["cs"] is not None and math.isfinite(s_out["cs"]), f"CS outlier={s_out['cs']}"
    assert s_out["cs_cross"] > s_same["cs_cross"], f"outlier injection not detected: {s_out['cs_cross']}"

    # covariance change (per-dim variance preserved, correlation changed):
    # y = x @ A with row-normalized random A -> each output dim keeps variance,
    # but the cross structure differs -> cross term must respond.
    A = torch.randn(64, 64)
    A = A / A.norm(dim=1, keepdim=True)
    s_cov_self = bank64.evaluate(x64)
    s_cov = bank64.evaluate(x64 @ A)
    assert s_cov["cs_cross"] is not None and s_cov["cs_cross"] > s_cov_self["cs_cross"] + 1e-2, (
        f"covariance change not detected: {s_cov_self['cs_cross']:.4f} vs {s_cov['cs_cross']:.4f}"
    )

    # token permutation: the Gaussian-kernel cross term is a permutation-
    # invariant mean over ALL pairs -> must stay (numerically) identical.
    perm = torch.randperm(256)
    s_perm = bank.evaluate(x[perm])
    assert s_perm["cs_cross"] is not None
    assert abs(s_perm["cs_cross"] - s_same["cs_cross"]) < 1e-3, (
        f"cross term not permutation-invariant: {s_same['cs_cross']:.4f} vs {s_perm['cs_cross']:.4f}"
    )
    # (CKA is row-correspondence sensitive: permuting one side drops it.)
    assert s_perm["cka"] is not None and s_perm["cka"] < 0.9, f"CKA(X,perm(X))={s_perm['cka']}"

    # --- stratified pooling (positional-only; review round 4) ---
    x3 = torch.randn(4, 32, 16)  # (B, L, D), vision-prefix layout
    sp = pool_samples_stratified(x3, max_tokens=64, front_frac=0.5)
    assert sp is not None and sp.shape[0] <= 64 and sp.shape[1] == 16
    x3b = torch.randn(4, 64, 16)
    sp2 = pool_samples_stratified(x3b, max_tokens=64)
    assert sp2.shape[0] <= 64
    # rank-2 fallback
    sp3 = pool_samples_stratified(x, 128)
    assert sp3 is not None
    # split_blocks is deterministic in row counts for fixed shapes
    f1, b1 = split_blocks(x3b, 64)
    f2, b2 = split_blocks(x3b * 2.0, 64)
    assert f1.shape == f2.shape and b1.shape == b2.shape

    # --- ref/q pairing regression (review round 4) ---
    # padding row 24 sits in the back block: ref sees it as (near-)zero and
    # stores its mask; the quantized pool's SAME row has drifted above the
    # threshold — evaluate_blocks must still drop it via the REF mask, so
    # CKA/CS of the two sides stays 1.0/0.0 (identical kept tokens).
    xp = torch.randn(4, 32, 16)
    xp[:, 24, :] = 0.0
    rf, rb = split_blocks(xp, 64)
    rmask = rb.norm(dim=1) > 1e-9
    bankp = LayerScoreBank("pair", max_tokens=64)
    bankp.accumulate_ref_blocks(rf, rb, rmask)
    bankp.finalize_ref()
    qp = xp.clone()
    qp[:, 24, :] = 1.0  # quantized drift: the padding row is now nonzero
    qf, qb = split_blocks(qp, 64)
    sp_ok = bankp.evaluate_blocks(qf, qb)
    assert sp_ok["cka"] is not None and abs(sp_ok["cka"] - 1.0) < 1e-2, sp_ok
    assert sp_ok["cs"] is not None and sp_ok["cs"] < 1e-3, sp_ok
    # a row-count mismatch must raise loudly (desynced pools)
    try:
        bankp.evaluate_blocks(qf[: qf.shape[0] - 1], qb)
        raise AssertionError("row-count mismatch not detected")
    except RuntimeError:
        pass

    # sat_rate on a tensor above torch.quantile's 2^24-element limit
    big = torch.randn(17_000_000)
    big_q = big * 8.0
    s_big = sat_rate(big, big_q)
    assert 0.0 <= s_big <= 1.0, s_big
    assert s_big > 0.5, f"scaled saturation should be large, got {s_big}"

    # --- v1.3 feasibility guards ---
    g_same = rms_ratio_median(x, x)
    assert abs(g_same) < 1e-4, f"rms_ratio(X,X)={g_same}"
    g8 = rms_ratio_median(x, x * 8.0)
    assert abs(g8 - math.log(8.0)) < 1e-3, f"rms_ratio(X,8X)={g8} (expect log 8)"
    sat_x = sat_rate(x, x)
    sat_8 = sat_rate(x, x * 8.0)
    assert sat_8 > sat_x, f"sat_rate not monotonic under scaling: {sat_x} vs {sat_8}"
    amax8 = amax_ratio(x, x * 8.0)
    assert abs(amax8 - 8.0) < 1e-2, f"amax_ratio(X,8X)={amax8} (expect ~8)"

    print("[kernel_scores] selftest OK (v1.3 battery)")
    print(f"  CKA(X,X)      = {s_same['cka']:.6f}")
    print(f"  CKA(X,3.7X)   = {s_scaled['cka']:.6f}   (scale-invariant, expect ~1)")
    print(f"  CKA(X,XR)     = {s_rot['cka']:.6f}   (rotation-invariant, expect ~1)")
    print(f"  CKA(X,permX)  = {s_perm['cka']:.6f}   (row-correspondence sensitive)")
    print(f"  CS(X,X)       = {s_same['cs']:.6f}")
    print(f"  CS(X,3.7X)    = {s_scaled['cs']:.6f}")
    print(f"  cross ladder  = {['%.2f' % ladder[c] for c in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)]}  (monotonic)")
    print(f"  guards        rms(X,8X)={g8:.4f} amax(X,8X)={amax8:.2f} sat {sat_x:.2e}->{sat_8:.2e}")
    selftest_cka_estimators()


def selftest_cka_estimators() -> None:
    """v1.4 forensic-audit estimator battery (D-020 route 2, Audit 1).

    Property checks for the unbiased-HSIC CKA and RV2 estimators plus the
    shuffled/random control battery; the D>=N dimension-bias phenomenon itself
    is REPORTED (printed), not asserted, because the audit measures it on real
    activations.
    """
    torch.manual_seed(0)

    # --- invariants: identity / scaling ---
    x = torch.randn(512, 96)
    xc = _center(x)
    assert debiased_linear_cka(xc, xc) is not None
    assert abs(debiased_linear_cka(xc, xc) - 1.0) < 1e-3
    assert abs(debiased_linear_cka(xc, xc * 3.7) - 1.0) < 1e-2, "scale invariance violated"
    assert abs(rv2_adjusted(xc, xc) - 1.0) < 1e-2, f"RV2(X,X)={rv2_adjusted(xc, xc)}"

    # --- noise ladder: similarity must decrease as noise grows ---
    prev = None
    for sig in (0.0, 0.05, 0.2, 1.0):
        v = debiased_linear_cka(xc, _center(x + sig * torch.randn_like(x)))
        assert v is not None
        if prev is not None:
            assert v <= prev + 1e-6, f"debiased CKA not monotone under noise: {prev} -> {v}"
        prev = v

    # --- shared-factor pair: real >> shuffled ---
    z = torch.randn(512, 16)
    a = torch.randn(96, 16) / 4.0
    b = torch.randn(96, 16) / 4.0
    x_s = _center(z @ a.T + 0.1 * torch.randn(512, 96))
    y_s = _center(z @ b.T + 0.1 * torch.randn(512, 96))
    bat = cka_control_battery(x_s, y_s, seed=0)
    assert bat["real"]["cka_debiased"] is not None and bat["real"]["cka_debiased"] > 0.5, bat["real"]
    assert bat["shuffled"]["cka_debiased"] is not None
    assert bat["real"]["cka_debiased"] > 3.0 * max(bat["shuffled"]["cka_debiased"], 1e-4), \
        f"real vs shuffled separation too small: {bat['real']} vs {bat['shuffled']}"
    assert bat["real"]["rv2"] is not None and bat["real"]["rv2"] > 0.5, bat["real"]

    # --- independent-random control + N/D sweep (the D>=N bias phenomenon) ---
    # Empirical facts baked into the gate (measured on this estimator):
    #   * BIASED CKA inflates with D/N (0.20 -> 0.50 -> 0.80 on pure noise);
    #   * the unbiased-HSIC ratio still carries a POSITIVE O(1/N) bias
    #     (+0.20 @ N=64 -> +0.002 @ N=1024, D/N in [2,16]) — it decays with N
    #     but never vanishes at finite N, so the audit must gate on the
    #     real-vs-control SEPARATION at fixed N, not on absolute thresholds;
    #   * RV2 saturates at ~1 for D>N regardless (recorded as evidence).
    print("[cka-estimators] independent-random sweep (biased CKA should inflate with D/N;")
    print("                 raw debiased CKA carries O(1/N) positive bias -> audit gates")
    print("                 on real-vs-control separation at fixed N; RV2 recorded only):")
    for n, d in ((256, 64), (256, 256), (256, 1024), (64, 1024), (32, 2048), (512, 2048), (1024, 2048)):
        raw_means, biased_means, rv2_means, shuf_means = [], [], [], []
        for s in range(5):
            xa = _center(torch.randn(n, d))
            ya = _center(torch.randn(n, d))
            b = cka_control_battery(xa, ya, seed=10 + s)
            raw_means.append(b["real"]["cka_debiased"])
            biased_means.append(b["real"]["cka_biased"])
            rv2_means.append(b["real"]["rv2"])
            shuf_means.append(b["shuffled"]["cka_debiased"])
        rm = float(sum(v for v in raw_means if v is not None)) / len(raw_means)
        bm = sum(biased_means) / len(biased_means)
        vm = sum(v for v in rv2_means if v is not None) / len(rv2_means)
        sm = float(sum(v for v in shuf_means if v is not None)) / len(shuf_means)
        if n >= 64:
            # independence consistency: real and shuffled are both independent
            # pairs under this control, so their raw debiased values must agree
            assert abs(rm - sm) < 0.05, f"real/shuffled control mismatch: {rm:.4f} vs {sm:.4f}"
            # loose sanity: nowhere near the 0.5-0.9 biased-CKA inflation.
            # (N=32/D=2048 is EVIDENCE-ONLY: the ratio bias reaches ~0.67 there,
            # which is why GR00T audit pools must keep N >= 1024, where the
            # measured bias is <= 0.02.)
            assert rm < 0.25, f"raw debiased CKA grossly inflated on independent randoms: {rm:.4f}"
        else:
            assert abs(rm - sm) < 0.05, f"real/shuffled control mismatch (evidence row): {rm:.4f} vs {sm:.4f}"
        print(f"  N={n:4d} D={d:4d} (D/N={d/n:4.1f}): "
              f"biased={bm:.4f} raw_debiased={rm:+.4f} shuffled_raw={sm:+.4f} rv2={vm:.4f}")

    print("[cka-estimators] selftest OK (debiased CKA + RV2 + controls)")


if __name__ == "__main__":
    selftest()
