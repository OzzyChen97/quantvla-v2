#!/usr/bin/env python3
"""GR00T CKA forensic audit (v1.4, D-020 route 2) — bounded, one pass per suite.

Five audits on the existing three GR00T checkpoints, reusing the v1.3 probe
protocol (REFERENCE = weight_bits=0 pipeline, single-layer W4 intervention,
paired noises) so every Spearman is apples-to-apples with the v1.3 d_solver
evidence:

  1. estimator battery: biased CKA vs raw unbiased-HSIC CKA vs RV2, plus
     shuffled-row / independent-random controls, over an N sweep
     (256/512/1024/2048 rows). Gate: real >> shuffled/random at fixed N.
  2. hook location: raw Linear output / parent block output (post-residual) /
     DiT final output / final-norm output (post-LN), each Spearman
     (1 - CKA, d_solver_b4) using the EXISTING probe sensitivity JSON.
  3. data source: synthetic (L1) vs FP16-policy rollout states (L2) vs
     pre-extracted real demo obs npz (L3). PTQ stays calibration-only.
  4. token range: all rows / front half / back half / last-8 tail of the
     DiT/backbone sequence (positional stratification; the DiT hidden space
     is action-token space, so action vs state conditioning lives in the
     cross-attention path — recorded as a scope note).
  5. action-conditioned subspace: per layer, Jacobian J = d a_T / d H_i at the
     final action step via batched vjp, U = TopSV(J^T J) (top-32), then
     CKA(H_ref U, H_q U). Linear-probe (ridge) fallback if the vjp path OOMs.

Usage (groot_test env, one idle GPU):
    python scripts/tools/gr00t_cka_audit.py --suite spatial \
        --sensitivity checkpoints/packs/gr00t/sensitivity_libero_spatial_g64_b4.json \
        --layers-subset 30 --audits 1,2,4 --smoke 3
"""

from __future__ import annotations

import argparse
import gc
import json
import math
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
    SUITE_DIRS,
    chunked,
    ensure_flash_attn_rpath,
    fixed_calibration_buffer,
    load_policy,
    make_obs,
    resolve_data_config,
    restore_quant_env,
    set_quant_env,
    strip_quant_env,
)
from gr00t_sensitivity_probe import (  # noqa: E402
    REF_BITS,
    discover_targets,
    run_activations,
    run_rollouts,
    set_all_bits,
    set_single_layer_bits,
    subset_names,
)
from gr00t.quantization.duquant_layers import all_calibrated, calibration_progress  # noqa: E402
from gr00t.quantization.kernel_scores import _center, cka_control_battery  # noqa: E402

PACKDIR_TEMPLATE = "duquant_packed_libero_{suite}_w4a8_b64c32ls015"
PROBE_BITS = 4


# --------------------------------------------------------------------------- #
# Collectors: raw fp/q matrices per location (front/back/tail splits)
# --------------------------------------------------------------------------- #
def _split_front_back_tail(t: torch.Tensor, max_tokens: int) -> Dict[str, Optional[torch.Tensor]]:
    """(front, back, tail-8) positional splits; deterministic row counts."""
    if t is None:
        return {"all": None, "front": None, "back": None, "tail": None}
    t = t.detach().to(torch.float32)
    if t.dim() < 3:
        flat = t.reshape(-1, t.shape[-1]).cpu()
        return {"all": flat[:max_tokens], "front": None, "back": None, "tail": None}
    t = t.reshape(-1, t.shape[-2], t.shape[-1]).cpu()
    L = t.shape[1]
    n_front = max(1, round(L * 0.5))
    cap = max(2, max_tokens // 2)
    front = t[:, :n_front].reshape(-1, t.shape[-1])
    back = t[:, n_front:].reshape(-1, t.shape[-1])
    tail = t[:, -8:].reshape(-1, t.shape[-1]) if L >= 8 else back
    return {
        "all": front[:cap * 2],
        "front": front[:cap],
        "back": back[:cap],
        "tail": tail[:cap],
    }


class _RawCollector:
    """Hook one module by name; store raw outputs per pass."""

    def __init__(self, names: List[str], max_tokens: int):
        self.names = set(names)
        self.max_tokens = max_tokens
        self.buf: Dict[str, List[torch.Tensor]] = {}
        self.handles: List[Any] = []
        self.hit: Dict[str, int] = {}

    def _fn(self, name: str):
        def hook(module, args, output):
            # some modules (backbone decoder blocks) return tuples
            if isinstance(output, tuple):
                output = output[0]
            if not isinstance(output, torch.Tensor):
                return
            self.buf.setdefault(name, []).append(output)
            self.hit[name] = self.hit.get(name, 0) + 1
        return hook

    def install(self, model: torch.nn.Module) -> None:
        self.remove()
        for name, mod in model.named_modules():
            if name in self.names:
                self.handles.append(mod.register_forward_hook(self._fn(name)))

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []

    def pooled(self, name: str) -> Dict[str, Optional[torch.Tensor]]:
        outs = self.buf.pop(name, [])
        if not outs:
            return {"all": None, "front": None, "back": None, "tail": None}
        cat = torch.cat([o for o in outs], dim=0) if len(outs) > 1 else outs[0]
        return _split_front_back_tail(cat, self.max_tokens)


# --------------------------------------------------------------------------- #
# Battery helpers
# --------------------------------------------------------------------------- #
def _deinference_duquant_tensors(model: torch.nn.Module) -> int:
    """Clone every tensor attribute of DuQuantLinear modules out of the
    inference-marked state.

    Frozen A8 scales / rotation caches / pre-quantized weights are computed
    inside torch.no_grad() during calibration, which marks them as inference
    tensors; using them in a GRAD-ENABLED forward (Audit 5's vjp) raises
    "Inference tensors cannot be saved for backward" at the first division.
    clone() outside inference mode produces normal tensors (the documented
    workaround). Returns the number of cloned tensors.
    """
    from gr00t.quantization.duquant_layers import DuQuantLinear

    n = 0
    for m in model.modules():
        if not isinstance(m, DuQuantLinear):
            continue
        for name in list(m.__dict__):
            v = getattr(m, name)
            if isinstance(v, torch.Tensor):
                setattr(m, name, v.clone())
                n += 1
            elif isinstance(v, dict):
                for k in list(v):
                    if isinstance(v[k], torch.Tensor):
                        v[k] = v[k].clone()
                        n += 1
    return n


def battery(ref: torch.Tensor, q: torch.Tensor, seeds: int = 5, n_sweep: bool = False) -> Dict[str, Any]:
    """Control battery for one aligned (ref, q) pair; seed-averaged raw values.

    Fast path: the reference Gram K = Xc Xc^T is computed ONCE per (level,
    pair); all shuffle/random control Grams reuse it via the Gram-based HSIC
    helpers (hsic_biased_from_gram / hsic_unbiased_from_gram). RV2 is computed
    for the real pair and the first control permutation only (D x D
    covariance products dominate at D=6144).
    """
    from gr00t.quantization.kernel_scores import (  # noqa: PLC0415
        hsic_biased_from_gram,
        hsic_unbiased_from_gram,
        rv2_adjusted,
    )

    mask = ref.norm(dim=1) > 1e-9
    if int(mask.sum()) < 16:
        return {"n": 0, "note": "too few nonzero rows"}
    ref = ref[mask].contiguous()
    q = q[mask].contiguous()
    n = ref.shape[0]

    def level(rows_ref: torch.Tensor, rows_q: torch.Tensor) -> Dict[str, float]:
        m = rows_ref.shape[0]
        xc = _center(rows_ref)
        yc = _center(rows_q)
        k = xc @ xc.T
        kk_b = hsic_biased_from_gram(k, k, m)
        kk_u = hsic_unbiased_from_gram(k, k, m)
        g = torch.Generator().manual_seed(0)
        perms = [torch.randperm(m, generator=g) for _ in range(seeds)]
        groups: Dict[str, List[torch.Tensor]] = {
            "real": [yc],
            "shuffled": [yc[p] for p in perms],
            "random": [rr - rr.mean(dim=0, keepdim=True)
                       for rr in (torch.randn_like(yc) for _ in range(seeds))],
        }
        acc: Dict[str, float] = {}
        cnt: Dict[str, int] = {}
        for group, mats in groups.items():
            for j, ym in enumerate(mats):
                ll = ym @ ym.T
                xy_b = hsic_biased_from_gram(k, ll, m)
                ll_b = hsic_biased_from_gram(ll, ll, m)
                biased = xy_b / math.sqrt(max(kk_b * ll_b, 1e-30))
                xy_u = hsic_unbiased_from_gram(k, ll, m)
                ll_u = hsic_unbiased_from_gram(ll, ll, m)
                deb = None
                if xy_u is not None and ll_u is not None and kk_u is not None \
                        and kk_u > 0 and ll_u > 0:
                    deb = xy_u / math.sqrt(kk_u * ll_u)
                rv = None
                if j == 0:  # RV2 only for the real pair + first control perm
                    rv = rv2_adjusted(xc, ym)
                for name, v in (("cka_biased", biased), ("cka_debiased", deb), ("rv2", rv)):
                    if v is None:
                        continue
                    key = name if group == "real" else f"{group}_{name}"
                    # accumulate raw sums; normalize by the ACTUAL count per
                    # key (real has 1 accumulation, controls have `seeds`).
                    # Dividing every accumulation by seeds here would scale
                    # the real pair by 1/seeds (the 0.1999 = 0.9993/5 bug).
                    acc[key] = acc.get(key, 0.0) + v
                    cnt[key] = cnt.get(key, 0) + 1
        return {k: acc[k] / cnt[k] for k in acc}

    out = {"n": int(n), "d": int(ref.shape[1]), "real": level(ref, q)}
    if n_sweep:
        out["sweep"] = {}
        for cap in (256, 512, 1024, 2048):
            if n <= cap:
                continue
            step = n // cap + 1
            sub_r = ref[::step][:cap]
            sub_q = q[::step][:cap]
            out["sweep"][str(cap)] = level(sub_r, sub_q)
    return out


def _spearman(a: List[float], b: List[float]) -> Optional[float]:
    from scipy.stats import spearmanr

    if len(a) < 3 or len(a) != len(b):
        return None
    r = spearmanr(a, b)
    return float(r.statistic)


# --------------------------------------------------------------------------- #
# Model location discovery
# --------------------------------------------------------------------------- #
def parent_block_name(name: str) -> str:
    for suf in (".q_proj", ".k_proj", ".v_proj", ".o_proj", ".gate_proj", ".up_proj", ".down_proj"):
        if name.endswith(suf):
            return name[: -len(suf)]
    if ".ff.net" in name:
        return name.rsplit(".ff.net", 1)[0]
    return name.rsplit(".", 1)[0]


def discover_locations(model: torch.nn.Module, target_names: List[str]) -> Dict[str, str]:
    """location key -> module name (or None if not found)."""
    named = dict(model.named_modules())
    locs: Dict[str, Optional[str]] = {}
    parents = sorted({parent_block_name(n) for n in target_names})
    for p in parents:
        if p in named:
            locs[f"block:{p}"] = p
    dit = None
    for cand in ("action_head.model", "model.action_head"):
        if cand in named:
            dit = cand
            break
    locs["dit_output"] = dit
    norm = None
    if dit is not None:
        for cand in (f"{dit}.final_norm", f"{dit}.norm_final", f"{dit}.final_layer_norm"):
            if cand in named:
                norm = cand
                break
        if norm is None:  # last norm-ish module under the DiT
            for n, m in named.items():
                if n.startswith(dit + ".") and "norm" in n.lower() and not list(m.children()):
                    norm = n  # keep the last one found
    locs["dit_final_norm"] = norm
    return locs


# --------------------------------------------------------------------------- #
# Audit driver
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="spatial", choices=["spatial", "goal", "object", "90", "10"])
    ap.add_argument("--sensitivity", required=True, help="existing probe sensitivity JSON (d_solver source)")
    ap.add_argument("--layers-subset", type=int, default=30)
    ap.add_argument("--audits", default="1,2,3,4,5")
    ap.add_argument("--smoke", type=int, default=0, help="limit to first N subset layers (smoke mode)")
    ap.add_argument("--obs-source", default="synthetic", choices=["synthetic", "l2", "l3"],
                    help="Audit 3 data source: synthetic (L1) / l2 / l3 (both require --obs-npz)")
    ap.add_argument("--obs-npz", default=None, help="npz from collect_libero_states.py for --obs-source l2/l3")
    ap.add_argument("--n-obs", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--denoising-steps", type=int, default=8)
    ap.add_argument("--jacobian-n-obs", type=int, default=4, help="obs for Audit 5 vjp")
    ap.add_argument("--jacobian-topk", type=int, default=32)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    audits = [int(x) for x in args.audits.split(",") if x.strip()]

    args.data_config = resolve_data_config(args.suite, None)
    suite_dir = SUITE_DIRS[args.suite]
    model_path = str(REPO_ROOT / "checkpoints" / "gr00t" / suite_dir)
    packdir = str(REPO_ROOT / "checkpoints" / "packs" / "gr00t" / PACKDIR_TEMPLATE.format(suite=args.suite))
    if args.out is None:
        args.out = str(REPO_ROOT / "checkpoints" / "packs" / "gr00t" / f"cka_audit_{args.suite}.json")

    sens = json.loads(Path(args.sensitivity).read_text())
    d_solver = {}
    for n, v in sens.get("layers", {}).items():
        if f"d_solver_b{PROBE_BITS}" in v:
            d_solver[n] = v[f"d_solver_b{PROBE_BITS}"]

    ensure_flash_attn_rpath()
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    saved_env = strip_quant_env()
    # discovery must run on the PURE FP16 model (same as the probe): the
    # DuQuant-wrapped model is invisible to select_targets
    policy_fp = load_policy(model_path, data_config=args.data_config, denoising_steps=args.denoising_steps, device=args.device)
    target_names = discover_targets(policy_fp.model, argparse.Namespace(include=DEFAULT_INCLUDE, exclude=DEFAULT_EXCLUDE))
    attr_names = subset_names(target_names, args.layers_subset)
    if args.smoke:
        attr_names = attr_names[: args.smoke]
    locs = discover_locations(policy_fp.model, attr_names)
    print(f"[cka-audit] targets={len(target_names)} subset={len(attr_names)}")
    print(f"[cka-audit] locations: { {k: (v or 'N/A') for k, v in locs.items()} }")
    if args.dry_run:
        for n in attr_names:
            print(f"  - {n} (d_solver_b4={d_solver.get(n)})")
        return
    del policy_fp
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    set_quant_env(DEFAULT_INCLUDE, DEFAULT_EXCLUDE, packdir, bits_default=REF_BITS,
                  group=64, ls=0.15, act_pct=99.9, calib_steps=32, row_rot="restore",
                  act_dynamic=False)
    policy = load_policy(model_path, data_config=args.data_config, denoising_steps=args.denoising_steps, device=args.device)
    model = policy.model

    horizon = int(model.action_head.config.action_horizon)
    action_dim = int(model.action_head.config.action_dim)

    # ---- A8 calibration in the all-zero reference state (same as probe) ----
    set_all_bits(model, REF_BITS)
    n_warm_batches = 32
    n_warm_obs = n_warm_batches * args.batch_size
    warm_obs, warm_noises, warm_sha = fixed_calibration_buffer(0, n_warm_obs, horizon, action_dim, fmt="libero")

    t0 = time.time()
    run_rollouts(model, policy, warm_obs, warm_noises, args.batch_size, return_trajectory=False)
    if not all_calibrated(model):
        raise SystemExit("[cka-audit] A8 calibration did not complete "
                         f"({calibration_progress(model)})")
    print(f"[cka-audit] A8 calibrated in {time.time() - t0:.1f}s (sha {warm_sha[:16]})")

    # ---- obs source (Audit 3) ----
    # synthetic (L1): fixed random obs, data-free. l2/l3: real states bridged
    # through an npz produced by scripts/tools/collect_libero_states.py
    # (l2 = FP16-policy env rollout states; l3 = unlabeled train/demo obs) —
    # the audit process itself never imports the LIBERO env.
    obs_list: List[Dict[str, Any]] = [make_obs(rng, "libero") for _ in range(args.n_obs)]
    if args.obs_source in ("l2", "l3"):
        if not args.obs_npz:
            raise SystemExit("--obs-source l2/l3 requires --obs-npz (see collect_libero_states.py)")
        z = np.load(args.obs_npz, allow_pickle=False)
        keys = list(z.keys())
        obs_list = [{k: z[k][i] for k in keys} for i in range(min(args.n_obs, len(z[keys[0]])))]
        print(f"[cka-audit] Audit-3 {args.obs_source.upper()}: {len(obs_list)} obs from {args.obs_npz}")

    noises = [torch.randn(horizon, action_dim) for _ in obs_list]

    results: Dict[str, Any] = {
        "meta": {
            "suite": args.suite, "model_path": model_path,
            "sensitivity_source": args.sensitivity,
            "n_targets": len(target_names), "n_subset": len(attr_names),
            "layers": attr_names, "locations": locs,
            "obs_source": args.obs_source,
            "audits": audits, "smoke": args.smoke,
            "calibration_buffer_sha256": warm_sha[:16],
            "protocol": "REF=weight_bits=0 pipeline; single-layer W4 intervention; paired noises",
            "scope_note": "DiT hidden space is action-token space; front/back/tail are "
                          "positional strata. O(1/N) ratio bias measured in kernel_scores selftest: "
                          "gates use real-vs-control separation at fixed N (N>=1024 preferred).",
        },
        "audit": {},
    }

    # ---- per-layer pass with batteries at all locations (Audits 1/2/4) ----
    if any(a in audits for a in (1, 2, 4)):
        lin_names = attr_names
        other_names = [locs[k] for k in ("dit_output", "dit_final_norm") if locs.get(k)]
        block_names = [locs[k] for k in locs if k.startswith("block:")]
        set_all_bits(model, REF_BITS)
        ref_col = _RawCollector(lin_names + block_names + other_names, args.max_tokens)
        ref_col.install(model)
        run_activations(model, policy, obs_list, noises, args.batch_size)
        fp_pools = {n: ref_col.pooled(n) for n in lin_names + block_names + other_names}
        ref_col.remove()
        print(f"[cka-audit] REF pass done; pools for {len(fp_pools)} modules")

        loc_of = {n: "linear" for n in lin_names}
        for n in block_names:
            loc_of[n] = "block"
        for k in ("dit_output", "dit_final_norm"):
            if locs.get(k):
                loc_of[locs[k]] = k

        for li, name in enumerate(attr_names):
            set_all_bits(model, REF_BITS)
            if not set_single_layer_bits(model, name, PROBE_BITS):
                results["audit"].setdefault("skip", []).append(name)
                continue
            q_col = _RawCollector([name] + block_names + other_names, args.max_tokens)
            q_col.install(model)
            run_activations(model, policy, obs_list, noises, args.batch_size)
            set_single_layer_bits(model, name, REF_BITS)
            q_pools = {n: q_col.pooled(n) for n in [name] + block_names + other_names}
            q_col.remove()
            # the block containing THIS layer gets the intervention; every other
            # block module is untouched (its fp/q pools are identical up to
            # upstream drift => batteries there serve as sensitivity control)
            for n in fp_pools:
                if n not in q_pools:
                    continue
                for part in ("all", "front", "back", "tail"):
                    fp_t = fp_pools[n].get(part)
                    q_t = q_pools[n].get(part)
                    if fp_t is None or q_t is None or fp_t.shape != q_t.shape:
                        continue
                    key = f"{name}|{loc_of.get(n, n)}|{part}"
                    b = battery(fp_t, q_t, seeds=5, n_sweep=(part == "all" and 1 in audits))
                    b["d_solver_b4"] = d_solver.get(name)
                    results["audit"].setdefault("batteries", {})[key] = b
            del q_pools
            gc.collect()
            if li % 5 == 4:
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                with open(args.out, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)
                print(f"[cka-audit] layer {li + 1}/{len(attr_names)} done (incremental save)")
        print("[cka-audit] battery passes done")

    # ---- Spearman per location (Audit 2) ----
    if 2 in audits:
        from collections import defaultdict

        per_loc: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"cka": [], "d": []})
        for key, b in results["audit"].get("batteries", {}).items():
            layer, loc, part = key.split("|")
            if part != "all" or b.get("d_solver_b4") is None:
                continue
            one_minus = 1.0 - b["real"]["cka_biased"]
            per_loc[loc]["cka"].append(one_minus)
            per_loc[loc]["d"].append(b["d_solver_b4"])
        results["audit"]["2_spearman_by_location"] = {
            loc: {"n": len(v["cka"]), "spearman_1minusCKA_vs_d": _spearman(v["cka"], v["d"]),
                  "spearman_negCKA_vs_d": _spearman([-x for x in v["cka"]], v["d"])}
            for loc, v in sorted(per_loc.items())
        }
        print("[cka-audit] Audit-2 Spearman by location:", json.dumps(results["audit"]["2_spearman_by_location"], indent=2))

    # ---- Audit 5: action-conditioned subspace (vjp Jacobian, top-SV projection) ----
    # Per obs: J_i = d a_T / d H_i at the last action step and last sequence
    # position, via batched vjp (is_grads_batched). U = TopSV of J^T J
    # (hidden-side Gram, accumulated per obs to bound memory).
    # STANDALONE-CAPABLE: if the Audit-1/2/4 battery pass did not run in this
    # process, collect the REF pool and per-layer W4 pools here (same obs).
    if 5 in audits:
        jac_obs = obs_list[: args.jacobian_n_obs]
        jac_noises = noises[: args.jacobian_n_obs]
        results["audit"]["5_action_conditioned"] = {}
        set_all_bits(model, REF_BITS)
        # the action head's get_action is @torch.no_grad()-wrapped; call the
        # UNWRAPPED function so the denoising graph is available for the vjp
        # (the returned trajectory entries are detached anyway — Audit 5 uses
        # the non-detached action_pred instead)
        ah = model.action_head
        wrapped = getattr(ah.get_action, "__wrapped__", None)
        if wrapped is None:
            raise SystemExit("[cka-audit] Audit-5: cannot unwrap no_grad get_action")
        import types

        ah.get_action = types.MethodType(wrapped, ah)
        # calibration-time caches (A8 scales/rotations) are inference tensors;
        # the grad-enabled Jacobian pass cannot save them for backward
        n_deinf = _deinference_duquant_tensors(model)
        print(f"[cka-audit] Audit-5: de-inferenced {n_deinf} DuQuant tensor attrs")
        fp_pools5: Dict[str, Dict[str, Optional[torch.Tensor]]] = {}
        if "fp_pools" in dir() and "q_pools" in dir():
            fp_pools5 = fp_pools
        else:
            ref_col5 = _RawCollector(attr_names, args.max_tokens)
            ref_col5.install(model)
            run_activations(model, policy, obs_list, noises, args.batch_size)
            fp_pools5 = {n: ref_col5.pooled(n) for n in attr_names}
            ref_col5.remove()
        for li, name in enumerate(attr_names):
            try:
                store: Dict[str, Optional[torch.Tensor]] = {"h": None}

                def hook_fn(module, args, output):
                    if isinstance(output, tuple):
                        output = output[0]
                    store["h"] = output  # (B, L, D)

                mod = dict(model.named_modules())[name]
                handle = mod.register_forward_hook(hook_fn)
                jtj_sum = None
                n_obs_used = 0
                for batched_obs, batched_noise in chunked(jac_obs, jac_noises, args.batch_size):
                    norm = policy.apply_transforms(batched_obs)
                    use_autocast = str(policy.device).startswith("cuda")
                    if use_autocast:
                        from gr00t.model.policy import COMPUTE_DTYPE

                        with torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
                            out = model.get_action(norm, action_noise=batched_noise, return_trajectory=False)
                    else:
                        out = model.get_action(norm, action_noise=batched_noise, return_trajectory=False)
                    a_final = out["action_pred"]  # (B, H, D) — grad-enabled
                    h = store["h"]
                    if h is None:
                        continue
                    # backbone sequences carry trailing PADDING tokens that the
                    # action head slices away (no graph path to the action) —
                    # use position 0 there; DiT sequences end on the final
                    # action token, position -1
                    pos = 0 if name.startswith("backbone.") else -1
                    d_h = h.shape[-1]
                    if jtj_sum is None:
                        jtj_sum = torch.zeros((d_h, d_h), device=h.device, dtype=torch.float32)
                    for i in range(h.shape[0]):
                        h_i = h[i, pos]  # (D_h,) chosen sequence position
                        a_i = a_final[i, -1]  # (D_a,) last action step
                        d_a = a_i.numel()
                        eye = torch.eye(d_a, device=h.device, dtype=h.dtype)
                        (jt,) = torch.autograd.grad(a_i, h_i, grad_outputs=eye,
                                                    is_grads_batched=True, retain_graph=False)
                        jtj_sum += (jt.T @ jt).to(torch.float32)  # (D_h, D_h)
                        n_obs_used += 1
                    store["h"] = None
                    del out, a_final
                handle.remove()
                evals, evecs = torch.linalg.eigh(jtj_sum)
                u = evecs[:, -args.jacobian_topk:]  # top-SV subspace (D_h, K)
                # q pool on the SAME obs as the ref pool (W4 single-layer pass)
                if "q_pools" in dir() and name in q_pools:
                    q_all = q_pools[name]["all"]
                else:
                    set_single_layer_bits(model, name, PROBE_BITS)
                    q5 = _RawCollector([name], args.max_tokens)
                    q5.install(model)
                    run_activations(model, policy, obs_list, noises, args.batch_size)
                    q_all = q5.pooled(name)["all"]
                    q5.remove()
                    set_single_layer_bits(model, name, REF_BITS)
                    # de-inference ONLY the rebuilt module (cheap), not the
                    # whole model — the next layer's Jacobian runs grad-enabled
                    _deinference_duquant_tensors(dict(model.named_modules())[name])
                fp_all = fp_pools5[name]["all"]
                if fp_all is None or q_all is None:
                    results["audit"]["5_action_conditioned"][name] = {"status": "no pools"}
                    continue
                proj_ref = _center(fp_all) @ u.cpu()
                proj_q = _center(q_all) @ u.cpu()
                bat = cka_control_battery(proj_ref, proj_q, seed=0)
                results["audit"]["5_action_conditioned"][name] = {
                    "n_obs": n_obs_used, "topk": args.jacobian_topk,
                    "battery": {k: v for k, v in bat.items() if k != "n"},
                    "n_rows": int(proj_ref.shape[0]),
                    "d_solver_b4": d_solver.get(name),
                }
                del jtj_sum, evals, evecs, u, proj_ref, proj_q
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"[cka-audit] Audit-5 {name}: real_biased="
                      f"{results['audit']['5_action_conditioned'][name]['battery']['real']['cka_biased']:.4f}")
            except Exception as e:  # noqa: BLE001 — audit must degrade gracefully
                results["audit"]["5_action_conditioned"][name] = {"status": "failed", "error": str(e)}
                print(f"[cka-audit] Audit-5 {name} FAILED: {e}")
                gc.collect()
            if li % 5 == 4:
                with open(args.out, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    restore_quant_env(saved_env)
    print(f"[cka-audit] saved -> {args.out}")
    print("[cka-audit] done.")


if __name__ == "__main__":
    main()
