# QuantVLA v2 — Action-weighted W4/FP16 Binary Layer Selection for GR00T N1.5

This repository contains the **QuantVLA v2** design (v1.3 converged revision), the GR00T-side
implementation of the training-free per-layer mixed-precision quantization pipeline, and the
first-round LIBERO experiment artifacts.

**Core idea (v1.3):** upgrade QuantVLA from "uniform W4A8 + static ATM/OHB scale correction" to

> **Action-weighted binary W4/FP16 layer selection + amplitude/saturation feasibility guards,
> with plan-specific static ATM/OHB applied only at deployment time.**

The method measures per-layer cost of quantizing to W4 with single-layer interventions
(CKA geometry + Cauchy-Schwarz distribution divergence + RMS/saturation hard guards + action
importance weight `w_i` from paired-noise solver-trajectory divergence), then solves a
0-1 knapsack under a static weight-storage byte budget to decide which layers become W4 and
which stay FP16. A Top-K config-level `D_solver` adjudication closes the loop before LIBERO.

## Repository layout

```
docs/
  quantvla_v2_design.md                  # v1.3 design document (the authoritative spec)
  quantvla_v2_gr00t_experiment_report.md # first-round LIBERO experiment report
  quantvla_libero_results.md             # v1 (uniform W4A8) baseline results
code/gr00t/                              # GR00T N1.5 stack (model, quantization, ATM, eval)
  quantization/duquant_layers.py         # DuQuant W4A8 layers + per-layer plan support
  quantization/kernel_scores.py          # CKA + CS divergence score bank (selftests)
  atm/dit_atm.py                         # ATM/OHB, per-step table support
scripts/
  tools/gr00t_sensitivity_probe.py       # P0-G: measurement (CKA/CS + v1.3 guards, w_i, D_solver)
  tools/gr00t_select_plan.py             # P1-G: binary W4/FP16 + guards + milp + diverse TopK
  tools/gr00t_metric_audit.py            # v1.3 gate 0: metric validity audit (§6.6.1)
  tools/gr00t_baselines.py               # v1.3 baselines: random/size/manual/uniform + 2-stage D_solver
  tools/gr00t_topk_scorer.py             # v1.3 TopK adjudicator: D_solver + select_final (8-step pipeline)
  tools/calibrate_atm_perstep_gr00t.py   # P2-G: ATM/OHB calibration (+ plan-aware FP16-block skip, CV stats)
  tools/calc_gr00t_duquant_memory.py     # static weight-storage byte calculator
  tools/gr00t_v2_common.py               # shared utilities (obs/env/policy/noise)
  tools/gr1_env_smoke.py                 # GR1 tabletop env smoke test helper
  run_quantvla.sh / run_libero_eval.sh / run_inference_server.sh / run_v2_gpu_experiment.sh
  inference_service.py / simulation_service.py   # ZMQ server/client eval harness
checkpoints/packs/gr00t/
  deprecated_v1.2/                       # INVALIDATED artifacts (pre-P0-fix probes/plans/ATM)
                                         # — see docs/quantvla_v2_p0_review_response.md
  (fresh sensitivity/plan/ATM artifacts must be regenerated with the P0-fixed code)
```

## Quickstart (GR00T + LIBERO)

```bash
cd /path/to/quantvla-v2
export PYTHONPATH=$PWD/code:$PYTHONPATH

# 0) metric validity audit (design doc §6.6.1, gate 0 — run before anything else)
python scripts/tools/gr00t_metric_audit.py --suite spatial --layers-subset 30 --bits 2,4,6,8

# 1) measure (main probe only scans W4)
python scripts/tools/gr00t_sensitivity_probe.py --suite spatial \
    --n-obs 16 --bits 4 --group 64 --n-rollout-obs 8

# 2) select (binary W4/FP16 + guards + diversity TopK)
python scripts/tools/gr00t_select_plan.py \
    --sensitivity checkpoints/packs/gr00t/sensitivity_libero_spatial_g64_b4.json \
    --ckpt <path-to-gr00t-libero-spatial-ckpt> \
    --out checkpoints/packs/gr00t/gr00t_quant_plan_libero_spatial.json \
    --solver greedy --binary --min-bits 4 --emit-env

# 3) serve + evaluate
export GR00T_DUQUANT_PLAN=checkpoints/packs/gr00t/gr00t_quant_plan_libero_spatial.json
./scripts/run_quantvla.sh libero_spatial            # terminal 1
./scripts/run_libero_eval.sh libero_spatial --headless  # terminal 2
```

## Notes

- **Model checkpoints are NOT included** (multi-GB). Download GR00T N1.5 fine-tunes from the
  official NVIDIA releases and precompute DuQuant packs with `duquant_preprocess.py` before
  running the probe/selector.
- The packed-weight directories (`duquant_packed_*`) are also not included; they are regenerated
  per (group, layout) combination.
- The method is data-free: all calibration uses synthetic observations and paired noise.
  See the design document §4.3 / §5.1.5.
- Papers referenced by the design doc (CKA, CS-Aligner, Q-DiT, QuantVLA) are not redistributed
  here; citations are in `docs/quantvla_v2_design.md` §8.

## License

Apache License 2.0 (see `LICENSE`).
