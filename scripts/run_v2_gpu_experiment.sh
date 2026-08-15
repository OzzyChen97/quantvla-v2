#!/bin/bash
# QuantVLA v2 (v1.3, P0-corrected) gated GPU experiment pipeline.
#
# Review round 2, item 7: the old script (probe -> selector -> LIBERO, probing
# Long during development) violated the design-doc policy and skipped every
# correctness gate. This pipeline follows docs/quantvla_v2_design.md §6.6:
#
#   CPU selftests
#     -> gate 0 metric audit (dev suites)
#     -> dev-suite W4 probe
#     -> binary selector (+ guard filter, diverse TopK)
#     -> baseline generation + stage-2 D_solver screening
#     -> TopK adjudication (gr00t_topk_scorer)
#     -> dev-suite LIBERO acceptance (spatial/goal/object ONLY)
#     -> freeze decisions (runs/v2_decisions.md)
#     -> Long/90 final evaluation (held-out)
#
# Usage:
#   ./scripts/run_v2_gpu_experiment.sh <suite>            # one dev suite end-to-end
#   ./scripts/run_v2_gpu_experiment.sh all-dev            # spatial+goal+object
#   ./scripts/run_v2_gpu_experiment.sh final-holdout      # libero_10/90 acceptance only
#   GPU=1 ./scripts/run_v2_gpu_experiment.sh spatial      # select GPU
set -euo pipefail

REPO=/home1/gyy/vla/QuantVLA
cd "$REPO"
export PYTHONPATH=$REPO/code:$REPO/scripts/tools:$PYTHONPATH
PY=/home1/gyy/probe/miniforge3/envs/groot_test/bin/python
GPU=${GR00T_GPU:-4}
export CUDA_VISIBLE_DEVICES=$GPU

DEV_SUITES=(spatial goal object)
MODE="${1:-all-dev}"

run_selftests() {
    echo "=== [gate] CPU selftests ==="
    $PY -m gr00t.quantization.kernel_scores >/dev/null 2>&1
    $PY -m gr00t.quantization.duquant_layers >/dev/null 2>&1
    $PY scripts/tools/gr00t_select_plan.py --selftest >/dev/null 2>&1
    $PY scripts/tools/gr00t_sensitivity_probe.py --selftest >/dev/null 2>&1
    $PY scripts/tools/gr00t_metric_audit.py --selftest >/dev/null 2>&1
    $PY scripts/tools/gr00t_baselines.py --mode selftest >/dev/null 2>&1
    $PY scripts/tools/gr00t_topk_scorer.py --selftest >/dev/null 2>&1
    $PY scripts/tools/calibrate_atm_perstep_gr00t.py --selftest >/dev/null 2>&1
    echo "=== selftests: ALL PASS ==="
}

# $1 = suite; runs the full dev pipeline for one suite
run_dev_suite() {
    local S=$1
    local PK=$REPO/checkpoints/packs/gr00t/duquant_packed_libero_${S}_w4a8_b64c32ls015
    echo ""
    echo "############################################################"
    echo "# DEV PIPELINE: libero_$S"
    echo "############################################################"

    echo "--- [gate 0] metric audit ($S, 30 layers × {2,4,6,8} × 3 seeds) ---"
    $PY scripts/tools/gr00t_metric_audit.py --suite "$S" \
        --layers-subset 30 --bits 2,4,6,8 --n-seeds 3 --n-obs 8 --n-rollout-obs 4
    echo ">>> GATE 0 CHECK: review metric_audit_libero_${S}.json — bit separation,"
    echo ">>> W2-vs-W8 ratio, seed stability, guard fire on W2 — BEFORE continuing."

    echo "--- [probe] W4-only sensitivity ($S) ---"
    $PY scripts/tools/gr00t_sensitivity_probe.py --suite "$S" \
        --n-obs 16 --bits 4 --group 64 --n-rollout-obs 8 --cs-in-situ-check
    local SENS=$REPO/checkpoints/packs/gr00t/sensitivity_libero_${S}_g64_b4.json

    echo "--- [selector] binary W4/FP16 + guards + diverse TopK ($S) ---"
    $PY scripts/tools/gr00t_select_plan.py \
        --sensitivity "$SENS" \
        --ckpt $REPO/checkpoints/gr00t/libero-$([ "$S" = "10" ] && echo long || echo "$S") \
        --out $REPO/checkpoints/packs/gr00t/gr00t_quant_plan_libero_${S}.json \
        --solver greedy --binary --min-bits 4 --emit-env

    echo "--- [baselines] generate + stage-2 D_solver screening ($S) ---"
    $PY scripts/tools/gr00t_baselines.py --mode generate --suite "$S" \
        --ckpt $REPO/checkpoints/gr00t/libero-$([ "$S" = "10" ] && echo long || echo "$S") \
        --ref-plan $REPO/checkpoints/packs/gr00t/gr00t_quant_plan_libero_${S}.json \
        --n-random 20 --out-dir $REPO/checkpoints/packs/gr00t/baselines_${S}
    $PY scripts/tools/gr00t_baselines.py --mode stage2 --suite "$S" \
        --plans-dir $REPO/checkpoints/packs/gr00t/baselines_${S} \
        --out $REPO/checkpoints/packs/gr00t/baselines_${S}_dsolver.json

    echo "--- [TopK adjudication] D_solver + select_final ($S) ---"
    $PY scripts/tools/gr00t_topk_scorer.py \
        --plan $REPO/checkpoints/packs/gr00t/gr00t_quant_plan_libero_${S}.json \
        --ckpt $REPO/checkpoints/gr00t/libero-$([ "$S" = "10" ] && echo long || echo "$S") \
        --suite "$S" --packdir "$PK" \
        --out $REPO/checkpoints/packs/gr00t/gr00t_quant_plan_libero_${S}_adjudicated

    echo "--- [dev LIBERO] final plan acceptance ($S, 50-rollout smoke) ---"
    echo "    (run in two terminals; ATM/OHB off by default)"
    echo "    export GR00T_DUQUANT_PLAN=$REPO/checkpoints/packs/gr00t/gr00t_quant_plan_libero_${S}_adjudicated.final_plan.json"
    echo "    terminal1: ./scripts/run_quantvla.sh libero_$S"
    echo "    terminal2: ./scripts/run_libero_eval.sh libero_$S --headless"
}

case "$MODE" in
    all-dev)
        run_selftests
        for S in "${DEV_SUITES[@]}"; do run_dev_suite "$S"; done
        echo ""
        echo ">>> NEXT (manual, per §6.6 exit gates):"
        echo ">>>   1. v2 >= uniform W6 AND v2 >= random mask (task-level paired test)"
        echo ">>>   2. FP16-mask Jaccard >= 0.7 across 3 calibration seeds"
        echo ">>>   3. record every switch decision in runs/v2_decisions.md"
        echo ">>>   then: ./scripts/run_v2_gpu_experiment.sh final-holdout"
        ;;
    spatial|goal|object)
        run_selftests
        run_dev_suite "$MODE"
        ;;
    final-holdout)
        echo "=== held-out acceptance: libero_10 + libero_90 (final plans only) ==="
        echo "    export GR00T_DUQUANT_PLAN=<adjudicated .final_plan.json>"
        echo "    libero_10: run_quantvla.sh + run_libero_eval.sh libero_10 --headless (3 seeds x 50)"
        echo "    libero_90: run_quantvla.sh + run_libero_eval.sh libero_90 --headless (3 seeds x 50)"
        echo "    Long/90 MUST NOT be used for tuning — this is the frozen evaluation."
        ;;
    *)
        echo "usage: $0 {spatial|goal|object|all-dev|final-holdout}"; exit 1 ;;
esac
