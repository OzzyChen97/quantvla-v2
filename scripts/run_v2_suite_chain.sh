#!/bin/bash
# Per-suite v1.3 chain: gate-0 -> probe -> selector -> TopK -> baselines.
# Honors every gate (set -euo pipefail: a gate failure aborts the chain).
#
# Usage: GR00T_GPU=7 ./scripts/run_v2_suite_chain.sh <suite>
set -euo pipefail

S="${1:?usage: run_v2_suite_chain.sh <suite>}"
REPO=/home1/gyy/vla/QuantVLA
cd "$REPO"
export PYTHONPATH="$REPO/code:$REPO/scripts/tools:${PYTHONPATH:-}"
PY=/home1/gyy/probe/miniforge3/envs/groot_test/bin/python
export CUDA_VISIBLE_DEVICES=${GR00T_GPU:-7}
export GR00T_ATM_ENABLE=0 GR00T_OHB_ENABLE=0
LOG="$REPO/runs/v2_gpu_logs/suite_${S}.log"
mkdir -p "$REPO/runs/v2_gpu_logs"
CKPT="$REPO/checkpoints/gr00t/libero-$S"
PACK="$REPO/checkpoints/packs/gr00t/duquant_packed_libero_${S}_w4a8_b64c32ls015"
AUDIT="$REPO/checkpoints/packs/gr00t/metric_audit_libero_${S}.json"

exec > >(tee "$LOG") 2>&1

echo "===== suite chain: $S (GPU ${GR00T_GPU:-7}) ====="

if [[ -f "$AUDIT" ]]; then
    echo "[$S] gate0 audit JSON exists — reusing; re-checking the gate"
else
    echo "[$S] gate0 audit (30 layers / 3 seeds / 16 obs)"
    $PY scripts/tools/gr00t_metric_audit.py --suite "$S" \
        --layers-subset 30 --bits 2,4,6,8 --n-seeds 3 --n-obs 16 --n-rollout-obs 8
fi
echo "[$S] gate0 check"
$PY scripts/tools/gr00t_gate0_check.py --audit "$AUDIT"   # hard gate: exit 1 aborts

echo "[$S] probe (W4-only)"
$PY scripts/tools/gr00t_sensitivity_probe.py --suite "$S" \
    --n-obs 16 --bits 4 --group 64 --n-rollout-obs 8 --cs-in-situ-check

echo "[$S] selector (binary + guards + diverse TopK)"
$PY scripts/tools/gr00t_select_plan.py \
    --sensitivity "$REPO/checkpoints/packs/gr00t/sensitivity_libero_${S}_g64_b4.json" \
    --ckpt "$CKPT" \
    --out "$REPO/checkpoints/packs/gr00t/gr00t_quant_plan_libero_${S}.json" \
    --solver greedy --binary --min-bits 4

echo "[$S] TopK adjudication"
$PY scripts/tools/gr00t_topk_scorer.py \
    --plan "$REPO/checkpoints/packs/gr00t/gr00t_quant_plan_libero_${S}.json" \
    --ckpt "$CKPT" --suite "$S" --packdir "$PACK" \
    --out "$REPO/checkpoints/packs/gr00t/gr00t_quant_plan_libero_${S}_adjudicated"

echo "[$S] baselines generate + stage2"
$PY scripts/tools/gr00t_baselines.py --mode generate --suite "$S" \
    --ckpt "$CKPT" \
    --ref-plan "$REPO/checkpoints/packs/gr00t/gr00t_quant_plan_libero_${S}.json" \
    --n-random 20 --out-dir "$REPO/checkpoints/packs/gr00t/baselines_${S}"
$PY scripts/tools/gr00t_baselines.py --mode stage2 --suite "$S" \
    --plans-dir "$REPO/checkpoints/packs/gr00t/baselines_${S}" \
    --out "$REPO/checkpoints/packs/gr00t/baselines_${S}_dsolver.json"

echo "SUITE-$S-CHAIN-DONE"
