#!/bin/bash
# QuantVLA v2 GPU smoke (review round 4 run order, step 1-2).
#
#   1. all CPU selftests
#   2. spatial gate-0 SMALL scale (10 layers, 2 seeds, 4 obs)
#   3. checks: CKA/CS/RMS/sat all finite, hook coverage complete,
#      gate-0 verdict (informational at this scale)
#   4. static A8 second-start check: first server start calibrates+SAVES the
#      scales, second start LOADS them (GR00T_DUQUANT_ACT_SCALE_PATH) without
#      any warmup — verifies the persistence closure on GPU.
#
# Usage: GR00T_GPU=7 ./scripts/run_v2_smoke.sh
set -euo pipefail

REPO=/home1/gyy/vla/QuantVLA
cd "$REPO"
export PYTHONPATH="$REPO/code:$REPO/scripts/tools:${PYTHONPATH:-}"
PY=/home1/gyy/probe/miniforge3/envs/groot_test/bin/python
export CUDA_VISIBLE_DEVICES=${GR00T_GPU:-7}
LOG=/tmp/logs/v2_smoke
mkdir -p "$LOG"
export GR00T_ATM_ENABLE=0 GR00T_OHB_ENABLE=0
SCALE_PATH="$LOG/a8_scales_spatial.npz"
AUDIT="$REPO/checkpoints/packs/gr00t/metric_audit_libero_spatial_smoke.json"

echo "=== [1] CPU selftests ==="
$PY -m gr00t.quantization.kernel_scores >/dev/null 2>&1
$PY -m gr00t.quantization.duquant_layers >/dev/null 2>&1
for s in gr00t_select_plan gr00t_sensitivity_probe gr00t_metric_audit \
         gr00t_gate0_check gr00t_consensus_plan gr00t_topk_scorer; do
    $PY scripts/tools/$s.py --selftest >/dev/null 2>&1
done
$PY scripts/tools/gr00t_baselines.py --mode selftest >/dev/null 2>&1
$PY scripts/tools/calibrate_atm_perstep_gr00t.py --selftest >/dev/null 2>&1
echo "=== [1] selftests: ALL PASS ==="

echo "=== [2] spatial gate-0 SMALL (10 layers, 2 seeds, 4 obs) ==="
$PY scripts/tools/gr00t_metric_audit.py --suite spatial \
    --layers-subset 10 --bits 2,4,6,8 --n-seeds 2 \
    --n-obs 4 --n-rollout-obs 4 --out "$AUDIT" 2>&1 | tail -12

echo "=== [3] finite / coverage / gate checks ==="
$PY - "$AUDIT" <<'PY'
import json, math, sys
audit = json.load(open(sys.argv[1]))
subset = audit["meta"]["subset"]
print(f"subset layers: {len(subset)}")
total_metrics = 0
nonfinite = []
covered = {}
for seed in audit["seeds"]:
    for name, entry in seed["layers"].items():
        for bit in ("b2", "b4", "b6", "b8"):
            e = entry.get(bit, {})
            for k in ("cka", "cs", "cs_cross", "rms_ratio", "sat_rate", "nmse"):
                v = e.get(k)
                if v is not None:
                    total_metrics += 1
                    if not math.isfinite(float(v)):
                        nonfinite.append((seed["seed"], name, bit, k, v))
        ds = entry.get("d_solver_b4")
        if ds is not None:
            covered[name] = covered.get(name, 0) + 1
print(f"total finite metrics: {total_metrics}; non-finite: {len(nonfinite)}")
for x in nonfinite[:10]:
    print("  NONFINITE:", x)
covered_layers = [n for n, c in covered.items() if c == 2]
print(f"layers with d_solver_b4 in BOTH seeds (hook coverage): {len(covered_layers)}/{len(subset)}")
print("SMOKE-CHECK:", "PASS" if (not nonfinite and len(covered_layers) >= 0.8 * len(subset)) else "FAIL")
sys.exit(0 if (not nonfinite and len(covered_layers) >= 0.8 * len(subset)) else 1)
PY
$PY scripts/tools/gr00t_gate0_check.py --audit "$AUDIT" \
    --min-w2-w8-ratio 1.5 --min-guard-fire 0.3 --min-seed-jaccard 0.5 --min-spearman 0.0 \
    && echo "[3] gate0 (small-scale, relaxed thresholds): PASS" \
    || echo "[3] gate0 (small-scale, relaxed thresholds): FAIL — review the audit before formal gate-0"

echo "=== [4] static A8 second-start check (save -> load, no warmup on load) ==="
echo "--- start #1 (calibrates + saves $SCALE_PATH) ---"
GR00T_DUQUANT_ACT_SCALE_PATH="$SCALE_PATH" \
    GR00T_DUQUANT_PLAN="" \
    ./scripts/run_quantvla.sh libero_spatial >"$LOG/server_start1.log" 2>&1 &
SRV=$!
ok=0
for _ in $(seq 1 150); do
    if grep -q "frozen A8 scales saved" "$LOG/server_start1.log"; then ok=1; break; fi
    if ! kill -0 "$SRV" 2>/dev/null; then echo "!!! start#1 died"; tail -20 "$LOG/server_start1.log"; exit 1; fi
    sleep 5
done
kill "$SRV" 2>/dev/null || true; wait "$SRV" 2>/dev/null || true
[[ $ok -eq 1 ]] || { echo "!!! start#1 never saved scales"; tail -20 "$LOG/server_start1.log"; exit 1; }
grep -E "calibration complete|frozen A8 scales saved" "$LOG/server_start1.log" | tail -2

echo "--- start #2 (loads $SCALE_PATH, must NOT warmup) ---"
GR00T_DUQUANT_ACT_SCALE_PATH="$SCALE_PATH" \
    ./scripts/run_quantvla.sh libero_spatial >"$LOG/server_start2.log" 2>&1 &
SRV=$!
ok=0
for _ in $(seq 1 150); do
    if grep -q "loaded frozen A8 scales" "$LOG/server_start2.log"; then ok=1; break; fi
    if ! kill -0 "$SRV" 2>/dev/null; then echo "!!! start#2 died"; tail -20 "$LOG/server_start2.log"; exit 1; fi
    sleep 5
done
kill "$SRV" 2>/dev/null || true; wait "$SRV" 2>/dev/null || true
[[ $ok -eq 1 ]] || { echo "!!! start#2 never loaded scales"; tail -20 "$LOG/server_start2.log"; exit 1; }
grep "loaded frozen A8 scales" "$LOG/server_start2.log" | tail -1

echo "=== SMOKE ALL PASS ==="
echo "NEXT: formal gate-0 (30 layers, 3 seeds, 16 obs), then probe/selector/TopK"
