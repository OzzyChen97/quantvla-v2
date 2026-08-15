#!/bin/bash
# QuantVLA v2 (v1.3, P0-corrected) gated GPU experiment pipeline.
#
# Review round 3, item 7: this is a REAL pipeline — gate failures abort, the
# consensus plan is frozen for held-out suites, and LIBERO comparisons run
# inside this script with a managed server lifecycle (not printed commands).
#
# Flow (docs/quantvla_v2_design.md §6.6):
#   CPU selftests
#     -> gate 0 metric audit (+ gr00t_gate0_check.py HARD gate)
#     -> dev-suite W4 probe
#     -> binary selector (guards, diverse TopK)
#     -> baseline generation + stage-2 D_solver screening
#     -> TopK adjudication (gr00t_topk_scorer)
#     -> consensus plan (mask Jaccard >= 0.7 gate)
#     -> dev LIBERO: v2 final plan vs uniform W6 vs random best/median/worst (same seed)
#     -> freeze decisions (runs/v2_decisions.md)
#     -> final-holdout: libero_10 + libero_90, 3 seeds
#
# Usage:
#   ./scripts/run_v2_gpu_experiment.sh all-dev          # spatial+goal+object
#   ./scripts/run_v2_gpu_experiment.sh dev-accept       # run the dev LIBERO comparison table
#   ./scripts/run_v2_gpu_experiment.sh final-holdout    # held-out acceptance (3 seeds)
#   GR00T_GPU=1 ./scripts/run_v2_gpu_experiment.sh all-dev
set -euo pipefail

REPO=/home1/gyy/vla/QuantVLA
cd "$REPO"
export PYTHONPATH="$REPO/code:$REPO/scripts/tools:${PYTHONPATH:-}"
PY=/home1/gyy/probe/miniforge3/envs/groot_test/bin/python
export CUDA_VISIBLE_DEVICES=${GR00T_GPU:-4}
PORT=${GR00T_PORT:-5556}
LOG="$REPO/runs/v2_gpu_logs"
export LIBERO_LOG_DIR="$REPO/runs/libero_logs"
mkdir -p "$LOG"
export GR00T_ATM_ENABLE=0 GR00T_OHB_ENABLE=0   # core-method runs: NO scale correction

DEV_SUITES=(spatial goal object)
MODE="${1:-all-dev}"

run_selftests() {
    echo "=== [gate] CPU selftests ==="
    $PY -m gr00t.quantization.kernel_scores >/dev/null 2>&1
    $PY -m gr00t.quantization.duquant_layers >/dev/null 2>&1
    for s in gr00t_select_plan gr00t_sensitivity_probe gr00t_metric_audit \
             gr00t_baselines gr00t_topk_scorer gr00t_consensus_plan gr00t_gate0_check; do
        case $s in
            gr00t_baselines) $PY scripts/tools/$s.py --mode selftest >/dev/null 2>&1 ;;
            *) $PY scripts/tools/$s.py --selftest >/dev/null 2>&1 ;;
        esac
    done
    $PY scripts/tools/calibrate_atm_perstep_gr00t.py --selftest >/dev/null 2>&1
    echo "=== selftests: ALL PASS ==="
}

# --------------------------------------------------------------------------- #
# Managed server lifecycle (real LIBERO runs)
# --------------------------------------------------------------------------- #
start_server() {
    local suite=$1 plan=$2
    local model_suffix=$suite
    [[ "$suite" == "10" ]] && model_suffix="long"
    local logf="$LOG/server_${suite}_$(basename "$plan").log"
    echo "--- starting server: libero_$suite plan=$plan (log: $logf)"
    # Force-free the port: a stale server from an earlier wave must never
    # satisfy the readiness check while the new server dies with
    # "Address already in use" (this exact failure stalled the whole fleet).
    local holder hcmd
    holder=$(ss -tlnp 2>/dev/null | grep ":$PORT " | grep -oP 'pid=\K[0-9]+' | head -1 || true)
    if [[ -n "$holder" ]]; then
        hcmd=$(ps -p "$holder" -o args= 2>/dev/null || true)
        # only kill OUR OWN stale servers, never another user's service
        if [[ "$hcmd" == *inference_service* || "$hcmd" == *run_quantvla* || "$hcmd" == *run_inference_server* ]]; then
            echo "!!! port $PORT held by our stale server pid $holder — killing it first"
            kill "$holder" 2>/dev/null || true
            for _ in $(seq 1 20); do
                ss -tln 2>/dev/null | grep -q ":$PORT " || break
                sleep 1
            done
            if ss -tln 2>/dev/null | grep -q ":$PORT "; then
                kill -KILL "$holder" 2>/dev/null || true
                sleep 5
            fi
        else
            echo "!!! port $PORT held by pid $holder which is NOT our server — refusing to kill; aborting"
            return 1
        fi
    fi
    GR00T_DUQUANT_PLAN="$plan" GR00T_GPU=${GR00T_GPU:-4} GR00T_PORT="$PORT" \
        ./scripts/run_quantvla.sh "libero_$suite" >"$logf" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 150); do
        if grep -q "Address already in use" "$logf" 2>/dev/null; then
            echo "!!! new server could not bind $PORT — tail:"; tail -15 "$logf"; return 1
        fi
        # port up AND the log must show the CORRECT suite checkpoint AND the
        # freshly-launched process must still be alive
        if ss -tln 2>/dev/null | grep -q ":$PORT "; then
            if grep -q "Model: .*libero-$model_suffix" "$logf" 2>/dev/null && kill -0 "$SERVER_PID" 2>/dev/null; then
                return 0
            fi
            echo "!!! port $PORT answered but log does not show libero-$model_suffix model or new server died — killing and retrying"
            kill "$SERVER_PID" 2>/dev/null || true
            return 1
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "!!! server exited early — tail of $logf:"; tail -25 "$logf"; return 1
        fi
        sleep 5
    done
    echo "!!! server port timeout"; return 1
}

stop_server() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=""
    sleep 5
}

run_libbero() {
    local suite=$1; shift
    # task sharding (review round 5, speedup): TASK_IDS="0 1 2 3 4"
    local extra=()
    if [[ -n "${TASK_IDS:-}" ]]; then
        extra=(--task_ids ${TASK_IDS})
    fi
    # config-level timeout (ZMQ client also has a 15s per-request timeout)
    LIBERO_PORT="$PORT" timeout --signal=TERM "${EVAL_TIMEOUT:-21600}" \
        ./scripts/run_libero_eval.sh "libero_$suite" --headless "$@" "${extra[@]}"
}

# progress watchdog: run the eval in background; if the episode counter stops
# growing for STALL_LIMIT seconds, dump diagnostics, kill the pair, retry ONCE.
run_eval_watchdog() {
    local suite=$1; shift
    local stall_limit=${STALL_LIMIT:-1800}
    local attempt
    for attempt in 1 2; do
        run_libbero "$suite" "$@" &
        local EVAL_PID=$!
        local last=-1 now=0 stall=0
        while kill -0 "$EVAL_PID" 2>/dev/null; do
            now=$(grep -c "episodes completed so far" "$LOG/liberos_${suite}.log" 2>/dev/null || true)
            if [[ "$now" != "$last" ]]; then last=$now; stall=0; else stall=$((stall + 60)); fi
            if [[ $stall -ge $stall_limit ]]; then
                echo "!!! watchdog: no episode growth for ${stall}s (attempt $attempt) — dumping diagnostics"
                nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader > "$LOG/watchdog_${suite}_attempt${attempt}.smi" 2>/dev/null || true
                ps -ef --forest | grep -A3 -B1 "inference_service\|run_libero" >> "$LOG/watchdog_${suite}_attempt${attempt}.ps" 2>/dev/null || true
                tail -30 "$LOG/liberos_${suite}.log" > "$LOG/watchdog_${suite}_attempt${attempt}.log" 2>/dev/null || true
                kill "$EVAL_PID" 2>/dev/null || true
                stop_server
                break
            fi
            sleep 60
        done
        if ! kill -0 "$EVAL_PID" 2>/dev/null; then
            wait "$EVAL_PID" 2>/dev/null && return 0 || {
                [[ $attempt -eq 2 ]] && { echo "!!! eval failed twice — aborting"; return 1; }
                echo "!!! eval attempt $attempt failed — retrying once"
                stop_server
                continue
            }
        fi
        [[ $attempt -eq 2 ]] && { echo "!!! watchdog fired twice — aborting"; return 1; }
    done
    return 1
}

# --------------------------------------------------------------------------- #
run_dev_suite() {
    local S=$1
    local Ckpt=$REPO/checkpoints/gr00t/libero-$S
    local PK=$REPO/checkpoints/packs/gr00t/duquant_packed_libero_${S}_w4a8_b64c32ls015
    echo ""
    echo "############################################################"
    echo "# DEV PIPELINE: libero_$S"
    echo "############################################################"

    echo "--- [gate 0] metric audit ($S) ---"
    $PY scripts/tools/gr00t_metric_audit.py --suite "$S" \
        --layers-subset 30 --bits 2,4,6,8 --n-seeds 3 --n-obs 8 --n-rollout-obs 4
    $PY scripts/tools/gr00t_gate0_check.py \
        --audit "$REPO/checkpoints/packs/gr00t/metric_audit_libero_${S}.json" \
        || { echo "!!! GATE 0 FAILED for $S — pipeline aborts"; exit 1; }

    echo "--- [probe] W4-only sensitivity ($S) ---"
    $PY scripts/tools/gr00t_sensitivity_probe.py --suite "$S" \
        --n-obs 16 --bits 4 --group 64 --n-rollout-obs 8 --cs-in-situ-check

    echo "--- [selector] binary W4/FP16 ($S) ---"
    $PY scripts/tools/gr00t_select_plan.py \
        --sensitivity "$REPO/checkpoints/packs/gr00t/sensitivity_libero_${S}_g64_b4.json" \
        --ckpt "$Ckpt" \
        --out "$REPO/checkpoints/packs/gr00t/gr00t_quant_plan_libero_${S}.json" \
        --solver greedy --binary --min-bits 4 --emit-env

    echo "--- [baselines] generate + stage-2 ($S) ---"
    $PY scripts/tools/gr00t_baselines.py --mode generate --suite "$S" \
        --ckpt "$Ckpt" \
        --ref-plan "$REPO/checkpoints/packs/gr00t/gr00t_quant_plan_libero_${S}.json" \
        --n-random 20 --out-dir "$REPO/checkpoints/packs/gr00t/baselines_${S}"
    $PY scripts/tools/gr00t_baselines.py --mode stage2 --suite "$S" \
        --plans-dir "$REPO/checkpoints/packs/gr00t/baselines_${S}" \
        --out "$REPO/checkpoints/packs/gr00t/baselines_${S}_dsolver.json"

    echo "--- [TopK adjudication] ($S) ---"
    $PY scripts/tools/gr00t_topk_scorer.py \
        --plan "$REPO/checkpoints/packs/gr00t/gr00t_quant_plan_libero_${S}.json" \
        --ckpt "$Ckpt" --suite "$S" --packdir "$PK" \
        --out "$REPO/checkpoints/packs/gr00t/gr00t_quant_plan_libero_${S}_adjudicated"
}

# --------------------------------------------------------------------------- #
# dev LIBERO comparison table: v2 final vs uniform W6 vs random best/median/worst
# --------------------------------------------------------------------------- #
dev_accept() {
    local S=${1:-spatial}
    local FINAL="$REPO/checkpoints/packs/gr00t/gr00t_quant_plan_libero_${S}_adjudicated.final_plan.json"
    [[ -f "$FINAL" ]] || { echo "!!! final plan missing: $FINAL"; exit 1; }
    local BDIR="$REPO/checkpoints/packs/gr00t/baselines_${S}"

    # representative random masks from the stage-2 report
    mapfile -t REPS < <($PY - "$REPO/checkpoints/packs/gr00t/baselines_${S}_dsolver.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
rep = d["representatives"]
print(rep["best"]["file"]); print(rep["median"]["file"]); print(rep["worst"]["file"])
PY
)
    echo "--- dev LIBERO ($S): v2 final + uniform_w6 + random best/median/worst (seed 0) ---"
    local PLAN
    for PLAN in "$FINAL" "$BDIR/uniform_w6.json" \
                "$BDIR/${REPS[0]}" "$BDIR/${REPS[1]}" "$BDIR/${REPS[2]}"; do
        start_server "$S" "$PLAN" || exit 1
        run_eval_watchdog "$S" --seed 0
        stop_server
    done
    echo "--- dev LIBERO ($S) done; record numbers in runs/v2_decisions.md ---"
}

# --------------------------------------------------------------------------- #
consensus_freeze() {
    echo "--- [consensus] mask Jaccard gate + frozen unified plan ---"
    $PY scripts/tools/gr00t_consensus_plan.py \
        --plans "$REPO/checkpoints/packs/gr00t/gr00t_quant_plan_libero_spatial_adjudicated.final_plan.json" \
                "$REPO/checkpoints/packs/gr00t/gr00t_quant_plan_libero_goal_adjudicated.final_plan.json" \
                "$REPO/checkpoints/packs/gr00t/gr00t_quant_plan_libero_object_adjudicated.final_plan.json" \
        --ckpt "$REPO/checkpoints/gr00t/libero-spatial" \
        --budget uniform-w6 \
        --out "$REPO/checkpoints/packs/gr00t/gr00t_quant_plan_consensus.json" \
        || { echo "!!! consensus gate failed — masks not reproducible"; exit 1; }
}

final_holdout() {
    # HOLD_SUITES env selects the held-out suites (default "10 90"; the v1.3
    # full-test scope runs libero_10 only).
    for S in ${HOLD_SUITES:-10 90}; do
    # D-008: the 3-suite consensus Jaccard gate FAILED (0.39-0.50 < 0.7) — layer
    # sensitivity is checkpoint-specific, so no unified plan was frozen. The
    # held-out Long suite therefore runs the SPATIAL adjudicated plan zero-shot
    # (spatial = primary suite, most-developed protocol; recorded in
    # runs/v2_decisions.md D-008).
    local PLAN="$REPO/checkpoints/packs/gr00t/gr00t_quant_plan_libero_spatial_adjudicated.final_plan.json"
    [[ -f "$PLAN" ]] || { echo "!!! spatial adjudicated plan missing — run spatial pipeline first"; exit 1; }
        # per seed: transfer-v2 (spatial mask + Long pack) AND the same-budget
        # uniform-W6 baseline on the Long checkpoint (review round 5, item 5)
        local PLANS=(
            "$REPO/checkpoints/packs/gr00t/gr00t_quant_plan_long_transfer_v2.json"
            "$REPO/checkpoints/packs/gr00t/gr00t_quant_plan_long_transfer_w6.json"
        )
        for SEED in ${HOLD_SEEDS:-0 1 2}; do
            for PLAN2 in "${PLANS[@]}"; do
                echo "--- held-out: libero_$S seed=$SEED plan=$(basename "$PLAN2") ---"
                start_server "$S" "$PLAN2" || exit 1
                run_eval_watchdog "$S" --seed "$SEED"
                stop_server
            done
        done
    done
    echo "--- held-out acceptance done; task-level cluster stats go into the paper report ---"
}

case "$MODE" in
    all-dev)
        run_selftests
        for S in "${DEV_SUITES[@]}"; do run_dev_suite "$S"; done
        consensus_freeze
        echo ""
        echo ">>> NEXT: ./scripts/run_v2_gpu_experiment.sh dev-accept <suite>"
        echo ">>> then record decisions in runs/v2_decisions.md and run final-holdout"
        ;;
    dev-accept)
        S2="${2:-}"
        [[ -n "$S2" ]] || { echo "usage: $0 dev-accept <suite>"; exit 1; }
        dev_accept "$S2"
        ;;
    final-holdout)
        final_holdout
        ;;
    *)
        echo "usage: $0 {all-dev|dev-accept|final-holdout}"; exit 1 ;;
esac
