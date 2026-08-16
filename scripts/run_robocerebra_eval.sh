#!/bin/bash
# RoboCerebra pi0.5 eval client. Talks to the pi05 LIBERO policy server:
#   fp16:  pi05/run_libero_serve.sh          (default port 8001)
#   W4A8:  pi05/run_libero_serve_quant.sh    (default port 8002)
#
# Usage:
#   ./scripts/run_robocerebra_eval.sh --task_types '["Ideal"]' --num_trials_per_task 2 --port 8001
#   ./scripts/run_robocerebra_eval.sh --cases case1,case3 --num_trials_per_task 1
# Full benchmark (paper protocol, 60 tasks x 10 trials):
#   ./scripts/run_robocerebra_eval.sh --num_trials_per_task 10
#
# Outputs: <repo>/runs/robocerebra_logs/  (txt + results json)
#          <repo>/runs/robocerebra_rollouts/ (per-episode/segment mp4)
set -e

REPO=/home1/gyy/vla/QuantVLA
source /home1/gyy/probe/miniforge3/etc/profile.d/conda.sh
conda activate robocerebra_test

export PYTHONPATH=$REPO/code:$REPO:$REPO/code/pi05/openpi/packages/openpi-client/src:$PYTHONPATH
export ROBOCEREBRA_BENCH_ROOT=${ROBOCEREBRA_BENCH_ROOT:-$REPO/data/RoboCerebra_dl/RoboCerebraBench}
export ROBOCEREBRA_INIT_FILES_ROOT=${ROBOCEREBRA_INIT_FILES_ROOT:-$ROBOCEREBRA_BENCH_ROOT/init_files}
export ROBOCEREBRA_ROLLOUT_DIR=${ROBOCEREBRA_ROLLOUT_DIR:-$REPO/runs/robocerebra_rollouts}
export ROBOCEREBRA_LOG_DIR=${ROBOCEREBRA_LOG_DIR:-$REPO/runs/robocerebra_logs}
export MUJOCO_GL=egl
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-$REPO/runs/numba_cache}
mkdir -p "$NUMBA_CACHE_DIR"

cd $REPO/code/examples/RoboCerebra/eval
ENTRY=eval_pi05.py
[[ "${ROBOCEREBRA_POLICY:-}" == "gr00t" ]] && ENTRY=eval_gr00t.py
exec python "$ENTRY" --local_log_dir "$ROBOCEREBRA_LOG_DIR" "$@"

# ---- GR00T mode (v1.4 Stage D bridge): point at the GR00T LIBERO inference
# service (scripts/run_inference_server.sh libero_spatial, default :5556) ----
#   ./scripts/run_robocerebra_eval.sh --policy gr00t --task_types '["Ideal"]' --num_trials_per_task 1 --port 5565
if [[ "${1:-}" == "--policy" ]]; then
    POLICY="$2"; shift 2
    if [[ "$POLICY" == "gr00t" ]]; then
        export ROBOCEREBRA_POLICY=gr00t
    fi
fi
