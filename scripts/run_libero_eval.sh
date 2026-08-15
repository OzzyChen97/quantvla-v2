#!/bin/bash
# Script to run Libero evaluation
# Usage: ./scripts/run_libero_eval.sh [task_suite_name] [extra args...]
# task_suite_name: libero_spatial (default), libero_goal, libero_object, libero_90, libero_10

TASK=${1:-libero_10}
shift || true
EXTRA_ARGS=("$@")

HEADLESS_FLAG="no"
for arg in "${EXTRA_ARGS[@]}"; do
    if [[ "$arg" == "--headless" ]]; then
        HEADLESS_FLAG="yes"
        break
    fi
done

# Activate libero_test environment
source /home1/gyy/probe/miniforge3/etc/profile.d/conda.sh
conda activate libero_test

# Add QuantVLA and LIBERO to Python path
export PYTHONPATH=/home1/gyy/vla/QuantVLA/code:/home1/gyy/vla/QuantVLA:/home1/gyy/vla/QuantVLA/code/LIBERO:$PYTHONPATH

# numba cache must live in a writable location (background jobs have a
# restricted /tmp; the default cache path makes numba raise "no locator
# available" when it tries to (re)cache robosuite's jitted functions)
mkdir -p /home1/gyy/vla/QuantVLA/runs/numba_cache
export NUMBA_CACHE_DIR=/home1/gyy/vla/QuantVLA/runs/numba_cache

echo "=========================================="
echo "Running Libero evaluation for $TASK"
echo "Headless mode: $HEADLESS_FLAG"
echo "Port: 5556 (GR00T)"
echo "=========================================="
echo ""
echo "Make sure the inference server is running in another terminal!"
echo "Run: ./scripts/run_inference_server.sh $TASK"
echo ""
echo "Results will be saved to:"
echo "  - Log: /tmp/logs/libero_eval_${TASK}.log"
echo "  - Videos: /tmp/logs/rollout_*.mp4"
echo "=========================================="
echo ""

cd /home1/gyy/vla/QuantVLA/code/examples/Libero/eval

python -u run_libero_eval.py --task_suite_name "$TASK" --port ${LIBERO_PORT:-5556} "${EXTRA_ARGS[@]}"
