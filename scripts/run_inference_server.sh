#!/bin/bash
# Script to run GR00T inference server for Libero evaluation
# Usage: ./scripts/run_inference_server.sh [task_suite_name]
# task_suite_name: libero_spatial (default), libero_goal, libero_object, libero_90, libero_10

TASK=${1:-libero_10}

# Use idle GPU (GPU 0-3 are occupied by other users on this machine)
export CUDA_VISIBLE_DEVICES=${GR00T_GPU:-4}

# flash-attn RPATH fix (manual .so install)
export LD_LIBRARY_PATH=/home1/gyy/probe/miniforge3/envs/groot_test/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH

export PYTHONPATH=/home1/gyy/vla/QuantVLA/code:$PYTHONPATH

# Activate groot_test environment
source /home1/gyy/probe/miniforge3/etc/profile.d/conda.sh
conda activate groot_test

# Set model path and data config based on task
case $TASK in
    libero_spatial)
        MODEL_PATH="/home1/gyy/vla/QuantVLA/checkpoints/gr00t/libero-spatial"
        DATA_CONFIG="examples.Libero.custom_data_config:LiberoDataConfig"
        ;;
    libero_goal)
        MODEL_PATH="/home1/gyy/vla/QuantVLA/checkpoints/gr00t/libero-goal"
        DATA_CONFIG="examples.Libero.custom_data_config:LiberoDataConfigMeanStd"
        ;;
    libero_object)
        MODEL_PATH="/home1/gyy/vla/QuantVLA/checkpoints/gr00t/libero-object"
        DATA_CONFIG="examples.Libero.custom_data_config:LiberoDataConfig"
        ;;
    libero_90)
        MODEL_PATH="/home1/gyy/vla/QuantVLA/checkpoints/gr00t/libero-90"
        DATA_CONFIG="examples.Libero.custom_data_config:LiberoDataConfig"
        ;;
    libero_10)
        MODEL_PATH="/home1/gyy/vla/QuantVLA/checkpoints/gr00t/libero-long"
        DATA_CONFIG="examples.Libero.custom_data_config:LiberoDataConfig"
        ;;
    *)
        echo "Unknown task: $TASK"
        echo "Available tasks: libero_spatial, libero_goal, libero_object, libero_90, libero_10"
        exit 1
        ;;
esac

# Allow override via GR00T_MODEL_PATH environment variable
MODEL_PATH="${GR00T_MODEL_PATH:-$MODEL_PATH}"

# Allow override of denoising steps via environment variable
DENOISING_STEPS=${GR00T_DENOISING_STEPS:-8}

echo "=========================================="
echo "Starting GR00T inference server for $TASK"
echo "Model: $MODEL_PATH"
echo "Data Config: $DATA_CONFIG"
echo "Port: 5556"
echo "Denoising Steps: $DENOISING_STEPS"
echo "=========================================="

cd /home1/gyy/vla/QuantVLA

python scripts/inference_service.py \
    --model_path $MODEL_PATH \
    --server \
    --data_config $DATA_CONFIG \
    --denoising-steps 8 \
    --port ${GR00T_PORT:-5556} \
    --embodiment-tag new_embodiment
