#!/bin/bash
# RoboCasa365 GR00T quantized inference server (v1.4, D-031 criterion-4).
#
# Launches scripts/inference_service.py directly (not the LIBERO-specific
# run_quantvla.sh) with the RoboCasa365 data config + DuQuant plan + pack dir.
#
# Usage:
#   GR00T_GPU=4 GR00T_PORT=5571 GR00T_DUQUANT_PLAN=<plan.json> \
#     ./scripts/run_robocasa365_quant_serve.sh
#   # plan must reference (or be accompanied by) the robocasa365 pack dir:
#   export GR00T_DUQUANT_PACKDIR=checkpoints/packs/robocasa365/duquant_packed_robocasa365_w4a8_b64c32ls015
set -euo pipefail

REPO=/home1/gyy/vla/QuantVLA
cd "$REPO"
export PYTHONPATH="$REPO/code:$REPO/scripts/tools:${PYTHONPATH:-}"
PY=/home1/gyy/probe/miniforge3/envs/groot_test/bin/python
export CUDA_VISIBLE_DEVICES=${GR00T_GPU:-4}
PORT=${GR00T_PORT:-5571}

MODEL_PATH=${GR00T_MODEL_PATH:-$REPO/checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/target_posttraining/atomic_seen/checkpoint-60000}
DATA_CONFIG=${GR00T_DATA_CONFIG:-examples.RoboCasa365.custom_data_config:RoboCasa365DataConfig}
PACKDIR=${GR00T_DUQUANT_PACKDIR:-$REPO/checkpoints/packs/robocasa365/duquant_packed_robocasa365_w4a8_b64c32ls015}
PLAN=${GR00T_DUQUANT_PLAN:-}

export GR00T_DUQUANT_SCOPE=""
export GR00T_DUQUANT_INCLUDE=".*(backbone\.eagle_model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|action_head\.model\.transformer_blocks\.\d+\.ff\.net\.(0\.proj|2)).*"
export GR00T_DUQUANT_EXCLUDE="(?:^|\.)(vision|radio|norm|ln|layernorm|embed|lm_head|attn1)(?:\.|$)"
export GR00T_DUQUANT_WBITS_DEFAULT=4
export GR00T_DUQUANT_ABITS=8
export GR00T_DUQUANT_BLOCK=64
export GR00T_DUQUANT_PERMUTE=0
export GR00T_DUQUANT_ROW_ROT=restore
export GR00T_DUQUANT_ACT_PCT=99.9
export GR00T_DUQUANT_CALIB_STEPS=32
export GR00T_DUQUANT_LS=0.15
export GR00T_DUQUANT_PACKDIR="$PACKDIR"
export GR00T_DUQUANT_ACT_DYNAMIC=0
export GR00T_DUQUANT_DEBUG=0
export GR00T_OBS_FORMAT=robocasa365
if [[ -n "$PLAN" ]]; then
    export GR00T_DUQUANT_PLAN="$PLAN"
fi

exec "$PY" scripts/inference_service.py --server \
    --model-path "$MODEL_PATH" \
    --data-config "$DATA_CONFIG" \
    --embodiment-tag new_embodiment \
    --port "$PORT" \
    --denoising-steps 8
