#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

groot_python=/home1/gyy/probe/miniforge3/envs/groot_test/bin/python
matrix_runner=scripts/tools/run_robocasa_atomic_matrix.py
matrix_parser=scripts/tools/parse_robocasa_atomic_matrix.py
seeds=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49

if [[ -n "${ROBOCASA_WAIT_PID:-}" ]]; then
    while kill -0 "$ROBOCASA_WAIT_PID" 2>/dev/null; do
        sleep 30
    done
fi

ensure_matrix() {
    local task_set=$1
    local spec=$2
    local run_dir=$3
    local checkpoint=$4

    if [[ -f "$run_dir/manifest.json" ]] && \
       "$groot_python" "$matrix_parser" --run-dir "$run_dir" --bootstrap 10000 \
           >"$run_dir/strict_parse.log" 2>&1; then
        return
    fi
    until "$groot_python" "$matrix_runner" \
        --spec "$spec" \
        --run-dir "$run_dir" \
        --phase formal \
        --task-set "$task_set" \
        --seeds "$seeds" \
        --checkpoint "$checkpoint" \
        --n-shards 2 \
        --trial-batch-size 10 \
        --trial-timeout 3600 \
        --action-noise paired \
        --gpu-sample-interval 10; do
        sleep 30
    done
    "$groot_python" "$matrix_parser" --run-dir "$run_dir" --bootstrap 10000 \
        >"$run_dir/strict_parse.log"
}

checkpoint_root=checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/target_posttraining

ensure_matrix \
    atomic_seen \
    runs/robocasa365_official_full_atomic_spec.json \
    runs/robocasa365_official_full_atomic_paired50 \
    "$checkpoint_root/atomic_seen/checkpoint-60000"

ensure_matrix \
    composite_seen \
    runs/robocasa365_official_full_composite_seen_spec.json \
    runs/robocasa365_official_full_composite_seen_paired50 \
    "$checkpoint_root/composite_seen/checkpoint-60000"

ensure_matrix \
    composite_unseen \
    runs/robocasa365_official_full_composite_unseen_spec.json \
    runs/robocasa365_official_full_composite_unseen_paired50 \
    "$checkpoint_root/composite_unseen/checkpoint-60000"

"$groot_python" scripts/tools/aggregate_robocasa365_official.py \
    --run-dir runs/robocasa365_official_full_atomic_paired50 \
    --run-dir runs/robocasa365_official_full_composite_seen_paired50 \
    --run-dir runs/robocasa365_official_full_composite_unseen_paired50 \
    --out-dir runs/robocasa365_official_full_paired50 \
    --bootstrap 10000
