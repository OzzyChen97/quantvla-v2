#!/bin/bash
# QuantVLA v2 GR00T GPU 实验编排（仅 v2 plan 完整量化版）
# 阶段：probe 冒烟 → 正式 probe(spatial+long 并行) → 选择 → spatial 评测 → long 分片评测 → 聚合
# 用法:
#   ./scripts/run_v2_gpu_experiment.sh            # 自动选空闲卡
#   ./scripts/run_v2_gpu_experiment.sh 0,1,2,3,4  # 显式指定卡
# 建议后台运行:  nohup ./scripts/run_v2_gpu_experiment.sh > runs/v2_experiment/orchestrator.log 2>&1 &
set -uo pipefail

ROOT=/home1/gyy/vla/QuantVLA
LOGDIR=$ROOT/runs/v2_experiment/logs
mkdir -p "$LOGDIR"
CONDA_BASE=/home1/gyy/probe/miniforge3
source "$CONDA_BASE/etc/profile.d/conda.sh"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOGDIR/orchestrator.log"; }

# ---------------- GPU 选择 ----------------
FREE_GPUS=""
if [ "${1:-}" != "" ]; then
    FREE_GPUS=$(echo "$1" | tr ',' ' ')
else
    mapfile -t ALL_FREE < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | awk -F', ' '$2 < 2048 {print $1}')
    FREE_GPUS="${ALL_FREE[*]}"
fi
read -ra GPUS <<< "$FREE_GPUS"
if [ "${#GPUS[@]}" -lt 2 ]; then
    log "错误：至少需要 2 张空闲 GPU（当前: '${GPUS[*]}'）。"
    log "查看: nvidia-smi --query-gpu=index,memory.used --format=csv"
    exit 1
fi
log "使用 GPU: ${GPUS[*]}（空闲判定: 显存<2GB）"
PROBE_G0=${GPUS[0]}; PROBE_G1=${GPUS[1]}

SERVER_PIDS=()

cleanup() {
    for pid in "${SERVER_PIDS[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT

wait_port() {  # host port timeout_s
    local host=$1 port=$2 tries=${3:-600}
    for _ in $(seq 1 "$tries"); do
        (exec 3<>"/dev/tcp/$host/$port") 2>/dev/null && { exec 3>&- 3<&-; return 0; }
        sleep 1
    done
    return 1
}

# ---------------- Stage 1: probe 冒烟 ----------------
log "Stage 1/6: probe 冒烟 (spatial, n_obs=4, bits 4,8)"
conda activate groot_test
cd "$ROOT"
export PYTHONPATH=$ROOT/code:${PYTHONPATH:-}
CUDA_VISIBLE_DEVICES=$PROBE_G0 python scripts/tools/gr00t_sensitivity_probe.py \
    --suite spatial --n-obs 4 --bits 4,8 --skip-layer-rollouts \
    --out runs/v2_smoke_sensitivity.json 2>&1 | tee "$LOGDIR/smoke_probe.log"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then log "冒烟 probe 失败，终止。见 $LOGDIR/smoke_probe.log"; exit 1; fi
python - "$ROOT/runs/v2_smoke_sensitivity.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
lay = d["layers"]; bad = []
for n, e in lay.items():
    for k, v in e.items():
        if k.startswith("b") and isinstance(v, dict):
            if v.get("cka") is not None and not (0 <= v["cka"] <= 1): bad.append((n, k, v["cka"]))
            if v.get("cs") is not None and v["cs"] < 0: bad.append((n, k, v["cs"]))
assert d["meta"]["reference_protocol"].startswith("wrapped pipeline"), "参照协议字段缺失"
assert len([n for n in lay if not n.startswith("attn:")]) >= 100, "目标层数不足"
print(f"冒烟 sanity OK: {len(lay)} 层, 异常分数 {len(bad)}")
assert not bad, bad
EOF
log "Stage 1 通过"

# ---------------- Stage 2: 正式 probe（spatial + long 并行） ----------------
log "Stage 2/6: 正式 probe (spatial on GPU${PROBE_G0}, long on GPU${PROBE_G1}, 并行)"
CUDA_VISIBLE_DEVICES=$PROBE_G0 python scripts/tools/gr00t_sensitivity_probe.py \
    --suite spatial --n-obs 16 --bits 2,3,4,6,8 > "$LOGDIR/probe_spatial.log" 2>&1 &
PID_SP=$!
CUDA_VISIBLE_DEVICES=$PROBE_G1 python scripts/tools/gr00t_sensitivity_probe.py \
    --suite 10 --n-obs 16 --bits 2,3,4,6,8 > "$LOGDIR/probe_long.log" 2>&1 &
PID_LG=$!
wait $PID_SP; R1=$?
wait $PID_LG; R2=$?
if [ $R1 -ne 0 ] || [ $R2 -ne 0 ]; then log "probe 失败（spatial=$R1 long=$R2），见 probe_*.log"; exit 1; fi
log "Stage 2 完成"

# ---------------- Stage 3: 选择 ----------------
log "Stage 3/6: 生成 quant plan"
for suite in spatial 10; do
    ckpt=$ROOT/checkpoints/gr00t/$( [ "$suite" = spatial ] && echo libero-spatial || echo libero-long )
    python scripts/tools/gr00t_select_plan.py \
        --sensitivity "$ROOT/checkpoints/packs/gr00t/sensitivity_libero_${suite}_g64_b2_3_4_6_8.json" \
        --ckpt "$ckpt" \
        --out "$ROOT/checkpoints/packs/gr00t/gr00t_quant_plan_libero_${suite}.json" \
        --solver greedy --emit-env 2>&1 | tee "$LOGDIR/select_${suite}.log"
    python - "$ROOT/checkpoints/packs/gr00t/gr00t_quant_plan_libero_${suite}.json" <<'EOF'
import json, sys
p = json.load(open(sys.argv[1]))
assert p["total_bytes"] <= p["budget_bytes"] + 1e-6, f"超预算: {p['total_bytes']} > {p['budget_bytes']}"
print(f"预算 OK: {p['total_bytes']/1e6:.1f}/{p['budget_bytes']/1e6:.1f} MB, 目标={p['objective']:.4f}")
EOF
done
log "Stage 3 完成"

# ---------------- Stage 4+5（全并行，占用全部空闲卡）: spatial + long 分片同跑 ----------------
N_LONG=$(( ${#GPUS[@]} < 4 ? ${#GPUS[@]} : 4 ))   # long 分片数 = 卡数（上限 4）
log "Stage 4+5/6: spatial 与 long（${N_LONG} 片）并行评测，卡布局: spatial 与 shard0 共卡 GPU${GPUS[0]}"
SHARD_TASKS=()
if [ "$N_LONG" -eq 4 ]; then SHARD_TASKS=("0 1 2" "3 4" "5 6 7" "8 9");
elif [ "$N_LONG" -eq 3 ]; then SHARD_TASKS=("0 1 2 3" "4 5 6" "7 8 9");
elif [ "$N_LONG" -eq 2 ]; then SHARD_TASKS=("0 1 2 3 4" "5 6 7 8 9");
else SHARD_TASKS=("0 1 2 3 4 5 6 7 8 9"); fi

conda activate groot_test   # 服务器必须用 groot_test
# spatial 服务器（与 shard0 共卡 GPU[0]）
CUDA_VISIBLE_DEVICES=${GPUS[0]} GR00T_GPU=${GPUS[0]} \
GR00T_DUQUANT_PLAN=$ROOT/checkpoints/packs/gr00t/gr00t_quant_plan_libero_spatial.json \
    "$ROOT/scripts/run_quantvla.sh" libero_spatial > "$LOGDIR/server_spatial.log" 2>&1 &
SERVER_PIDS+=($!)
# long 分片服务器（round-robin 铺满所有卡）
for i in $(seq 0 $((N_LONG - 1))); do
    G=${GPUS[$((i % ${#GPUS[@]}))]}
    PORT=$((5560 + i))
    CUDA_VISIBLE_DEVICES=$G GR00T_GPU=$G GR00T_PORT=$PORT \
    GR00T_DUQUANT_PLAN=$ROOT/checkpoints/packs/gr00t/gr00t_quant_plan_libero_10.json \
        "$ROOT/scripts/run_quantvla.sh" libero_10 > "$LOGDIR/server_long_shard${i}.log" 2>&1 &
    SERVER_PIDS+=($!)
done
if ! wait_port 127.0.0.1 5556 600; then log "spatial 服务器未就绪，终止"; exit 1; fi
log "spatial 服务器就绪 (port 5556)"
for i in $(seq 0 $((N_LONG - 1))); do
    PORT=$((5560 + i))
    if ! wait_port 127.0.0.1 "$PORT" 600; then log "long shard${i} 服务器未就绪"; exit 1; fi
    log "long shard${i} 服务器就绪 (port $PORT)"
done
conda activate libero_test
export PYTHONPATH=$ROOT/code:$ROOT:$ROOT/code/LIBERO
cd "$ROOT/code/examples/Libero/eval"
CLIENT_PIDS=()
LIBERO_LOG_DIR="/tmp/logs/spatial" SKIP_VIDEO=1 \
    python run_libero_eval.py --task_suite_name libero_spatial --port 5556 --headless \
    > "$LOGDIR/eval_spatial.log" 2>&1 &
CLIENT_PIDS+=($!)
for i in $(seq 0 $((N_LONG - 1))); do
    PORT=$((5560 + i))
    LIBERO_LOG_DIR="/tmp/logs/shard${i}" SKIP_VIDEO=1 \
        python run_libero_eval.py --task_suite_name libero_10 --port "$PORT" --headless \
        --task-ids ${SHARD_TASKS[$i]} > "$LOGDIR/eval_long_shard${i}.log" 2>&1 &
    CLIENT_PIDS+=($!)
done
for pid in "${CLIENT_PIDS[@]}"; do wait "$pid" || log "某评测客户端异常退出，见 eval_*.log"; done
log "Stage 4+5 完成"

# ---------------- Stage 6: 聚合 ----------------
log "Stage 6/6: 聚合结果"
python - <<EOF
import glob, json, re
total = 0; per = {}
for f in sorted(glob.glob("/tmp/logs/shard*/libero_eval_libero_10.log")):
    txt = open(f).read()
    m = re.findall(r"Current task success rate: ([\d.]+)", txt)
    s = re.findall(r"# successes: (\d+)", txt)
    last = int(s[-1]) if s else 0
    shard = f.split("/")[-2]
    total += last; per[shard] = {"successes": last, "episodes": len(m) * 5 if m else None}
    print(f"  {shard}: successes={last}")
summary = {"suite": "libero_10", "total_successes": total, "total_episodes": 50,
           "success_rate": total / 50.0, "per_shard": per}
json.dump(summary, open("$ROOT/runs/v2_experiment/summary.json", "w"), indent=2)
print(f"long 聚合: {total}/50 = {total/50.0*100:.1f}%")
EOF
log "全部阶段完成。汇总: $ROOT/runs/v2_experiment/summary.json"
log "产物: sensitivity/plan json 在 checkpoints/packs/gr00t/；日志在 $LOGDIR/"
