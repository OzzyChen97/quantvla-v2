# QuantVLA v2 — P0 正确性审查回应（2026-08-15）

对公开仓库提交 `f770f3d`（v1.3 scaffolding）的独立代码审查结论：**方法框架完整，但存在
数个会直接污染敏感度、动作权重与 baseline 的实现错误，正式 LIBERO 实验暂不可跑。**
本文档记录全部 P0 项的修复、回归测试与产物处置。

## 结论对照

| 审查项 | 状态 | 修复 |
|---|---|---|
| P0-1 权重原地重复旋转 | ✅ 已修复 | `transform_weight_for_forward_optimized` 在块旋转前 `W.detach().clone()`；`_weight` 不再被原地修改。回归测试：`python -m gr00t.quantization.duquant_layers`（权重不可变性 + 4→8→4 顺序不变性 + bits=0 可重复性） |
| P0-2 A8 校准第一批即冻结 | ✅ 已修复 | `_get_act_scale` 静态路径：校准器未满 `cfg.calib_batches` 批次前只返回**临时 scale（不冻结）**，满批后 finalize 才冻结；probe/audit/baselines/calibrator 全部改为 **weight_bits=0（或 plan 部署态）下、calib_steps×batch_size 个 obs = 真正的 calib_steps 个 batch** 校准，并用 `all_calibrated()` 强校验，不满即 SystemExit。回归测试同上（生命周期：3 batch 不冻结 → 第 4 batch 冻结 → 冻结后不变） |
| P0-3 D_solver 加权和写成平均 | ✅ 已修复 | `solver_divergence` 改为 `sum(dim=0)`（Σw_k=1）；selftest 新增 `D_solver(2·ref) == 1.0` 精确断言（旧实现得 1/(T+1)） |
| P0-3 第二组噪声未配对 | ✅ 已修复 | probe 在 all-zero 参照态下补跑 `ref_traj_b = rollout(ref, noises_b)`；w_i 现在比较 `D(x^R(ε_a), x^Q(ε_a))` 与 `D(x^R(ε_b), x^Q(ε_b))` 两组**真正配对**的轨迹，再取中位数 |
| P0-4 metric audit 先 reset 再测 d_solver | ✅ 已修复 | `collect_one_seed` 顺序改为：设目标 bit → 激活采集 → q_traj rollout → 计算 d_solver → **最后才 reset**。旧代码测的是 D(R,R)≈0 |
| P0-5 run_quantvla.sh 强制 Long ATM/OHB | ✅ 已修复 | 默认 `GR00T_ATM_ENABLE=${GR00T_ATM_ENABLE:-0}`、`GR00T_OHB_ENABLE=${GR00T_OHB_ENABLE:-0}`；开启 ATM 但未给 ALPHA_PATH 直接报错退出。核心方法实验默认无尺度校正 |
| P0-6 baseline 候选集错误 / 字节不匹配 / wrap 0 层 | ✅ 已修复 | generate 模式候选集 = **ref-plan 的 layer keys**（缺失即报错）；random mask 用局部交换法匹配 **静态字节**（\|ΔC\|/C_v2 ≤ 0.5%）；stage2 改为 **GR00T_DUQUANT_PLAN 逐计划加载**（skip=不包装=真实 FP16 语义），`count_wrapped() == expected` 强校验，并完成 A8 校准 |
| P0-7 missing score=0 / λ-sweep 绕过 guard / 异质 objective 直接比较 / 无预算断言 | ✅ 已修复 | `build_scores` 缺失测量 → **None（不可用，保留 FP16）**；`lambda_sweep_plans(..., guarded_names=...)` 继承护栏过滤；所有候选用 **canonical (λ=1,1, 护栏过滤后) objective 重打分**后再排序/TopK（原生值存 `objective_native`）；主 plan 与 TopK 逐项 **bytes ≤ budget 硬断言**，违例即 SystemExit；`assert_plan_guards` 保证被护栏删除的层在任何候选里都不被量化 |
| P0-7 TopK 闭环缺失 | ✅ 已补齐 | 新增 `scripts/tools/gr00t_topk_scorer.py`：读取 selector TopK → 逐计划物化为 GR00T_DUQUANT_PLAN JSON → **真实部署语义加载**（skip=不包装，wrap 数强校验）→ D_solver 打分（含 per-obs std）→ `select_final()`（min D_solver，5% 平局集由 canonical proxy 打破）→ 输出最终 plan + 报告 |
| P0-8 calibrator 非 plan-specific / static_sufficient 只查 α | ✅ 已修复 | `--plan` 现在在加载量化模型**之前**设置 `GR00T_DUQUANT_PLAN`，attention 统计采集自**真实 W4/FP16 部署态**；post-process 中性强制仅作安全网；`compute_cv_stats` 输出 `alpha_static_sufficient` / `beta_static_sufficient` / `overall = α && β` |
| 护栏是近似（审查 §九） | ✅ 已声明 | `sat_rate`/probe meta/设计文档明确：D_sat 是**单标量 P99.9 代理**，不是下一层 per-channel A8 scale 的精确 clipping 测量；guard 比较的量化侧上游含 wrapper/A8，深层会混入累计漂移——工程启发式，论文需如实表述 |

## 回归测试（CPU 全绿）

```bash
python -m gr00t.quantization.kernel_scores          # v1.3 自测电池
python -m gr00t.quantization.duquant_layers         # P0-1 顺序不变性 + P0-2 校准生命周期
python scripts/tools/gr00t_select_plan.py --selftest          # P0-7 全部 + 护栏点火/milp/diversity/bootstrap
python scripts/tools/gr00t_sensitivity_probe.py --selftest    # P0-3 sum 断言（2·ref → 1.0）+ 离线 helpers
python scripts/tools/gr00t_metric_audit.py --selftest
python scripts/tools/gr00t_baselines.py --mode selftest       # 字节匹配随机 mask 断言
python scripts/tools/gr00t_topk_scorer.py --selftest          # 物化 + select_final 平局集
python scripts/tools/calibrate_atm_perstep_gr00t.py --selftest  # plan-aware 三态 + α/β 拆分 CV
```

## 产物处置

- `checkpoints/packs/gr00t/deprecated_v1.2/`：旧的 12 个 JSON（4×sensitivity + 4×quant_plan + 4×atm_alpha_beta）
  因 P0-1/P0-2 污染**全部作废**，仅留档。任何新计划必须从修复后的 probe 重跑生成。
- 旧实验报告 `docs/quantvla_v2_gr00t_experiment_report.md` 的数字基于被污染的敏感度数据，
  不再作为 v1.3 方法的证据；正式实验需按设计文档 §6.6 的 Phase 1 流程重跑。

## 第二轮审查（2f4ec47 复核）回应

| # | 审查项 | 状态 |
|---|---|---|
| 1 | TopK scorer 未完成 A8 校准（用 provisional scale 打分） | ✅ 每个 TopK plan 加载后先 `ensure_a8_calibrated()`（共享固定 buffer，`all_calibrated` 强校验）再跑 D_solver |
| 2 | TopK scorer 输出嵌套 final_plan，loader 读不到顶层 layers | ✅ 拆分为 `<out>.report.json` + `<out>.final_plan.json`（后者顶层即 `{packdirs, layers, meta}`，可直接 `GR00T_DUQUANT_PLAN=` 部署） |
| 3 | 真实 LIBERO 服务未闭环 A8 校准（在线用 test obs 校准） | ✅ `inference_service.py` 在启动 server 前执行 `_maybe_close_a8_calibration()`：FP16/动态模式 no-op；静态模式用固定合成 buffer（seed 0，sha256 固定）完成校准；支持 `GR00T_DUQUANT_ACT_SCALE_PATH` 持久化（`save_act_scales`/`load_act_scales`） |
| 4 | `--act-dynamic` 下 `all_calibrated()==False (0/0)` 被拦死 | ✅ 新增 `static_calibrators_required()`；`ensure_a8_calibrated(act_dynamic=...)` 在动态模式下只校验 wrapped 层数、不检查静态校准；calibrator 传入 `act_dynamic=args.act_dynamic` |
| 5 | baseline 每个 plan 用不同校准 buffer | ✅ stage2/TopK scorer 的 calibration buffer 在循环外生成一次、全部 plan 复用；报告 meta 记录 `calibration_buffer_sha256` |
| 6 | Goal suite 的 data config 不一致（工具 vs 服务） | ✅ `SUITE_DATA_CONFIG` 映射（goal→MeanStd）+ `resolve_data_config()`；probe/audit/baselines/topk_scorer/calibrator 的 `--data-config` 默认改为按 suite 解析 |
| 7 | 编排脚本是旧流程；旧报告未作废 | ✅ `run_v2_gpu_experiment.sh` 重写为 gated 管线（selftests → gate 0 audit → dev probe → selector → baselines → TopK 裁决 → dev LIBERO → freeze → held-out Long/90）；`docs/quantvla_v2_gr00t_experiment_report.md` 顶部加 INVALIDATED banner |

## 第三轮审查（3069a67 复核）回应

| # | 审查项 | 状态 |
|---|---|---|
| 1 | `load_act_scales` 后 calibrator 未满 → 第二次启动 0/0 误报退出 | ✅ 加载时标记 calibrator full；新增 `static_scales_ready()` 作为加载态判据；round-trip 回归（A 校准→保存→B 加载→无 warmup 输出一致 + sidecar 不匹配报错） |
| 2 | "固定 buffer" 不固定（全局 torch RNG / 调用者 RNG 已消耗 / hash 只含 noise） | ✅ `fixed_calibration_buffer(seed, ...)` 自包含（np_rng + 本地 torch.Generator）；sha256 覆盖 obs（图像/state/指令）+ noise + 形状/fmt 元数据；probe/audit/baselines/topk/calibrator/server 全部改用它 |
| 3 | ATM/OHB 校准与部署 A8 scale 不一致 | ✅ calibrator 增加 `--act-scale-path`，在**与部署相同的 frozen A8 scale** 上采集 α/β；baselines/topk_scorer/server 同样支持；scale 文件带 sidecar 元数据（plan/checkpoint/buffer/data_config/act_pct/calib/denoising）并校验 |
| 4（清单5） | padding mask + 模态分层采样未实现 | ✅ `pool_samples_stratified`：vision 前块 / text 后块各自 cap max_tokens/2（stride 子采样）+ 后块零行 padding mask；probe 三个采集点全接入；定位为 position-stratified 近似（vision-prefix layout） |
| 5（清单6） | gate 0 只是路牌 | ✅ 新增 `gr00t_gate0_check.py` 硬门（finite=1.0 / W2-W8≥2× / 护栏点火率≥50% / 种子 Jaccard≥0.7 / Spearman≥0.2），编排失败即中止 |
| 6（清单7） | 三套件无共识 plan | ✅ 新增 `gr00t_consensus_plan.py`：Jaccard≥0.7 硬门 + 多数投票 + 预算修复 + 冻结统一 plan（Long/90 zero-shot） |
| 7（清单8/9） | LIBERO 只是打印命令；TopK 只有点估计+固定 5% | ✅ 编排脚本真实托管 server 生命周期并跑 v2/uniform W6/random best·median·worst 同 seed（eval 客户端新增 `--seed`）；TopK n-obs=16 + paired bootstrap（best vs runner-up 的 p 值）报告 |
| 8（清单10） | ATM/OHB 核心实验必须关 | ✅ 编排脚本默认 `GR00T_ATM_ENABLE=0 GR00T_OHB_ENABLE=0`（核心方法对比全程无校正） |
| 9 | 论文定位（完整版 vs 现状） | ✅ 设计文档新增 §9「论文定位与主张边界」：可主张 6 项 / 暂不主张 6 项 + 建议标题方向 |

## 第四轮（测试前两项）回应

| # | 审查项 | 状态 |
|---|---|---|
| 1 | CKA stratified pooling 的 ref/q token index 必须配对（correctness） | ✅ 重构为 paired-blocks：padding 零行 mask 只在 **ref 侧**计算并存入 bank（`accumulate_ref_blocks`），q 侧用**同一 `_row_keep` 掩码**（`evaluate_blocks`）——量化把 padding 行从近零推过阈值也不会再让两侧行集失配；行数不一致直接 raise；回归：ref 零行/q 该行漂移为 1.0 → CKA=1、CS=0；`split_blocks` 为纯位置确定性拆分 |
| 2 | consensus 必须吃 `*_adjudicated.final_plan.json`，不能绕过 TopK | ✅ consensus 强制校验输入 meta.adjudicated==true，否则拒绝运行；编排脚本 consensus 步骤改用各套件裁决后的 `.final_plan.json` |
| 3（其余，正式比较前修） | Gate 0 多 seed 聚合 | ✅ `gr00t_gate0_check` 改为跨全部 seed 取**最差值**（min of rates/ratios）——一个坏 seed 即 fail |
| 4 | baseline report 路径 | ✅ 编排脚本 stage2 用 `--out` 指定 `baselines_${S}_dsolver.json`，dev_accept 读取同路径（已核对一致） |
| 5 | 日志隔离 | ✅ 冒烟/编排输出到 `/tmp/logs/v2_smoke`、`/tmp/logs/v2_gpu`，server 按 plan 命名日志 |
| 6 | 真正的 policy-noise 配对 | ⏳ 待正式比较前核查：probe 已用 `action_noise=` 配对注入；如存在模型内部随机源需在下一步 audit 中定位 |

## 仍需 GPU 验证（执行层，非代码正确性）

1. 全模型 bit 扫描顺序不变性（[2,4,8] vs [8,4,2] 的 b4 一致）——layer 级已由 CPU selftest 覆盖；
2. probe/audit/baselines/topk_scorer 的 GPU 实跑（A8 校准 32 batch 完成性断言会在实跑中生效）；
3. gate 0 指标审计 + W2 护栏点火；八步 TopK 裁决；基线矩阵两阶段评测。

状态标签：**v1.3 P0-corrected scaffolding，correctness gate 已过 CPU 级回归，GPU 实跑待执行。**
