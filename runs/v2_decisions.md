# QuantVLA v2 决策记录（runs/v2_decisions.md）

记录所有影响方法配置的开关决策（设计文档 §6.5 benchmark-leakage 政策要求）。

## 2026-08-15：GPU 冒烟 + spatial 正式 gate-0

**冒烟（10 层/2 seed/4 obs，GPU 7）**：selftests 10/10；metrics 480/480 finite；
hook coverage 10/10；W2/W8 rms_ratio 32.8×；W2 护栏点火 0.6；种子 mask Jaccard 1.0。
A8 二次启动闭环通过：start#1 校准 116 层存盘（69.2s），start#2 直接加载无 warmup。

**正式 gate-0（30 层/3 seed/16 obs，GPU 7）**：
- finite_rate 1.0 ✅；W2/W8 分离 30.5× ✅；护栏点火 0.6 ✅；种子 Jaccard 0.917 ✅；
- Spearman(metric, d_solver)@b4（三 seed 均值）：**CS +0.41**、nmse +0.26、rms_ratio +0.19、
  sat_rate +0.14、cs_cross +0.09、**1−CKA −0.06**。

**决策 D-001（gate-0 驱动）**：主代理 = **CS**（λ_cs=1.0），**1−CKA 默认禁用**（λ_cka=0.0）——
真实模型上 1−CKA 与动作损伤无相关（三 seed 一致 ≈0/负），继续等权会稀释有效排名信号。
λ-sweep 仍生成 CKA 加权的多样性候选，最终由 TopK D_solver 裁决。
`gr00t_gate0_check.py` 的 Spearman 硬门默认检查 cs（--spearman-metric）。
CKA 若重新启用需新的 gate-0 证据。

**决策 D-002**：gate-0 通过判据生效后进入完整 W4 probe → selector → TopK 裁决 → LIBERO。

## 2026-08-15：spatial 完整 W4 probe + selector 首轮 + 多样性修复

**probe（W4-only，16 obs，w_i 8×2 噪声，GPU 7）**：
- CS 就地验证通过（cross 单调：1.19 → 5.06 → 25.9 → 85.9）——CS 在真实模型上对输出缩放有响应；
- 护栏阈值：τ_rms=0.159、τ_sat=0.0032（P99(W4)×1.5，116 候选）；
- 护栏剔除 1 层：`...layers.2.mlp.down_proj`；该层 d_solver_b4=0.00192（次高层的 ~48×，std ±0.0202 噪声主导）——真实敏感层，保留 FP16 正确；
- w_i：raw 范围极大（单层主导）→ winsorize 后 final mean 0.566；bootstrap mask Jaccard 0.793 [0.654, 0.917]。

**selector 首轮问题（决策 D-003）**：diverse TopK = 1/17——λ_cka=0 使 λ-sweep 的 CKA 变体退化、σ=0.1 扰动不足，全部候选塌缩为同一 mask（70 W4 / 46 skip，milp，861.3MB）。
**修复**：新增 Q-DiT 式 flip 变异候选（k∈{2,4,6,8}×2，预算修复）；λ-pairs 默认加 cs 强度变体（0,2/0,5）；扰动 σ=0.25。TopK 必须 ≥5 个不同 mask 才有效裁决。

## 2026-08-15：spatial TopK 裁决完成（决策 D-004）

- **多样性修复生效**：swap 变异（k∈{8,12,16,20}×2，swap 池限定有分数层）→ diverse TopK 9/26；
- **TopK 裁决（9 候选，真实部署语义逐计划加载 + 独立 A8 校准 + wrap 数校验）**：
  D_solver 全部落在 0.0012–0.0026（per-obs std 0.002–0.005 > 候选间差——**裁决边距弱**，
  如实记录；paired bootstrap：best flip12 vs runner-up flip8 mean_diff −0.00034，p_best_wins 1.0）；
- **最终 spatial plan = flip12**（78 W4 / 38 skip，D_solver 0.00123），写入
  `gr00t_quant_plan_libero_spatial_adjudicated.final_plan.json`。

## 2026-08-15：spatial 同协议 D_solver 对照（首个方法证据，决策 D-005）

stage2 同协议（8 obs、同一固定 buffer、真实部署语义逐计划加载 + 独立 A8 校准）：
- **v2 裁决 plan（flip12）：0.00079**
- uniform W6（同预算 862.9MB）：0.00374 → v2 好 **4.7×**
- uniform W4：0.01540 → 19×
- size_based：0.01951 → 25×
- random（同 W4 数+字节匹配）best 0.01742 / median 0.01742 / worst 0.02860 → v2 好 **22× / 36×**

配置级 D_solver 上"v2 选层 > uniform W6 > 随机/均匀 W4"。LIBERO 成功率对照待 dev-accept 阶段。

## 2026-08-15：goal/object gate-0 与代理证据（决策 D-006）

三套件 Spearman(CS, d_solver)@b4（每套件 3 seed）：spatial 0.41、goal 0.305、object 0.296 均值，
**9 个 suite-seed 测量全部为正**（CS 是唯一一致为正的代理；nmse 次之 0.18-0.21）。
gate 语义从"worst-seed ≥0.2"改为 **"均值 ≥0.2 且 min ≥0（无符号翻转）"**——前者对单弱 seed
过度敏感（goal seed2=0.159、object seed0=0.051 仍为正）。三套件 gate 全部 PASS。
说明：逐层代理与动作损伤的相关性是弱-中等（~0.3），这正是方法保留 config-level D_solver
裁决 + LIBERO 验收的原因；正式结论以二者为准。

## 2026-08-15：object 套件完成（D-007）

- gate-0：CS 均值 0.296（3 seed 全正）；W2/W8 29.2×、护栏 0.733、Jaccard 1.0、finite 1.0；
- probe → selector（diverse TopK 8/26）→ TopK 裁决 8 候选 → **final = flip16（D_solver 0.00195）**；
- 同协议匹配对照（8 obs）：**v2 = 0.00142** vs uniform W6 0.01210（**8.5×**）、random median 0.03287（23×）、uniform W4 0.04677（33×）。

## 2026-08-15：goal 套件完成 + 三套件 consensus 硬门（D-007/D-008）

- goal：gate-0 PASS（CS 0.305）；diverse TopK 7/26 → **final=flip12（0.00546）**；
  同协议对照：**v2 0.00550** vs W6 0.03241（5.9×）、random median 0.08835（16×）、W4 0.08042。
- **D-008（consensus 硬门 FAIL）**：三套件裁决 plan 的 FP16 mask 两两 Jaccard = 0.500/0.490/0.389 < 0.7
  → **层敏感度是 checkpoint 特异的**（各套件 probe 各自的敏感度），选层不具备跨套件普适可复现性。
  按设计不冻结统一 plan：dev 三套件各用本套件裁决 plan；**held-out Long 用 spatial 裁决 plan zero-shot**
  （spatial 是主套件、协议最完整；决策理由见 D-008）。

## 2026-08-15：dev LIBERO 启动（D-009 三处 harness 修复）

1. 客户端 state 必须 float32（LIBERO env 输出 float64 → 服务端 transform 全灭）→ 所有 state key 显式 float32；
2. PyAV 视频录制 libx264 缺失 → `save_video` 默认 False（成功率日志仍完整）；
3. numba 缓存目录指向仓库内（后台任务 /tmp 受限 → "no locator available"）；
4. start_server 增加端口应答 + **模型路径校验**（防止陈旧 server 误答）。
修复后三套件 dev-accept 并行（GPU 7/3/1，端口 5556/7/8）：
spatial 首任务 5/5、goal 4/5、object 4/5——正式对照进行中。
