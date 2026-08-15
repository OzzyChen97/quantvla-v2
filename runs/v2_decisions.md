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
