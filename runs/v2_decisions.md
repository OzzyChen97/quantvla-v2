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
