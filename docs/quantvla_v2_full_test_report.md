# QuantVLA v2（v1.3）GR00T N1.5 LIBERO Full Test 报告（草稿骨架）

> 状态：实验进行中（2026-08-15 启动）；dev 三套件 LIBERO 对照在 GPU 7/3/1 并行运行
> 方法：动作加权 CS 表征保持型 W4/FP16 二元选层（v1.3，P0 修复后代码）
> 决策档案：`runs/v2_decisions.md`（D-001…D-009）
> 本报告仅收录 P0 修复后、通过 gate-0 的实验数据；旧报告已作废（见
>   `docs/quantvla_v2_gr00t_experiment_report.md` 顶部 banner）。

## 1. 方法配置（冻结）

- bit 候选 {W4, FP16}，min_bits=4；预算 = uniform-W6 静态字节（862.9MB）
- 主代理 CS（λ_cs=1，λ_cka=0，D-001/D-006 证据）；护栏 τ_rms/τ_sat 硬约束
- w_i：单层干预 paired d_solver（8 obs × 2 噪声），winsorize [0.5, 2.0]
- 二元求解：贪心 + 0-1 背包 + swap 变异多样性 TopK → D_solver 裁决
- 静态 A8：固定 buffer（seed 0）满 32 batch 冻结；skip = 真实未包装 FP16
- 核心方法实验 ATM/OHB 全关

## 2. Gate-0 结果（三套件）

| 套件 | finite | W2/W8 | 护栏点火 | 种子 Jaccard | Spearman(CS,d_solver)@b4（3 seed） |
|---|---|---|---|---|---|
| spatial | 1.0 | 30.5× | 0.60 | 0.917 | 0.41 |
| goal | 1.0 | 27.1× | 0.77 | 0.917 | 0.31 |
| object | 1.0 | 29.2× | 0.73 | 1.000 | 0.30 |

## 3. 配置级 D_solver（同协议 8 obs，同 buffer）

| 套件 | v2 裁决 plan | uniform W6 | uniform W4 | random median | v2/W6 |
|---|---|---|---|---|---|
| spatial | 0.00079 | 0.00374 | 0.0154 | 0.0174 | 4.7× |
| goal | 0.00550 | 0.03241 | 0.0804 | 0.0884 | 5.9× |
| object | 0.00142 | 0.01210 | 0.0468 | 0.0329 | 8.5× |

## 4. LIBERO 成功率对照（进行中）

（待 dev-accept 完成后填入：v2 / uniform W6 / uniform W4 / random best/median/worst
各 50 rollouts，task-level 报告。）

## 5. Held-out（libero_10，spatial plan zero-shot，3 种子）

（待运行。）

## 6. 发现与结论

- D-008：三套件 FP16 mask Jaccard 0.39–0.50 → 层敏感度为 checkpoint 特异。
- （待 LIBERO 完成后补充。）
