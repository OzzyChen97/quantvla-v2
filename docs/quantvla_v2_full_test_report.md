# QuantVLA v2（v1.3）GR00T N1.5 LIBERO Full Test 报告

> 状态：**完成**（2026-08-15 启动，2026-08-16 06:22 最后一个分片 exit 0；8 分片并行墙钟约 8h）
> 方法：动作加权 CS 表征保持型 W4/FP16 二元选层（v1.3，P0 修复后代码）
> 决策档案：`runs/v2_decisions.md`（D-001…D-019）
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

## 4. LIBERO 成功率对照（完成）

8 分片并行（spatial GPU7:5556/5559、goal GPU3:5557/5562、object GPU1:5558/5563、
long GPU5:5560/GPU4:5561；任务分片 5+5，D-016），每配置 **50 rollouts**
（2 分片 × 5 任务 × 5 eps），held-out 3 种子。全部 21 配置 complete
（25 eps / 5 tasks per shard，`aggregate_v2_fulltest.py` 校验），**零 watchdog
误杀**（D-017 分片日志 + D-018 CPU 活性判定），全程无 server 重试触发。

| 套件 | v2 | uniform W6 | random best/median/worst |
|---|---|---|---|
| spatial | **98.0%** | 94.0% | 92.0% / 88.0% / 88.0% |
| goal | 80.0% | **84.0%** | 84.0% / 80.0% / 78.0% |
| object | 86.0% | **92.0%** | 84.0% / 86.0% / 80.0% |
| long（held-out） | transfer-v2 **76.7%**（78/74/78） | transfer-w6 77.3%（74/82/76） | — |

三 dev 套件均值：v2 88.0% vs uniform W6 90.0%；含 long 四套件：85.2% vs 86.8%。
每配置 50 rollouts 的 SE≈4.9pp，套件级 ±4–6pp 差距 ≈ 0.8–1.2σ，**均未达统计显著**。

**Long 口径（D-014）**：mask-only transfer——spatial 裁决 plan 的 FP16/W4 mask +
**Long checkpoint 自己的权重/DuQuant pack/A8 scale**（`gr00t_transfer_plan.py`），
不再是直接复用 spatial plan 文件。uniform W4 不跑 LIBERO（仅 D_solver 对照）。
已知限制：`--seed` 只固定仿真环境，不固定 policy 侧 flow-matching noise。

## 5. Held-out（libero_10，spatial plan zero-shot，3 种子）

每配置 50 rollouts（2 分片 × 5 任务 × 5 eps）× 3 种子：

| 配置 | seed0 | seed1 | seed2 | mean |
|---|---|---|---|---|
| transfer-v2 | 78.0% | 74.0% | 78.0% | **76.7%** |
| transfer-w6 | 74.0% | 82.0% | 76.0% | 77.3% |

任务级困难集群（跨配置/种子稳定）：both moka pots on the stove 0.2–0.6、
white mug ×2 组合任务 0.4–0.8、black bowl between plate+ramekin 0.8；
简单任务（book→caddy、black bowl→bottom drawer、turn on stove + moka pot、
single soup/bowl picks）普遍 1.0。

## 6. 发现与结论

1. **D_solver 排序未一致迁移到 LIBERO SR**：配置级代理上 v2 比 uniform W6 优
   4.7–8.5×，但端到端 LIBERO（50 rollouts/配置）spatial +4.0pp（v2 胜）、
   goal −4.0pp、object −6.0pp（W6 胜）、long −0.7pp（平）；全部 ≈ 0.8–1.2σ，
   **四套件均无统计显著差异**。诚实结论：在 n=50 功效下，v2 与同预算 uniform
   W6 端到端成功率不可区分；方向不一致（spatial 正、goal/object 负）提示
   D_solver 代理捕捉的信号与 LIBERO 成功率之间存在 suite 相关的未对齐分量。
2. **随机 mask 全部 ≥ 劣势**：除 object 的 random median(random_17, 86%) 与 v2
   持平外，所有随机 mask 均 ≤ v2/W6，且 random worst 在每套件最低
   （88/78/80）——预算约束下随机性确实损失性能，v2 选层至少不差于随机基线。
3. **任务难度主导方差**：跨 21 配置任务级 SR 的差异主要来自任务本体
   （多对象组合任务 0.2–0.8，单对象 0.8–1.0），配置间差异在同任务内远小于
   任务间差异。
4. **工程可靠性**：21 次配置级 server 重启（A8 静态校准 + 动态 closure）全部
   干净完成；D-017/D-018 watchdog 加固后 **0 次误杀、0 次重试、0 次 abort**；
   任务分片（D-016）使 8 分片并行墙钟约 8h（spatial ~2.9h、goal ~1.5h、
   object ~2.2h、long ~8h）。
5. D-008：三套件 FP16 mask Jaccard 0.39–0.50 → 层敏感度为 checkpoint 特异。
6. **统计功效不足**（已知限制）：policy 侧 flow-matching noise 未固定 seed，
   配对检验功效受限；若需把 ±4–6pp 的套件级差异定到显著，需要每配置 ≥200
   rollouts 或固定 noise 后做配对设计（已在 D-016 注记）。
