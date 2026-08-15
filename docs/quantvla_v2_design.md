# QuantVLA v2 设计文档（v1.3 收敛修订版）

## 动作加权 W4/FP16 二元选层 + 可行性护栏 + 部署期尺度校正

> 状态：设计修订 v1.3（2026-08-15，第三轮评审·范围收敛修订；修订记录见文末附录）
> 范围：GR00T N1.5 与 pi0.5（openpi）两条量化栈；**实施顺序：先 GR00T（§6 详细方案），后 pi0.5 镜像**
> 相关论文（本地副本见 `docs/paper/`）：
> - ICLR 2025: *Improving Language Model Distillation Through Hidden State Matching*（CKA）
> - ICLR 2026: *Distributional Vision Language Alignment by Cauchy-Schwarz Divergence*（CS-Aligner）
> - CVPR 2025: *Q-DiT: Accurate Post-Training Quantization for Diffusion Transformers*
> - QuantVLA（本仓库 v1 基线）: CVPR 2026, arXiv:2602.20309

---

## 0. 一句话总结

把 QuantVLA 从 **"全局统一 W4A8 + 静态尺度校正"** 升级为 **"动作加权的 W4/FP16 二元逐层选层：用 CKA/CS 相似度做代理打分，用幅度/饱和硬护栏保证可行性，在静态权重存储预算内决定哪些层压 W4、哪些层保留 FP16；核心选层不依赖尺度校正，最终部署版叠加 plan-specific 静态 ATM/OHB"** 的训练无关（training-free）量化框架。

说人话：现在是"所有层一视同仁地压到 4bit，再打几个补丁"；v2 要变成"先量出每一层压到 4bit 的代价（几何/分布/幅度/饱和各打一个分），按'哪层对动作影响大'加权，把字节预算花在最划算的层上——压不动的层干脆保留 FP16，最后整机验证一次动作还像不像"。

**v1.3 与 v1.2 的区别（本轮实验驱动）**：第一轮 LIBERO 实测（`docs/quantvla_v2_gr00t_experiment_report.md`）表明：bit 档 {4,6,8} 之间信号基本平坦（四个 plan 中 W6/W8 合计只被选 0–1 层）、W2 计划直接崩（spatial 0%）、CKA/CS 对 W2 的输出幅度爆炸（±8.1 vs W8 ±1.3）失明。因此 v1.3 把方法**收缩为二元 W4/FP16 选层**，把幅度/饱和硬护栏提为一级组件，把 ATM/OHB 明确移出核心方法（只作为部署期校正），并补上 CS 自测、w_i 记录规范与同预算基线矩阵。

---

## 1. 背景与动机：为什么要做这个升级

### 1.1 v1（现在的 QuantVLA）做了什么

v1 已经在仓库里完整实现，由三个组件组成：

| 组件 | 作用 | 代码落点 |
|---|---|---|
| 选择性量化布局 | 用正则表达式选出要量化的层（pi0.5 是 180 层，GR00T 是 116 层），统一 DuQuant W4A8 | `OPENPI_DUQUANT_INCLUDE/EXCLUDE`（pi05）、`GR00T_DUQUANT_*`（GR00T） |
| ATM（attention temperature matching） | 每个注意力头乘一个 α，α = FP16 与量化版 attention logits **标准差**的比值，夹在 [0.7, 1.4]（对齐二阶统计量 std，不匹配均值） | `code/pi05/openpi/src/openpi/quant/atm_pi05.py`、`code/gr00t/atm/dit_atm.py` |
| OHB（output head balancing） | 论文 v1 定义为 **per-layer residual 接口标量 β**（对齐输出 RMS）；本仓库扩展了 **per-head β**（`beta_perhead`），无 per-head 表时回退标量——这是本仓库的定制扩展，不是原论文定义 | 同上 |

v1 的成绩（`docs/quantvla_libero_results.md`、`code/pi05/results.md`）：

| 模型 | W4A8 平均成功率 | FP16 平均成功率 | 最弱点 |
|---|---|---|---|
| pi0.5 | 95.0–95.5% | 95.5–96.0% | Long 套件 88%（论文 96%） |
| GR00T N1.5 | 86–87% | 89.5% | Long 套件 70%（FP16 80%） |

量化整体损失不大，这说明 v1 的底盘（DuQuant 算子、ATM/OHB）是可靠的——**v2 不是推倒重来，而是在这个底盘上换一个更聪明的"决策大脑"**。

### 1.2 v1 的三个痛点

1. **选层靠"拍脑袋"**：哪些层量化、哪些层跳过，是手工正则表达式定的（如"DiT 注意力层一律保留 FP"）。没有任何度量数据支撑"这一层该不该压、压到几 bit"。`scripts/scan_linear_layers.py` 里的推荐策略是硬编码规则。
2. **只对齐二阶统计量，不对齐相似度**：ATM 对齐 std、OHB 对齐 RMS，都不涉及分布形状与表示几何。两个分布可以有相同的 std/RMS 而形状完全不同；两个表示可以尺度完全一样而几何结构已经扭曲。v1 没有度量"压完之后，这一层输出的**形状（CKA）/分布（CS 散度）/幅度（RMS/amax）/饱和（A8）/最终动作**还像不像原来的"。
3. **静态 + 全局**：所有层同样的 bit、同样的旋转块；α/β 不随去噪步变化；选层指标与最终任务（动作成功率）不闭环。

### 1.3 v2 的三个目标（v1.3 收敛版）

1. **稳定的动作加权 W4/FP16 二元选层**：固定 bit 候选 `{W4, skip=FP16}`、静态 A8、DuQuant g=64 下，用 CKA/CS 代理 + 幅度/饱和硬护栏 + 动作权重 w_i，在同一静态权存预算内选出"哪些层 W4、哪些层 FP16"，且结果**可复现（跨校准种子稳定）、显著优于随机选层**。
2. **同预算对照验证**：与 uniform W6（同字节预算）、uniform W4、随机 mask、按层大小选层、v1 手工 mask 对比，证明"层敏感性排序"本身有价值，而不是搜索器只会把字节挪给大层。
3. **部署级尺度校正（与核心方法解耦）**：核心选层方法**不依赖 ATM/OHB**；最终部署 = v2 plan + plan-specific 静态 ATM/OHB。per-step ATM/OHB 与动态激活量化是第三阶段扩展，不是第一版方法成立的前提。

#### 1.3.1 v1.3 范围收缩（实验事实驱动，不是"降级"）

| 项 | v1.3 固定值 | 依据 |
|---|---|---|
| bit 候选 | `{W4, skip=FP16}` 二元 | 四个套件 plan 中 W6/W8 合计仅 0–1 层被选（bit 维信号平坦）；真正的活跃决策是"哪些层 W4、哪些层 FP16" |
| `min_bits = 4` | **正式安全约束**（不是临时补丁） | 首版计划 75 层 W2 → spatial 四个任务 0%；W2 输出范围 ±8.1 vs W8 ±1.3；CKA/CS 对此失明（§5.1.3 护栏由此引入） |
| 激活量化 | 静态 A8 | dynamic A8 为第三阶段扩展 |
| 旋转块 | g=64 | block ∈ {128,256} 搜索为后续扩展 |
| probe 期 ATM/OHB | 关闭 | 敏感度表对应"未做尺度校正的量化管线"（v1.2 已定义） |
| ATM/OHB | plan 冻结后 plan-specific 校准 | 与选层解耦；校准/部署同 act 模式 |
| W2/W3 | 不进入搜索空间 | 重开门槛见下 |

#### 1.3.2 重开 W2/W3 的门槛（六项实验，全部通过才重开）

1. W2A16 是否正常；
2. W2A8 是否出现 clipping；
3. 实际 A8 饱和率；
4. dynamic A8 能否恢复；
5. 输出 RMS/amax 校正后是否恢复；
6. W2 在完整 LIBERO 上是否仍有稳定收益。

其中 2–5 与 §5.1.3 的护栏直接对应：护栏点火测试（§5.1.3）不通过，说明当前度量无法区分"可行"与"爆炸"，W2/W3 一律不重开。

---

## 2. 灵感来源：三篇论文讲了什么（通俗版）

### 2.1 CKA：看一组样本的几何形状（ICLR 2025 Hidden State Matching）

**论文背景**：知识蒸馏里常用 cosine 距离让学生模型对齐教师模型的隐藏状态。cosine 是**逐向量、一一配对**的匹配，要求两模型维度相同，且（对正缩放其实不敏感，但对**正交变换与跨维度**无能为力）。这篇论文改用 **CKA（Centered Kernel Alignment，中心核对齐）**：比较**整组样本**的 Gram 结构，允许维度不同、对正交变换与整体各向同性缩放不变。

**给我们的启发**：
- v2 用 CKA 作为"这层压完之后**几何形状**保住了吗"的分数；
- **注意它的盲区**：CKA 对整体缩放不敏感，所以它必须与幅度/饱和护栏（§5.1.3）与 ATM/OHB 的 scale 校正**并存**，而不是替代；
- 论文给了 mini-batch 估计器（§5.1.1），但**等 batch 有效 token 数、且明确样本构成**时才严格成立（§5.1.1 的注意事项）。

### 2.2 CS 散度：看两个分布重叠多少（ICLR 2026 CS-Aligner）

**论文背景**：CS-Aligner 用 **Cauchy-Schwarz (CS) 散度** 度量两个分布的距离，KDE 估计无需参数化假设。

**给我们的启发与三条必须说清的边界**：
1. **不是"零重叠也有限"**：CS 散度 `D_CS = −log(⟨p,q⟩²/(⟨p,p⟩⟨q,q⟩))` 在交叉项趋近 0 时同样 → +∞（与 KL 一样）；它的优势是 **KDE 平滑核下更易估计**，不是"无重叠时有限"。数值实现用 log-sum-exp 防下溢（代码已如此）。
2. **"对称"要分两层**：公式本身对称；但我们以 FP16 为参照**固定带宽**（σ 从 FP 样本估计），完整的"选带宽+打分"过程是有向的（FP→Q），不能宣称估计器严格对称。作为 FP16-reference 评估这是合理的设计，只是措辞必须准确。
3. **i.i.d. 假设不严格成立**：同一轨迹的多个去噪步、同一图像的多个 token 高度相关，所以这个数值是**启发式分数**，不严格等于论文中的 KDE 散度估计器。

### 2.3 Q-DiT：把"选配置"做成搜索，用任务指标当裁判（CVPR 2025）

**论文背景**：DiT 量化难（空间/时间方差大）。Q-DiT 给两个武器：

1. **granularity allocation（§5.1）**：给每一层搜一个量化分组大小，**目标直接用任务指标（FID）**、预算约束 BitOps、求解器是进化算法（种群→任务指标评估→TopK→交叉/变异）。
2. **sample-wise 动态激活量化（§5.2）**：每步推理按 min-max 现算激活量化参数，避免 per-timestep 静态表。

**给我们的启发与两个诚实的适配说明**：
1. Q-DiT 的 **group size = 沿输入通道的量化分组粒度**（per-group 独立 scale）；我们的 DuQuant `block_size` = **旋转预处理块大小**（scale 始终 per-output-channel）。两者不是同一个变量（§6.1 显存公式注）。v2 借鉴的是"**逐层搜索预处理粒度**"这个思路，而不是把 Q-DiT 的 group 直接搬过来。真正的 per-group 量化尺度（out×ceil(in/g) 个 scale）是第三阶段的扩展选项。
2. Q-DiT 对**每个种群个体**都算 FID。我们的任务级指标（配对 rollout）很贵，所以 v2 用**便宜代理做内循环搜索、任务级指标做 TopK 裁决**的两级结构——这正是 Q-DiT "逐层 MSE 与最终质量不相关"教训的直接后果（§5.3）。

---

## 3. 核心思路（v1.3：二元选层、双参照、护栏与裁决管线闭合）

### 3.1 度量与搜索的语义（先定义清楚，再谈公式）

**双参照协议（v1.3，参照分工无歧义）**：

> **归因参考 R = 量化管线本身，所有目标层 `weight_bits = 0`**（权重走全精度路径；旋转/置换/A8 激活量化保持开启——`weight_bits=0` 触发 `DuQuantLinear.forward` 的全精度分支，已核代码）。`weight_bits=16` **不是**参照（它本身有量化失真），已从候选集中移除。
>
> **部署参考 FP16 = 纯 FP16 模型（无 DuQuant wrapper）**。

| 度量 | 参照 | 语义 |
|---|---|---|
| CKA_i(b)、CS_i(b) | R | **权重量化归因**：隔离"只改第 i 层 weight bit"的几何/分布效应，无上游漂移、无 wrapper 混淆 |
| D_rms、D_sat（§5.1.3 护栏） | **FP16** | **部署可行性**：量化层输出相对纯 FP16 的幅度漂移与下游 A8 饱和 |
| d_solver_i（w_i 排序权重） | R | 纯归因（wrapper 全精度路径近无损，单层干预下与 FP16 参照的差异可忽略；实现统一走 R 以复用缓存） |
| D_solver 全局（TopK 裁决） | **FP16** | **部署语义**：纯 FP16 vs 完整配置，含层间交互与误差累积 |

**主问题（v1.3 二元形式）**：

```
min_x  Σ_i x_i · w_i · S_i(W4)                                  (1) 动作加权代理失真
s.t.   Σ_i [ x_i·C_i^W4 + (1−x_i)·C_i^FP16 ] ≤ Budget          (2) 静态权存预算
       D_i^rms(W4) ≤ τ_rms        ∀ i : x_i = 1               (3a) 幅度硬护栏
       D_i^sat(W4) ≤ τ_sat        ∀ i : x_i = 1               (3b) 饱和硬护栏
       x_i ∈ {0, 1}                                            (4) 二元选择
```

其中 `S_i(W4) = λ_cka·(1−ĈKA_i)_norm + λ_cs·(D̂_CS,i)_norm`（联合 min-max 归一化，见 §5.2）。
**护栏 (3a)(3b) 是硬约束，不是加权惩罚**：不满足的层直接从搜索空间删除——W2 灾难是可行性问题，不该让 CKA/CS/动作分数"投票"把它捞回来。

**候选集与语义（v1.3）**：

| 选择 | 语义 | 成本 C_i | 失真 |
|---|---|---|---|
| `W4` | DuQuant 包装的 4-bit 权重量化 + 静态 A8（g=64） | 见 §6.1 公式 | 实测（vs R 与 FP16） |
| `skip` | **不量化，保留原始 FP16 nn.Linear**（不是剪枝） | 2·d_out·d_in（+bias）字节 | **0**（它就是 FP16 参照本身） |
| `{2,3,6,8}` | **audit-only**：只在指标有效性实验（§6.6.1）中测量，用于验证度量能否区分 bit 档 | — | 不进入主搜索空间 |

`b=16` 从搜索空间移除（被 skip 严格支配）；0-bit 剪枝为第三阶段独立选项（失真必须实测）。

**预算（v1.3 口径修订）**：主预算 = **uniform-W6 的静态权重存储字节数**（GR00T 实测 862.9MB ≈ v1 W4 预算 636.4MB 的 135.6%），与已完成的 v2 实验一致（四个套件 plan 均 ≤ 862.9MB）。同时记录该预算占 **FP16 权存字节的比例**作为可移植口径（pi05 镜像时直接按同一比例换算，不必重算）。v1.2 文档"预算默认 = v1 W4 636.4MB"废止。预算语义不变：**静态权重存储的理论字节**（完美位打包、无对齐/容器开销），**不是**峰值显存、激活显存、临时缓冲、时延或 BitOps。**注意成本不保证随 bit 单调下降**（旋转矩阵开销对小层可能使量化比 FP16 更贵）：搜索器对每个降档动作过滤 `Δbyte ≤ 0`。

**TopK 裁决管线（v1.3 八步，D_solver 闭环强制执行）**：

```
1. selector 生成 20–50 个低代理损失候选 plan（贪心 + MCKP + 扰动近邻 + λ 扫描，§5.3）；
2. 去重，并按 FP16 mask 两两 Hamming 距离 ≥ max(3, 10%·n_layers) 保证多样性；
3. 选 Top10；
4. 对 Top10 跑完整配置配对 rollout → D_solver(c)（必须报告 per-obs std）；
5. 选 Top3；
6. 对 Top3 做 plan-specific 静态 ATM/OHB 校准（§5.4）；
7. 用校正后的 D_solver 决定最终方案（select_final() 规则，见下）；
8. 只对最终 1–2 个配置跑 LIBERO。
```

`select_final()`（v1.2 公式化，沿用）：

```
d_min = min_c D_solver(c)
T     = { c : D_solver(c) ≤ d_min + tol·max(d_min, ε) }，tol = 5%
c*    = argmin_{c ∈ T} L_proxy(c)        （字典序：任务指标优先，代理只打破近似平局）
```

D_solver **只出现在 TopK 裁决与最终验收**，不进入 Σ_i。

### 3.2 通俗类比：给 116 个"层"发"压或不压"的通行证

把模型想象成一支 116 人的接力队。v1 给所有人发同一双鞋（W4A8），再发矫正鞋垫（ATM/OHB）。v2：

1. **体检**：让每个队员**单独**穿一次 4bit 鞋跑一遍（其他人保持原样），测量他的姿势变形（CKA）、轨迹分布偏移（CS）、步幅漂移（D_rms）、踩线次数（D_sat）；再让每人单独穿一次 4bit 鞋看终点偏差多大（w_i）；
2. **发通行证**：总预算（静态权存）有限，按"偏差大 × 重要"的顺序让最扛得住的队员穿 4bit 鞋；**体检不合格（幅度/饱和护栏）的直接失去资格**，不压；
3. **终检**：全队按发好的配置一起跑一次，看整体终点偏差（D_solver）和最终任务成功率——单人体检之和只是估算，整体跑一次才是裁决。

---

## 4. 现状盘点：v2 的起点（代码地图）

### 4.1 两条量化栈（结构同构，改动双份同步）

| 组件 | GR00T 栈 | pi0.5 栈 |
|---|---|---|
| 量化线性层 | `code/gr00t/quantization/duquant_layers.py` | `code/pi05/openpi/src/openpi/quant/duquant_layers.py` |
| 预处理/打包 | `code/gr00t/quantization/duquant_preprocess.py` | `code/pi05/openpi/src/openpi/quant/duquant_preprocess.py` |
| ATM/OHB | `code/gr00t/atm/dit_atm.py` | `code/pi05/openpi/src/openpi/quant/atm_pi05.py` |
| 校准工具 | `scripts/tools/calibrate_atm_*.py` | `code/pi05/openpi/tools/pi05_calibrate_atm_ohb.py`、`pi05_calibrate_duquant.py` |
| 显存模型 | `scripts/tools/calc_gr00t_duquant_memory.py` 等 | 待补 |
| 启动脚本 | `scripts/run_quantvla.sh` | `code/pi05/run_libero_serve_quant.sh` |

### 4.2 现有能力（v2 直接复用的部分）

- **per-layer bit 已支持**：`GR00T_DUQUANT_WBITS="layer.name:4,layer2.name:8"`；v2 的搜索器只需**生成这个 map**，算子零改动。
- **`DuQuantLinear.weight_bits` 是普通属性**，权重缓存 key 含 bits：进程内改属性即触发重算，逐层扫描不用反复重载模型。
- **monkey-patch 采集基建**：`dit_atm.py` 已有运行时回调采集模式，CKA/CS/护栏采集器照抄。
- **配对噪声 rollouts 已支持**：`get_action(..., action_noise=...)` 接受外部噪声，FP16 与量化模型喂同一噪声得到可逐 step 对比的轨迹。

### 4.3 现有约束（v2 必须遵守）

1. **Data-free 校准**：不能使用真实数据集做校准，唯一参照物是 FP16 模型本身。输入只能用合成数据（随机 obs）+ FP16 自生成轨迹（§5.1.5）。
2. 所有分数都是"相同输入下 FP16 vs 量化"的**配对差异**，不是绝对质量。

---

## 5. 方案设计（怎么干，分四层）

### 5.1 度量层：分数怎么算（含测量协议与统计注意事项）

#### 5.1.1 CKA（几何相似度，线性核）

取自 ICLR 2025 论文 Eq.(2) 与 §2.1：

```
对一批 N 个 token：H_FP ∈ R^{N×d},  H_q ∈ R^{N×d}
linear CKA = ‖Σ_TQ‖²_F / ( ‖Σ_TT‖_F · ‖Σ_QQ‖_F )
其中 Σ_TQ = H̃_FPᵀ H̃_q / (N−1)，H̃ 为列中心化；mini-batch 版（B 个 batch 累加协方差）：
ĈKA = ‖ Σ_b Σ_TQb ‖²_F / ( ‖ Σ_b Σ_TTb ‖_F · ‖ Σ_b Σ_QQb ‖_F )
```

要点与注意事项：
- CKA ∈ [0,1]，对正交变换与整体各向同性缩放不变；**对整体缩放不敏感是盲区**（检测不到整体衰减）——这是 W2 幅度爆炸未被 CKA 检出的直接原因，由 §5.1.3 护栏补位；
- **计算对象是 D×D 协方差（隐藏维度尺度）**，不是 N×N 高斯核 Gram——与 CS 散度**不共用矩阵**，只共用样本池化与接口（§2.2 边界的实现说明）；
- **mini-batch 公式在 batch 有效 token 数不同时是"等权加协方差"，不等于全样本 pooled CKA**——实现须固定有效 token 数或按有效样本量加权；
- **token 样本构成（v1.2 记录进 meta 的高风险点，v1.3 提升为 Phase 1 任务）**：当前实现 = 层输出全部 token 展平 + 确定性子采样（stride，cap 1024），**padding 未 mask**、**模态 token 混合**（vision 占绝对多数会主导 LLM 侧分数）、batch 内 token 不足时有效 N 跨层变化。v1.3 起实施：**padding mask + 按模态分层子采样（vision/text/state 各自 cap，如 256/256/全部）**。CKA/CS 仍定位为**排名启发式**，不解释为绝对质量。

#### 5.1.2 CS 散度（分布重叠，未中心化高斯核 KDE）+ 自测电池

取自 CS-Aligner Eq.(7)(8)：

```
D_CS(p; q) = −log[ (∫p·q dω)² / (∫p² dω · ∫q² dω) ]         0 ≤ D_CS ≤ +∞（扩展实数），p=q 时 = 0
D̂_CS = log[(1/M²)ΣΣκ(xᵢ,xⱼ)] + log[(1/N²)ΣΣκ(yᵢ,yⱼ)] − 2·log[(1/MN)ΣΣκ(xᵢ,yⱼ)]
κ = exp(−‖x−y‖²/(2σ²))，σ 由参照样本 median-heuristic 固定，**下限 1e-3**（σ=0 边界已防护）
```

要点与注意事项（与 §2.2 三条边界对应）：
- **计算对象是 N×N 样本 Gram 矩阵**（与 CKA 的 D×D 协方差不同）；
- 完全无重叠时 D_CS → +∞（不是"有限而稳健"）；实现用 log-sum-exp 防下溢；
- **带宽固定于参照样本 → 估计过程有向（参照→干预），不宣称对称**；
- token/step 相关导致 i.i.d. 假设不成立 → 该数值定位为**启发式分数**，非严格的论文 KDE 估计器；
- **跨层不可比性（v1.2 声明）**：各层用各自的 σ_i、维度、有效 N，联合 min-max 只重缩放数值范围、不消除估计器尺度差异——"0.4 的层比 0.2 的层漂移大两倍"不成立，分数只支持层内 bit 间与近邻层的**序**比较。

**CS 自测电池（v1.3 新增，`kernel_scores.py` 必须全部通过；Phase 1 gate 0）**：

现有 selftest（6 断言）已含 `CS(X, 3.7X) > 1e-2` 且通过，说明 W2 失明**不是简单的核实现错误**。怀疑主因是 **within-set 项对冲**：`D̂_CS = log⟨q,q⟩ + log⟨p,p⟩ − 2log⟨p,q⟩`，当 q = cX 时 within-set 项 `⟨q,q⟩` 以 exp(−c²d²/2σ²) 塌缩、与 cross 项 `⟨p,q⟩` 的下降相互抵消，"完整 D̂_CS 对 8X 不敏感"可能是两项对冲而非核算错。因此自测必须**分项断言**：

| # | 输入 | 断言 |
|---|---|---|
| 1 | 缩放阶梯 X, 0.25X, 0.5X, 2X, 4X, 8X | CKA 基本不变；**cross 项 −2log⟨p,q⟩ 随 |log c| 单调递增**（完整 D̂_CS 只报告、不要求严格单调） |
| 2 | 均值漂移 X+δ | cross 项有响应 |
| 3 | 正交旋转 XR | CKA 不变、cross 项有响应（方向敏感） |
| 4 | 少量 outlier 注入 | 分数不因单 token 异常而爆炸 |
| 5 | covariance 改变（同均值同方差） | 有响应 |
| 6 | token permutation | 对 batch 内排列不变 |

**就地验证（probe 内，比单测更接近实况）**：对真实模型某一层输出手工乘 c ∈ {2,4,8} 后走完整 CS 评估路径，实测分数响应。**关闭准则**：cross 项阶梯不单调或就地验证无响应 → selector 关闭 CS（λ_cs=0），并在报告中声明负结果；不得让未验证分数参与预算分配。

#### 5.1.3 幅度与饱和护栏（v1.3 新增，一级组件）

W2 灾难的教训：**几何相似 ⇏ 尺度正常 ⇏ 控制可用**。护栏是可行性硬约束，不进入加权和。

```
D^rms_{i,b} = median_{(o,c)} | log( (rms(H^Q_{i,b,o,c}) + ε) / (rms(H^R_{i,o,c}) + ε) ) |   ε = 1e-6
              （o = 合成 obs，c = 输出通道；H^R 此处为纯 FP16 参照）
D^sat_{i,b} = mean_o [ #{ x ∈ out_i : |x| > q_max · s^{A8}_{i+1} } / #x ]     q_max = 127
              （s^{A8}_{i+1} = 下一层的静态激活量化 scale；下一消费者非量化层时记 n/a）
```

- **参考系**：两者均对**纯 FP16** 测量（部署可行性语义，§3.1 表）；采集可与 CKA/CS 同一 pass 完成（FP16 参照 pass 缓存复用）；
- **护栏是工程启发式，不是精确测量（P0 审查 §九，如实声明）**：D_sat 用单一标量 P99.9 代理下游 A8 范围（真实 A8 是 per-channel、且部分输出并不直接进入下一个量化 Linear）；guard 的量化侧上游仍含 wrapper/A8，深层会混入累计漂移。论文中必须如实表述为"估计的下游饱和代理"，不得宣称精确 clipping 测量；
- **阈值确定**：在指标审计集（§6.6.1）上取全部 W4 候选的 **P99 再加 margin（×1.5）**，写入 sensitivity.json meta 并随表记录——不拍脑袋；
- **点火测试（护栏自身的验收）**：护栏必须在已知坏例（W2 层）上触发。W2 不触发 = 采集位置/阈值/scale 语义有误，修好前不信任护栏（W2/W3 重开门槛 §1.3.2 的前提）；
- **与度量的分工**：CKA = 几何、CS = 分布、D_rms/D_sat = 可行性护栏、D_solver = 最终行为裁决。比"再加一个 λ_scale 塞进加权和"更稳定。

#### 5.1.4 D_solver（去噪轨迹散度，配置级全局量）

**命名（v1.1 起）**：x_k 是**单次 action chunk 内部的 flow-matching 去噪轨迹**，不是机器人环境的长程 rollout。本指标度量 **solver 级误差传播**；它与 Long 套件的关系是**假设**（每个 chunk 的 solver 误差随 chunk 数累积），由 LIBERO 评测验证。

```
配对噪声：对每个合成 obs 固定一个噪声，两个模型各生成一条轨迹
轨迹含 T+1 个状态：k = 0 是初始噪声 x_0，k = T 是最终动作 x_T（v1.2 已消除 off-by-one：
return_trajectory 现在把初始噪声也加入列表，每步 Euler 更新后追加一个状态）
div_k = ‖x_k^ref − x_k^q‖²₂ / (‖x_k^ref‖²₂ + ε)，ε = 1e-8
D_solver = Σ_k w_k·div_k，w_k = γ^{k+1} / Σ_j γ^{j+1}（k=0..T，γ ≈ 1.2）
```

- **索引方向**：k=0（噪声端）权重最小 γ¹，k=T（最终动作）权重最大 γ^{T+1}；
- **权重归一化**：Σ w_k = 1。注意：归一化只让分数"近似"跨步数可比——不同 num_steps 对应不同的时间采样网格与积分路径，不能自动保证 4 步与 8 步可比（当前实验固定 8 步，理论表述保持保守）；
- **单噪声配对只给点对，不给分布**：条件动作分布 p(a|o) 的 CS 散度需要同一 obs 多个噪声样本（第三阶段可选扩展）；
- **TopK 裁决必须报告 per-obs std**（v1.3 新增）：Top10 的 D_solver 差异要与噪声量级对照，防止"选了个噪声内的赢家"。

#### 5.1.5 Data-free 校准输入（三层，声明各自能解决什么）

1. **L1（现状）**：随机噪声 obs（随机图像/state + 任意 prompt）。所有基线都在这层。
2. **L2（FP16 自生成轨迹）**：用 FP16 自己的反过程生成 (state, x_t, t) 三元组。**诚实声明**：这只能让 x_t 落在"该模型在**随机条件输入**下"产生的流轨迹上（缓解 DiT 头输入分布偏差）；**不能**解决随机图像偏离真实视觉流形、图文不匹配、LLM 侧激活偏离等问题（LLM 侧 CKA 信号弱的风险依然存在，§6.8）。
3. **L3（可选）**：用 checkpoint 自带归一化统计量合成输入统计（若认定 stats 属模型产物）。

合成输入下所有分数只用于**排名**，必须归一化（§5.2），最终以 LIBERO rollout 为裁判。

#### 5.1.6 测量协议（与 §3.1 双参照表对应，probe 的具体实现）

- **参照 pass（跑一次，缓存复用）**：R（weight_bits=0 管线）与纯 FP16 各跑一遍，采集各层输出与参照轨迹；
- **CKA/CS**：协议 1（单层干预）——把第 i 层设为 W4、其余层保持 0，跑 n_obs 批量前向，采集第 i 层输出与参照 R 对比。**主 probe 只测 W4**（116 × 1 ≈ 116 次批量前向）；audit 模式（§6.6.1）在 20–30 层子集上测 {2,4,6,8}；
- **D_rms/D_sat**：同一 pass 内采集，vs 纯 FP16 参照（FP16 pass 的输出已缓存）；
- **d_solver_i**：协议 2，只在 W4 上测，干预 vs 参照 R，作为 w_i；采样规范见 §5.2；
- **D_solver 全局**：协议 3，纯 FP16 vs 完整配置（主路径：W4 全配置 + uniform W6 基线；TopK 阶段对 Top10 逐配置测量）。

### 5.2 打分层：归一化、加权与记录规范（v1.3）

```
S_i(W4) = λ_cka·(1−ĈKA_i)_norm + λ_cs·(D̂_CS,i)_norm
w_i = n·d_i / Σ_j d_j；Σ d ≤ 0 → 全 1；缺测取均值；再 winsorize 到 [0.5, 2.0]
```

- **归一化方式已精确定义**：每一项在**全部 (层, 测量点) 组合**上做**一次联合 min-max**（跨层一次）——这样贪心的 ΔS/Δbyte 比值在层间可比；`max = min` 时该项置 0；outlier 对 min-max 的影响记为已知局限（后续可换 rank 归一化）。
- λ 默认（**gate-0 实测驱动，2026-08-15**）：`λ_cs = 1.0`（主代理）、`λ_cka = 0.0`（禁用）——spatial 审计三 seed 的 Spearman(metric, d_solver)@b4：**CS +0.41**、nmse +0.26、rms_ratio +0.19、sat +0.14、cs_cross +0.09、**1−CKA −0.06**。真实模型上 1−CKA 与动作损伤无相关，默认不参与；λ 扫描（§5.3）仍生成 CKA 加权多样性候选供 TopK 裁决。λ 只有相对意义，且依赖上述归一化方式（换归一化须重调）。
- **w_i 采样规范（v1.3）**：`n_rollout_obs ∈ [8,16]`（默认 8，v1.2 的 4 废止）；**每个 obs 2 个配对噪声**；聚合取 (obs, noise) 的中位数（trimmed mean 可选）；记录跨噪声 std（`d_solver_b4_std` 随表）。
- **w_i 三段日志（v1.3 强制，selector 必须输出）**：

```
raw_d_solver_i:        min / max / mean / std
normalized_w_i_before_clip:  min / max / mean
final_w_i:             min / max / mean
```

必须满足 `mean(final_w_i) ≈ 1`、`0.5 ≤ final_w_i ≤ 2`。**实验报告中的 "w_i 范围 0.0002–0.0010" 是 raw d_solver_i（long 套件），不是归一化权重**——v1.3 起不允许只报 raw；否则无法确认动作权重是否真正参与搜索。
- **w_i 稳定性（v1.3）**：对 rollout obs 做 100 次 bootstrap 重采样，报告层排序 Spearman 均值、最终 FP16 mask Jaccard、计划字节数分布——"换一组合成输入就换掉一半 FP16 层"的方法不允许进入下一步。

### 5.3 选择层：二元求解 + 多样性 + TopK 裁决（v1.3）

```
内循环（离线，秒级）：解 §3.1 的二元问题 (1)–(4)，生成 20–50 个候选 plan
  - 贪心：按 w_i·S_i / Δbyte_i（Δbyte = C_i^FP16 − C_i^W4，过滤 ≤0 的伪降档）升序取层，直到预算满；
  - MCKP/0-1 背包精确解（scipy milp）：校验贪心与最优解的差距，报告 gap；
  - 扰动近邻：S_i × (1+δ)，δ ~ N(0, 0.1)，10 个 plan；
  - λ 扫描：(λ_cka, λ_cs) ∈ {(1,1), (1,0.5), (0.5,1), (1,2), (2,1)}；
  - 多样性：候选间 FP16 mask 两两 Hamming 距离 ≥ max(3, 10%·n_layers)——Top10 不允许是同个 mask 改一两层；
外循环（GPU）：对 Top10 跑完整配置配对 rollout → D_solver(c)（含 per-obs std）；
  Top3 → plan-specific 静态 ATM/OHB 校准（§5.4）→ 校正后 D_solver → select_final()（§3.1）；
终审（LIBERO）：协议见 §6.5。
```

- **为什么两级**：Q-DiT 对每个种群个体算 FID（任务指标当目标）；我们的任务指标（配对 rollout）太贵，用便宜可加的层代理做内循环，任务指标做 TopK 裁决。**代理目标不是最优性保证，只是搜索启发式**——这是 Q-DiT "逐层 MSE 与最终质量不相关"教训的直接应用。
- **能力声明（v1.3）**：当前代码仍保留 mixed-bit（W4/W6/W8/skip）形态；**v1.3 主路径 = 二元**（`--binary` 默认开、`--min-bits 4` 固定）。S_i 只对 W4 有测量；block 固定 g=64。
- 预算 `C_i` 与"BitOps/显存"解耦：**只算静态权重存储字节**（§6.1），命名上不再写"显存/BitOps"。

### 5.4 缩放参数层：部署期校正（v1.3 重新定位）

**定位（v1.3，与核心方法解耦）**：

> 核心选层方法**不依赖 ATM/OHB**；最终部署模型 = v2 plan + **plan-specific 静态 ATM/OHB**。论文同时报告"v2 无校正"与"v2 + static ATM/OHB"两个版本。per-step ATM/OHB 与动态激活是第三阶段扩展。

论文/报告的表述层次：

```
Base Quantization（DuQuant W4A8, g=64, 静态 A8）
  + Action-aware layer selection（本方法核心，v1.3 二元 W4/FP16 + 护栏）
  + Static ATM/OHB（plan-specific，部署版）
  + Per-step ATM/OHB / dynamic activation（扩展版，第三阶段）
```

规则（v1.3）：

1. **静态先行**：第一阶段只比较四配置——v2 无校正 / +static ATM / +static OHB / +static ATM+OHB。判断 ATM 单独贡献、OHB 单独贡献、互补性、是否伤害 Goal/Object。静态确认有效后才测 per-step。
2. **plan-aware 校准**：冻结 plan → 校准 → 评测。不能复用 v1 uniform-W4 的 JSON（混合计划改变了哪些层是 FP16、下游 A8 分布）。校准与部署必须同 act 模式（静态 A8）。
3. **全 FP16 attention block 不盲目校正**：block 内相关投影全部 skip 时 α≈1、β≈1 才是正确的。实现：校准器对全 FP16 block 输出 α=1/β=1 或关闭校正；至少检查校准值是否显著偏离 1（|α−1| < 0.02 视为无校正需求，写入 JSON 标记）。否则无损的 FP16 层可能被多余 correction 改坏。
4. **per-step 门槛**：先统计 `CV_t(α_{l,h,t})`、`CV_t(β_{l,h,t})`（跨 denoise step 的变异系数）。若 ≥95% 的 head 的 CV < 5%，静态版本足够，per-step 只增加方差与实现复杂度。
5. **2D 消融**：{static, dynamic} × {ATM off, on} 四格（第三阶段）——dynamic act 改变 attention logits 与 residual 输出统计，static act 下校准的参数不能直接假定适用。
6. **ATM/OHB 不负责修复低 bit 幅度爆炸**：ATM/OHB 作用于注意力与残差接口；一般 Linear 输出直接造成下一层 A8 饱和的问题由 §5.1.3 护栏兜底，不能指望 ATM/OHB 把 W2"抢救回来"。

#### 5.4.1 per-step α/β（查表模式，扩展阶段）

- 三处改动：denoise 循环维护 `_current_denoise_step`（已实现）；`dit_atm._ATMProcessor` 按 step 查表（缺失 key fallback 静态 `all`，已实现）；JSON v2 schema `{layer: {all, beta_perhead, steps: {"<t_discretized>": {all, beta_perhead}}}}`（校准器已实现，`GR00T_ATM_PER_STEP=1`）。
- **key 语义**：key 的**坐标系**（t_discretized ∈ [0,1000)）与 num_steps 无关，但**被访问的 key 集合取决于 num_steps**——8 步校准表用于 10 步推理时，多数位置 fallback 到 `all`。有 fallback 保证不崩，但**不保证跨步数等价**。
- **校准一致性（v1.2）**：per-step 校准器必须在**与部署相同的 act 模式**下运行（`--act-dynamic` 开关与部署的 `GR00T_DUQUANT_ACT_DYNAMIC` 保持一致），否则 α/β 是用静态激活统计校准、去修正动态激活模型，两模块互相干扰而非叠加。
- 等值测试护栏：per-step 表全等于静态 `all` 时，输出与 v1 位级一致。

#### 5.4.2 动态激活 scale（Q-DiT 5.2 式，扩展阶段）

`GR00T_DUQUANT_ACT_DYNAMIC=1`：`_get_act_scale` 每次前向按 min-max 现算（对称：max-abs / qmax），跳过校准器。消融：static（ACT_PCT=99.9）vs dynamic。**不再**引用 Q-DiT 的"50 步静态表 39% 显存"数字做类比（那是 activation 量化参数的开销，与我们的 per-step α/β 表量级不同，类比不成立）。

#### 5.4.3 实现注意事项（已核代码）

- 权重缓存 key 是 `(device, dtype, wbits, row_rot)`——per-step 缩放若 fold 进 dequant scale，key 必须加 step 维度；v2 默认保留运行时乘（不动缓存），fold 作为后续优化；
- per-layer `block_size/block_out_size/act_percentile/ls` 已支持 per-layer cfg 覆盖（`wrap_duquant` 扩展）。

### 5.5 数据流总览

```
合成输入(L1/L2/L3) + 配对噪声
        │
        ▼
┌─────────────────────────────────────────────┐
│ 度量层（GR00T: scripts/tools/                │
│   gr00t_sensitivity_probe.py；pi05 同构）    │
│  参照 pass：R（weight_bits=0 管线）+ 纯 FP16 │
│  协议1 单层干预 → CKA/CS（vs R）             │
│         + D_rms/D_sat 护栏（vs FP16）        │
│  协议2 单层干预+动作 → d_solver_i（w_i）     │
│  协议3 全配置 → D_solver（全局裁决标量）     │
│  产出 sensitivity.json（含护栏阈值 meta）    │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│ 选择层：联合归一化 + w_i 加权 + 护栏过滤     │
│  内循环：贪心/MCKP/扰动/λ扫描 → 20–50 候选   │
│  多样性：FP16 mask Hamming 距离 ≥ 阈值       │
│  产出 quant_plan.json（每层 W4 或 skip）     │
│  + w_i 三段日志 + bootstrap 稳定性报告       │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│ 裁决：Top10 配置 GPU 配对 rollout → D_solver │
│  Top3 → plan-specific 静态 ATM/OHB 校准      │
│  校正后 D_solver → select_final()            │
│ 部署：quant_plan → GR00T_DUQUANT_PLAN env    │
│      + ATM/OHB JSON → serve 脚本             │
└─────────────────────────────────────────────┘
```

---

## 6. GR00T 实现方案（详细设计，先做 GR00T）

### 6.0 为什么先做 GR00T

1. **提升空间更大**：GR00T 量化版 Long 套件 70% vs FP16 80%（差 10 个点），pi05 是 88% vs 94%（差 6 个点）——同样的方法在 GR00T 上能更快看到信号；
2. **工具链更全**：静态字节计算器已有（`scripts/tools/calc_gr00t_duquant_memory.py`）；GR00T 的 DiT（diffusers `Attention` + processor 机制）denoise 循环结构清晰，per-step 钩子落点明确；
3. 方案在 GR00T 验证通过后，pi05 按 §4.1 的同构映射镜像实现（预计 2–3 天）。**pi05 镜像在 Phase 1–2 通过后才启动**。

### 6.1 GR00T 栈速览与关键代码事实（v2 动工前必须知道）

**推理链路**（所有改动的坐标系）：

```
scripts/inference_service.py --server（ZMQ :5556）
  → Gr00tPolicy.get_action            code/gr00t/model/policy.py:147
  → GR00T_N1_5.get_action             code/gr00t/model/gr00t_n1.py:171
  → FlowmatchingActionHead.get_action code/gr00t/model/action_head/flow_matching_action_head.py:353
       :367  actions = torch.randn(...) / action_noise（配对噪声改造点）
       :377  for t in range(num_steps)      ← denoise 循环（per-step 钩子点）
       :379  t_discretized = int(t_cont * num_timestep_buckets)  ← step 表的 key
       :397  self.model(...)                ← DiT 前向（ATM processor 挂这里）
```

**量化/缩放挂载时序**（`policy.py:278–297`，必须保持）：attention patch → duquant → ATM → `.to(device)`；action head 重建（`policy.py:253–276`）在 quant 之前，新代码不得破坏该时序。

**静态权存字节公式**（`calc_gr00t_duquant_memory.py:48–77`，v2 预算底座）：

```
quantized_weights = out·in·wbits/8     （理论紧密打包：3/6-bit 按理想化计，无对齐/容器开销）
weight_scales     = out·2              （FP16 per-output-channel，与 block 无关）
R_in_blocks       = ceil(in/g)·g²·4    （g 只影响旋转矩阵存储）
R_out_blocks      = ceil(out/g)·g²·4   （restore/propagate）
permutation       = in·4（int32）      bias = out·2
```

**语义声明（v1.1）**：这里的 `g` 是 **DuQuant 旋转块大小**，不是 Q-DiT 式 per-group 量化粒度；若做后者，scale 数应为 out×ceil(in/g)（第三阶段扩展）。预算只含静态权存字节，不含激活/峰值显存/时延/BitOps。

**pack 缓存命名**：`{PACKDIR}/{sanitize_name(layer)}.npz`；R 矩阵依赖 (block, ls)，不同 (g, ls) 组合必须用不同 PACKDIR。

**进程内切 bit 已可行**（缓存 key 含 `weight_bits`，改属性自动重算）。

### 6.2 P0-G：度量层（`scripts/tools/gr00t_sensitivity_probe.py`，已实现 v1.2；v1.3 增项见 §6.7）

#### 6.2.1 核度量模块（`code/gr00t/quantization/kernel_scores.py`，已实现）

- `MiniBatchCKA`（实现于 `LayerScoreBank`）：**中心化线性核**，D×D 协方差路径（ICLR 2025 公式）；
- `CSDivergence`（同 bank）：**未中心化高斯核 KDE**，N×N Gram 路径（CS-Aligner 公式），log-sum-exp 防下溢，带宽由 FP 样本 median-heuristic 固定；
- **两者共用样本池化/子采样与 evaluate 接口，不共用矩阵**（一个 D×D、一个 N×N——v1.0 "共用高斯核矩阵"的表述已废弃）；
- **v1.3 增项**：自测电池（§5.1.2 表）必须全部通过；`evaluate` 同时返回 **cross 项** 与完整 D̂_CS，供阶梯断言与 selector 关闭准则使用。

#### 6.2.2 采集挂载点（三处）

| 挂载点 | 采集对象 | 方式 |
|---|---|---|
| 每个 `DuQuantLinear` 输出 | 116 个候选层输出（CKA/CS/护栏同 pass） | `register_forward_hook`（单层干预协议下采集） |
| DiT `Attention` 输出 | attention 层输出 | `register_output_capture`（dit_atm 扩展，带 step） |
| LLM decoder layer 输出 | `backbone.eagle_model.language_model.layers.*` | forward hook |

#### 6.2.3 data-free 校准输入（L1/L2/L3，各自能解决什么——见 §5.1.5 的诚实声明）

#### 6.2.4 配对噪声与 D_solver

`get_action(action_noise=..., return_trajectory=...)`（已实现，默认行为位级不变）。D_solver 公式与索引方向见 §5.1.4（γ^{k+1}、归一化、ε=1e-8）。**v1.3 增项**：全局 D_solver 输出 per-obs std。

#### 6.2.5 扫描流程与成本（v1.3 收敛）

- **协议 1（CKA/CS + 护栏）**：主 probe 只测 **W4**——把第 i 层设为 4、其余保持 0，跑 n_obs 批量前向采集（116 × 1 ≈ 116 次批量前向；CKA/CS vs R、D_rms/D_sat vs FP16 同一 pass）。audit 模式（`--layers-subset`）在 20–30 层子集上测 {2,4,6,8}。
- **协议 2（w_i）**：W4 × 116 层 × n_rollout_obs（默认 8，v1.3）个 obs × 2 个配对噪声；
- **协议 3（D_solver 全局）**：纯 FP16 vs 完整配置（主路径：全 W4 配置 + uniform W6 基线；TopK 阶段对 Top10 逐配置）；
- group 维度固定 g=64；{128,256} 只对 TopK 层补扫（第三阶段）。

#### 6.2.6 输出 schema（v1.3：二元 + 护栏）

```json
{"meta": {"calib_mode": "L1", "token_mix": "all_tokens_flattened_cap1024",
          "num_steps": 8, "n_obs": 16, "n_rollout_obs": 8,
          "ref": "weight_bits=0 pipeline (attribution) + pure FP16 (feasibility)",
          "guard_thresholds": {"tau_rms": 0.12, "tau_sat": 1e-3,
                               "estimation": "P99 of W4 candidates x1.5"}},
 "layers": {
   "backbone...q_proj": {
     "b4": {"cka": 0.99, "cs": 0.21, "cs_cross": 0.19,
            "rms_ratio": 1.03, "amax_ratio": 1.05, "sat_rate": 1e-4},
     "d_solver_b4": 0.032, "d_solver_b4_std": 0.004},
   "attn:action_head...attn1": {"b4": {"cka": ..., "cs": ...}}},
 "global": {"b4": {"d_solver": 0.11, "d_solver_std": 0.02},
            "uniform_w6": {"d_solver": ..., "d_solver_std": ...}}}
```

skip（FP16）**不测量**——它是参照本身；plan 里才出现 skip 选择。

### 6.3 P1-G：打分与选择（`scripts/tools/gr00t_select_plan.py`，已实现 v1.2；v1.3 增项见 §6.7）

- **打分**：联合 min-max 归一化（全部层一次）+ `S_i = λ_cka·(1−cka)_n + λ_cs·cs_n`（§5.2）；`w_i` 从 `d_solver_b4` 归一化（缺测取均值 1）；
- **护栏过滤（v1.3）**：D_rms > τ_rms 或 D_sat > τ_sat 的层从搜索空间删除（硬约束，§3.1）；
- **求解器（v1.3）**：二元贪心（默认）+ scipy milp 0-1 背包（报告 gap）+ 扰动近邻 + λ 扫描 → 20–50 候选；FP16 mask Hamming 多样性过滤 → Top10；`--min-bits 4` 为**正式安全约束**（默认值 4，文档化于 §1.3.1，不再是临时补丁）；
- **w_i 三段日志（v1.3 强制）**：raw / normalized_before_clip / final 的 min/max/mean 写入 plan meta（§5.2）；
- **预算**：默认 = uniform-W6 静态权存字节（862.9MB），同时记录占 FP16 权存比例；`skip` 计 FP16 字节（2P+bias）；
- **产物 `gr00t_quant_plan.json`**（含 meta 语义声明：skip=FP16 旁路、预算=静态权存、目标=代理、护栏已过滤、全局 D_solver 待 GPU 裁决）；`GR00T_DUQUANT_PLAN` env 一键接入（已实现 + 小模型端到端测试通过）。

### 6.4 P2-G：部署期缩放（plan-specific 静态 ATM/OHB，v1.3 重新定位）

见 §5.4（静态先行四配置消融、plan-aware 校准、全 FP16 block 跳过、per-step CV 门槛、2D 消融）。校准器 `scripts/tools/calibrate_atm_perstep_gr00t.py`（data-free、配对噪声；静态/按 t_discretized 分组两种模式），运行方式见 §6.7。**核心方法评测（Phase 1）不需要本组件**。

### 6.5 P3-G：评测协议、基线矩阵与验收（v1.3）

- **协议**（与 v1 一致，可直接对比）：`run_inference_server.sh libero_<suite>`（ZMQ :5556）+ `run_libero_eval.sh libero_<suite> --headless`；denoising=8；10 任务 × 5 trials = 50 条/套件；五套件。
- **统计现实（v1.2）**：50 条下成功率粒度为 2pp，**75% 不可观测**——首里程碑改为 **≥76%（38/50）**；70%→76% 只差 3 个 episode，单轮 50 条不足以做统计结论。
- **统计方案（v1.2，task-clustered）**：
  - 开发冒烟：单轮 50 条（快筛配置，不做结论）；
  - 结论级：每套件 **3 个随机种子 × 50 条**；**150 条不是 i.i.d. Bernoulli**（10 个任务 × 5 次 × 3 种子，同任务失败高度相关、任务难度差异大）——**不能**直接套二项 SE（理想化 SE 在 p=0.75 时 ≈3.5pp、95% 区间约 ±6.9pp，真实 task-clustering 只会更宽）；
  - 报告口径：per-task 成功率 + **task-level cluster bootstrap CI**（按任务重采样，不是按 episode）；配对比较用 **task-level 配对置换检验**（同任务同种子），episode-level McNemar 仅作次级参考；
  - 验收阈值只对结论级数据生效，且"显著提升"必须过上述 task-level 检验。
- **Benchmark 泄漏政策（v1.2）**：LIBERO 若反复用于调 λ、开关注、TopK 选择，就只是 validation。约定：**开发与调参只用 spatial/goal/object 三个套件**；libero_10（Long）与 libero_90 **仅在最终验收时跑**，作为 held-out 口径；所有开关决策记录到 `runs/v2_decisions.md`。
- **跨套件 mask 一致性（v1.3 新增）**：三个开发套件各自出 plan 后，FP16 mask 两两 **Jaccard ≥ 0.7** 才认为选层可复现；最终部署与验收使用统一 plan（取三套件共识或按 §6.6 的种子稳定性规则固定）。

**基线矩阵（v1.3，Phase 1 必须全部跑齐）**：

| 配置 | 目的 |
|---|---|
| FP16 | 确认当前代码基线 |
| uniform W4，无 ATM/OHB | 裸量化基线 |
| v1 W4 + ATM/OHB | 同轮复现原方法 |
| uniform W6 | **与 v2 相同预算（862.9MB）的最关键 baseline** |
| 随机 W4/FP16 mask × 20（两阶段：D_solver 筛出 best/median/worst 3–5 个再上 LIBERO） | **证明敏感度排序优于随机选层**（mask 的 W4 层数与 v2 plan 相同） |
| 按层大小选 W4/FP16（字节贪心：最大的层压 W4） | 排除"搜索器只是偏向大层" |
| v1 手工 mask、同 W4/FP16 数量 | 比较自动选层与人工规则 |
| v2 plan，无 ATM/OHB | 方法本体（核心方法不依赖尺度校正的证据） |
| v2 plan + static ATM/OHB | 建议最终基础版本（Phase 2） |
| v2 plan + per-step ATM/OHB | 扩展版本（Phase 3） |
| v2 plan + dynamic act | 扩展版本（Phase 3） |

**两个必须通过的比较**：

```
(1) v2 plan  vs  random mask（同 W4 层数）   —— 排序有价值
(2) v2 plan  vs  uniform W6（同字节预算）    —— 同预算优于均匀
```

两项均要求结论级 task-level 配对置换检验通过；不过 → 层敏感度选择不能被验证，方法不允许进入论文级评测。

### 6.6 GR00T 路线图：四阶段 + 退出准则（v1.3）

**阶段前 gate 0：指标有效性实验（§6.6.1）必须先做**——分数能否区分 bit、排名是否优于随机、同预算是否优于 uniform，是三个前置问题；不做指标审计直接跑 LIBERO 是浪费。

| 阶段 | 内容 | 退出准则（gate，全部满足才进入下一阶段） |
|---|---|---|
| **Phase 1 稳定选层** | (1) 指标审计（§6.6.1）；(2) CS 自测电池 + 就地验证（不过则 λ_cs=0 并声明负结果）；(3) 护栏采集 + W2 点火测试；(4) w_i 采样规范（8–16 obs × 2 噪声）+ 三段日志 + bootstrap 稳定性；(5) padding mask + 模态分层子采样；(6) 二元选择器（贪心/MCKP/扰动/λ）+ Hamming 多样性 TopK；(7) TopK D_solver GPU 裁决（八步管线）；(8) 基线矩阵 Phase 1 部分 | ① v2 ≥ uniform W6 且 v2 ≥ random mask（task-level 配对检验）；② 跨 3 个合成校准种子 FP16 mask Jaccard ≥ 0.7；③ 护栏在 W2 上点火；④ CS 过自测或被正式关闭 |
| **Phase 2 稳定尺度校正** | Top3 plan 分别做 plan-specific **静态** ATM/OHB；ATM-only / OHB-only / ATM+OHB 消融；全 FP16 block 跳过规则；冻结全部超参数 | 校正后 D_solver 不劣于未校正；消融结论可解释（谁贡献、谁伤害） |
| **Phase 3 扩展** | per-step ATM/OHB（CV 门槛，§5.4）；dynamic A8（2D 消融）；W6/W8 搜索；W3/W2（§1.3.2 六实验门槛）；多预算 Pareto；实测 latency/峰值显存 | 每项独立消融；W2 六实验全过才重开低 bit |
| **Phase 4 论文级评测** | LIBERO-90 冻结为最终验证（held-out）；task-cluster 统计全套；三配置报告（v2 无校正 / v2+static ATM/OHB / v2+扩展） | 论文级结论 |

**消融顺序（报告直接可用）**：v1 W4A8 → +CKA 选层 → +CS 打分（若启用）→ +w_i 重要性加权 → +护栏 → +D_solver 裁决 → +static ATM/OHB →（扩展）per-step α / dynamic act。每步在开发三套件上对比（结论级 3 种子），Long/90 只在验收跑。

#### 6.6.1 指标有效性实验（Phase 1 前置，gate 0）

- **对象**：20–30 个代表性层，覆盖 LLM early/middle/late、q/k/v/o projection、MLP、DiT early/middle/late、action decoder 邻近层；
- **bit**：{8,6,4} 全测 + **W2 压力测**（只作审计，不进搜索空间）；
- **指标**：1−CKA、CS（cross 项单列）、D_rms、D_sat、输出 NMSE、单层 d_solver；
- **报告**：`Spearman(metric, d_solver)` 逐 metric；bit 单调性 `D(W8) ≤ D(W6) ≤ D(W4)`（不要求所有层严格成立）；**W2 vs W8 分离度**——若大量层出现 `D(W2) ≈ D(W8)`，说明采集流程仍不可用于 mixed-bit 搜索（维持二元 + min_bits=4 不变）；
- **种子稳定性**：3 组合成校准 seed 下报告层排序 Spearman、FP16 mask Jaccard、最终计划字节数、TopK 稳定性。

### 6.7 P0-G / P1-G / P2-G 实现状态（2026-08-15：v1.2 已实现；v1.3 增项已实现 + CPU 单测通过，GPU 实跑待做）

**新增/修改文件**（全部只在 GR00T 栈，未触碰 pi0.5）：

| 文件 | 内容 | 验证状态 |
|---|---|---|
| `code/gr00t/quantization/kernel_scores.py`（新） | `LayerScoreBank`：中心化线性 CKA（D×D 协方差）+ 未中心化高斯核 CS（N×N Gram，log-sum-exp、FP 固定带宽） | ✅ selftest 6 项断言（CKA 尺度不变性等） |
| 同上 | **v1.3 增项**：自测电池（缩放阶梯 cross 项单调、均值漂移、旋转、outlier、covariance、permutation）；`evaluate` 返回 cross 项 | ✅ selftest 全过（cross 阶梯严格单调） |
| `code/gr00t/quantization/duquant_layers.py` | per-layer cfg + `GR00T_DUQUANT_PLAN` 接入；`GR00T_DUQUANT_ACT_DYNAMIC` 动态激活 scale | ✅ 小模型端到端 + 动态 scale 精确 ×5 |
| `flow_matching_action_head.py` / `gr00t_n1.py` | `action_noise` + `return_trajectory` + `_current_denoise_step` | ✅ 编译/导入 |
| `code/gr00t/atm/dit_atm.py` | per-step 查表应用（缺失 fallback）；step 采集回调；`compute_per_step_alpha/beta` | ✅ 单测（相同统计→α=β=1；回退三态） |
| `scripts/tools/gr00t_v2_common.py`（新） | 共享工具（obs/env/policy/噪声） | ✅ |
| `scripts/tools/gr00t_sensitivity_probe.py`（新） | 三协议测量（v1.2 参照 = weight_bits=0 管线；单层干预 CKA/CS、w_i 探测 vs 参照 R、全局 D_solver = 纯 FP16 vs 配置）；轨迹 T+1 状态；权重 γ^{k+1} 归一化；基准模式 ATM/OHB 关 + 静态 act（写入 meta） | ✅ transform 管线 CPU 验证；GPU 实跑待做 |
| 同上 | **v1.3 增项**：D_rms/D_sat 护栏采集（vs FP16 参照，同 pass）；主 probe 只测 W4（`--bits` 默认 4）；`--layers-subset` audit 模式；`--n-rollout-obs` 默认 8、每 obs 2 噪声（**各自配对参照轨迹**，中位数聚合）；D_solver per-obs std；CS 就地缩放验证；τ=P99×margin 写入 meta | ✅ 离线单测全过（含 P0-3 sum 断言）；GPU 实跑待做 |
| 同上 | **P0 修复**：A8 校准在 weight_bits=0 参照态下跑满 calib_steps 个真实 batch（calib_steps×batch_size 个 obs）并强校验 `all_calibrated()`；`solver_divergence` 加权和（Σw·div） | ✅ selftest：D_solver(2·ref)=1.0 精确 |
| `scripts/tools/gr00t_select_plan.py`（新） | v1.2 目标：联合归一化 + w_i·S_i（w_i winsorize [0.5,2.0]、Σ=0 回退）；贪心/进化（过滤 Δbyte≤0）；`select_final()` TopK 字典序裁决规则；skip=FP16 语义；`--min-bits`（默认 4，已存在） | ✅ 真实 116 层形状端到端；select_final/权重边界单测通过 |
| 同上 | **v1.3 增项**：二元模式（`--binary` 默认）；护栏硬过滤（τ 自动估计）；scipy milp 0-1 背包（gap 报告）；扰动近邻 + λ 扫描；FP16 mask Hamming 多样性 → Top10；w_i 三段日志 + bootstrap 稳定性输出；预算默认 `uniform-w6` | ✅ selftest 全过（护栏点火、milp gap、diversity、bootstrap） |
| `scripts/tools/gr00t_metric_audit.py`（新） | 指标有效性实验（§6.6.1）：子集层 × {2,4,6,8} × 指标 + Spearman vs d_solver/bit 单调性/W2-vs-W8 分离度/3 种子稳定性；`--layers-subset` stride 子集 | ✅ 离线统计 selftest 通过；GPU 实跑待做 |
| `scripts/tools/gr00t_baselines.py`（新） | 基线生成与两阶段评测：random mask（同 W4 层数 + **字节匹配 ≤0.5%**）×20 → D_solver 分布 → best/median/worst 上 LIBERO；按层大小贪心；v1 手工 mask（qkv 保留 FP16 启发式）；uniform W4/W6 计划；stage2 走 **GR00T_DUQUANT_PLAN 真实部署语义** + `n_applied==expected` 强校验 + A8 校准完成断言 | ✅ 生成器 + 代表选取 selftest 通过；GPU 实跑待做 |
| `scripts/tools/gr00t_topk_scorer.py`（新） | TopK D_solver 裁决（八步管线第 4–7 步）：逐计划物化 GR00T_DUQUANT_PLAN → 真实部署语义加载（wrap 数强校验）→ D_solver → `select_final()` → 最终 plan | ✅ 离线 selftest（物化 + 平局集）；GPU 实跑待做 |
| `code/gr00t/quantization/duquant_layers.py` | **P0 修复**：`transform_weight_for_forward_optimized` 先 clone 再旋转（权重不可变）；`_get_act_scale` 满批前只给临时 scale 不冻结；`all_calibrated()`/`calibration_progress()` 助手 | ✅ CPU 回归：顺序不变性 + 校准生命周期 |
| `scripts/tools/calibrate_atm_perstep_gr00t.py`（新） | data-free ATM/OHB 校准（静态/per-step） | ✅ selftest；GPU 待做 |
| 同上 | **v1.3 增项**：`--plan` plan-aware 校准（全 FP16 block 中性值强制 α=β=1、漂移值保留并标记）；CV_t(α/β) 统计 + `static_sufficient` 判定，sidecar 输出 `<out>.cv_stats.json` | ✅ selftest 全过（plan-aware 三态 + CV 统计） |
| `scripts/run_quantvla.sh` | v2 plan 模式注释；ATM/OHB **opt-in**（默认关，开 ATM 无表即报错） | ✅ |
| `scripts/inference_service.py` | **第二轮审查**：启动服务前闭环静态 A8 校准（固定合成 buffer + `all_calibrated` 强校验；`GR00T_DUQUANT_ACT_SCALE_PATH` 持久化/加载）；FP16 与 dynamic-act 模式 no-op | ✅ |
| `scripts/run_v2_gpu_experiment.sh` | **第二轮审查**：重写为 gated 管线（selftests → gate 0 audit → dev probe → selector → baselines 两阶段 → TopK 裁决 → dev LIBERO → freeze → held-out Long/90） | ✅ |
| `scripts/tools/gr00t_v2_common.py` | `SUITE_DATA_CONFIG`（goal→MeanStd）统一工具与服务坐标系；`fixed_calibration_buffer`（**自包含 seed 纯函数**：本地 torch.Generator + obs/noise/meta 全量 sha256）；`ensure_a8_calibrated`（static/dynamic 双态 + sidecar 校验） | ✅ |
| `code/gr00t/quantization/duquant_layers.py` | **第三轮**：`load_act_scales` 标记 calibrator full（修复"第二次启动 0/0 误报"）；`static_scales_ready()`；`save/load_act_scales` 带 sidecar 元数据（plan/checkpoint/buffer/data_config/act/denoising hash）并可校验 | ✅ round-trip 回归 |
| `code/gr00t/quantization/kernel_scores.py` | **第三轮**：`pool_samples_stratified`——按位置模态分层子采样（vision 前块 / text 后块各自 cap max_tokens/2）；**第四轮**：ref/q 配对重构——padding 零行 mask 仅 ref 侧计算、q 侧复用同一 `_row_keep`（`accumulate_ref_blocks`/`evaluate_blocks`），行数失配即 raise | ✅ selftest（配对回归：ref 零行/q 漂移行仍 CKA=1） |
| `scripts/tools/gr00t_consensus_plan.py` | **第四轮**：输入强制为 TopK 裁决后的 `.final_plan.json`（meta.adjudicated==true 校验） | ✅ |
| `scripts/run_v2_smoke.sh`（新） | **第四轮**：GPU 冒烟（selftests → 小规模 gate-0 → finite/coverage 检查 → A8 二次启动 save/load 验证） | ✅ |
| `scripts/tools/gr00t_gate0_check.py`（新） | **第三轮**：gate 0 硬门（finite=1.0 / W2-W8 分离 ≥2× / W2 护栏点火率 ≥50% / 种子 mask Jaccard ≥0.7 / Spearman 方向 ≥0.2），exit 0/1，编排脚本失败即中止 | ✅ selftest |
| `scripts/tools/gr00t_consensus_plan.py`（新） | **第三轮**：三开发套件 FP16 mask 两两 Jaccard ≥0.7 硬门 + 多数投票 + 预算修复 + **冻结统一 plan**（Long/90 zero-shot 使用） | ✅ selftest |
| `scripts/tools/gr00t_topk_scorer.py` | **第三轮**：n-obs 默认 16 + best-vs-runner-up **paired bootstrap** 显著性报告；`--act-scale-path` | ✅ |
| `scripts/run_v2_gpu_experiment.sh` | **第三轮**：真管线（gate 失败即中止、server 生命周期托管、v2 vs uniform W6 vs random best/median/worst 同 seed 对照、held-out 3 种子） | ✅ 语法检查 |
| `code/examples/Libero/eval/*` | **第三轮**：eval 客户端 `--seed` 支持（同 seed 跨配置对照） | ✅ |

**运行手册（groot_test 环境 + 空闲 GPU）**：

```bash
cd /home1/gyy/vla/QuantVLA
export PYTHONPATH=/home1/gyy/vla/QuantVLA/code:$PYTHONPATH

# gate 0) 指标有效性实验（先做；不过则不要继续）
python scripts/tools/gr00t_metric_audit.py --suite spatial --layers-subset 30 --bits 2,4,6,8

# 1) 度量层（主 probe 只测 W4；护栏同 pass；w_i 8 obs × 2 噪声）
python scripts/tools/gr00t_sensitivity_probe.py --suite spatial \
    --n-obs 16 --bits 4 --group 64 --n-rollout-obs 8

# 2) 选择层（二元 + 护栏过滤 + 多样性 TopK + w_i 三段日志）
python scripts/tools/gr00t_select_plan.py \
    --sensitivity checkpoints/packs/gr00t/sensitivity_libero_spatial_g64_b4.json \
    --ckpt checkpoints/gr00t/libero-spatial \
    --out checkpoints/packs/gr00t/gr00t_quant_plan_libero_spatial.json \
    --solver greedy --binary --min-bits 4 --emit-env

# 3) TopK 裁决：对 Top10 跑完整配置 D_solver（probe 协议 3 模式）→ Top3

# 4) Phase 2：对 Top3 做 plan-specific 静态 ATM/OHB 校准（校准与部署同 act 模式）
export GR00T_ATM_ENABLE=1 GR00T_ATM_ALPHA_PATH=checkpoints/packs/gr00t/atm_alpha_beta_<plan>.json
export GR00T_OHB_ENABLE=1

# 5) 按 plan 起服务 + 评测（50 条口径快筛；结论级 3 种子 × 50 条）
export GR00T_DUQUANT_PLAN=checkpoints/packs/gr00t/gr00t_quant_plan_libero_spatial.json
./scripts/run_quantvla.sh libero_spatial      # 终端1
./scripts/run_libero_eval.sh libero_spatial --headless   # 终端2
```

**已知事项（v1.3）**：probe 双参照 = R（weight_bits=0 管线，归因）+ 纯 FP16（护栏/全局 D_solver，部署）；主 probe 只测 W4（audit 模式测 {2,4,6,8} 于子集）；w_i 只测 W4（排序权重，8 obs × 2 噪声、三段日志、bootstrap）；护栏是硬约束且必须在 W2 上点火；min_bits=4 是正式安全约束（重开门槛 §1.3.2）；skip=FP16 旁路（成本 2P、失真 0），0-bit 剪枝为第三阶段独立选项且失真须实测；贪心过滤 Δbyte≤0 的伪降档；代理目标是一阶归因近似，最终以 select_final() 规则 + LIBERO 验收为准。

### 6.8 GR00T 专属风险与对策

| 风险 | 对策 |
|---|---|
| action head 重建（`policy.py:253–276`）会丢掉已挂的 quant/ATM | probe 与 plan 应用走 `Gr00tPolicy._load_model` 同一时序（重建后再挂） |
| pack 文件碎片：per-(g,ls) 组合 × 116 层 | 固定 g=64；group 只对 TopK 层补扫；plan 只生成用到的组合目录 |
| 合成输入下 LLM 侧 CKA 信号弱（随机图像激活不充分；L2 不能解决此问题） | LLM 侧分数主要靠 CS；w_i 提供任务端权重；最终以 LIBERO 验收 |
| 代理目标与真实成功率不单调（Q-DiT 教训） | 结构上闭环：TopK 必须过 D_solver 裁决 + LIBERO 验收；代理只用于搜索 |
| CS 对幅度爆炸失明（W2 实验实证） | 自测电池 + 就地验证；不过则 λ_cs=0 并声明负结果；幅度由护栏兜底 |
| probe 走 ZMQ 服务会因每配置重启而极慢 | probe 进程内直接加载模型（不走服务） |
| diffusers processor 覆盖风险 | 采集回调统一在 `ensure_dit_attention_patch` 之后挂 |
| per-step 表跨 num_steps 不等价（key 集合变化） | fallback `all` 保证不崩；评测与校准固定同一步数（8） |
| flash-attn RPATH / GPU 选择 / 环境差异 | 沿用 `run_quantvla.sh` 的 `LD_LIBRARY_PATH`、`GR00T_GPU`、`groot_test` |

---

## 7. 风险与对策（通用，两条栈适用）

| 风险 | 对策 |
|---|---|
| CKA/CS 是启发式分数（i.i.d. 不成立、token 构成需定义） | 定位为排名信号；样本构成记录进 JSON meta；Phase 1 落实 padding mask + 模态分层子采样；最终以 LIBERO 为裁判 |
| CS 幅度失明（within-set 项对冲） | 自测电池（cross 项单调）+ 就地验证；不过则关闭 CS（负结果照写报告）；幅度由 D_rms 护栏负责 |
| 护栏阈值不可移植（per-模型） | τ 由 W4 候选 P99+margin 自动估计并随 meta 记录；点火测试兜底；换模型/换栈重估 |
| 随机 mask 基线昂贵 | 两阶段：20 个 mask 先跑 D_solver（便宜），best/median/worst 3–5 个上 LIBERO |
| w_i 小样本（rollout obs 少） | 8–16 obs × 2 噪声 + 中位数聚合 + bootstrap 稳定性报告；不稳定则回到 gate 0 |
| 归一化方式影响 λ 含义 | 联合 min-max 已固定（§5.2）；换归一化须重调 λ 并重新验证 |
| mini-batch CKA 对不均匀 batch 的等权累加偏差 | 固定有效 token 数或按有效样本加权（实现中记录该选择） |
| 搜索过拟合合成校准集 | 只做排名；TopK 加扰动复核 + Hamming 多样性；跨种子 mask Jaccard ≥ 0.7 门槛；最终配置在 rollout 上复核 |
| 量化服务与 torch.compile 不兼容（pi05 既有坑） | 保持 `TORCHDYNAMO_DISABLE=1`（pi05）；GR00T 侧沿用现协议 |
| 3/6-bit 静态字节低估实际容器开销 | 预算口径已在 meta 声明为理论字节；部署前用实测 kernel 复核 |
| 预算口径变化（v2 用 862.9MB ≠ v1 636.4MB） | 报告同时给出 uniform-W6 预算与占 FP16 权存比例；"同预算"一律指同一静态权存字节口径 |

---

## 8. 参考资料

1. Dasgupta & Cohn, *Improving Language Model Distillation Through Hidden State Matching*, ICLR 2025. 本地：`docs/paper/ICLR-2025-improving-language-model-distillation-through-hidden-state-matching-Paper-Conference.pdf`
2. Yin et al., *Distributional Vision Language Alignment by Cauchy-Schwarz Divergence*, ICLR 2026 (OpenReview: UUAjF4xL0e; arXiv:2502.17028). 本地：`docs/paper/7367_Distributional_Vision_Lan.pdf`、`docs/paper/2502.17028v3.pdf`
3. Chen et al., *Q-DiT: Accurate Post-Training Quantization for Diffusion Transformers*, CVPR 2025 (arXiv:2406.17343). 本地：`docs/paper/2406.17343v2.pdf`
4. QuantVLA（本仓库 v1 基线）, CVPR 2026 (arXiv:2602.20309)
5. 相关延伸：QVLA（ICLR 2026, arXiv:2602.03782，通道级动作敏感度分配）、HoloQ-VLA（arXiv:2605.28803，per-step scaling）

## 9. 论文定位与主张边界（v1.3，第三轮审查）

当前实现是**分层多准则选择**：CKA/CS = 软代理、RMS/saturation = 硬可行性约束、
d_solver_i = 动作重要性权重、D_solver(config) = 全局裁决、ATM/OHB = 选层后的部署期校正。
它**不是**最初的 `D_i(b_i, α_i)` 联合优化框架。

**可以主张（有实现支撑）**：
- CKA/CS-based 层敏感度代理（定位为 **representation-stability proxy**，其与任务
  鲁棒性的相关性由 metric audit 的 Spearman 与 random-mask baseline 证明，而非论文外推）；
- action-weighted W4/FP16 二元分配（w_i 来自单层干预的 paired solver divergence）；
- 幅度/饱和可行性护栏（硬约束，工程启发式）；
- 预算约束二元选择（贪心 + 0-1 背包精确解 gap）；
- config-level D_solver 裁决（TopK 八步管线，真实部署语义）；
- plan-specific 静态 ATM/OHB 校正（部署版）。

**暂不主张**：
- general mixed-bit allocation（主路径只测 W4；{2,3,6,8} 仅 audit）；
- joint bit-scale 联合优化（先选 mask、再单独校准 α/β，非联合搜索）；
- action-distribution matching（固定噪声配对点估计，非分布散度）；
- long-horizon 环境闭环轨迹（D_solver 是单 chunk 内 8 步去噪轨迹）；
- Q-DiT 式 granularity allocation（g 固定 64，无 per-group 量化尺度搜索）；
- 真实显存/时延优化（预算 = 理论静态权存字节；内核仍保留 FP 缓存权重，
  验证的是 allocation policy，不是 INT4 kernel 的峰值显存/速度）。

建议论文标题方向：**Action-Weighted Representation-Preserving Layer Selection
for Post-Training Quantization of VLAs**（或 Geometry- and Distribution-Aware
Binary Mixed-Precision Quantization for VLAs）。

---

## 附录：术语表

| 术语 | 含义 |
|---|---|
| PTQ | Post-Training Quantization，训练后量化（不重训，只校准） |
| W4A8 | 权重 4bit、激活 8bit 的量化配置 |
| DuQuant | 用旋转矩阵+置换打散异常值再量化的方法（v1 已实现） |
| CKA | Centered Kernel Alignment，中心核对齐：度量一组样本表示的几何相似度，对整体缩放不敏感（盲区：检测不到整体衰减） |
| CS 散度 | Cauchy-Schwarz Divergence：分布重叠度量；公式对称，实现以 FP 固定带宽（有向评估）；无重叠时 → +∞；**cross 项** = −2log⟨p,q⟩，自测电池的单调性断言对象 |
| D_rms / D_sat | v1.3 可行性护栏：输出通道 RMS 的对数比值中位数 / 下游 A8 饱和率；硬约束，不进入加权和 |
| 点火测试 | 用已知坏例（W2 层）验证护栏确实触发的测试 |
| audit-only bit | {2,3,6,8}：只在指标有效性实验中测量、不进入主搜索空间的 bit 档 |
| ATM / OHB | v1 的缩放校正：ATM=per-head α（std 比值）；OHB=论文 per-layer 标量 β（RMS），本仓库扩展 per-head；v1.3 起仅作部署期 plan-specific 校正 |
| denoise step | **单次动作生成内部**的流匹配去噪步（约 8–10 步），不是环境 rollout 步 |
| D_solver | 配对噪声下 FP16 与量化模型去噪轨迹的逐 step 归一化散度（solver 级，非环境长程） |
| 单层干预 | 只改第 i 层、其余层保持参照精度的测量协议（归因基础） |
| 二元选择 | x_i ∈ {0,1}：x_i=1 → W4A8，x_i=0 → skip=FP16 |
| skip | 不量化、保留 FP16（成本 2P、失真 0）；与 0-bit 剪枝不同 |
| Budget | **静态权重存储字节**（理论打包；非峰值显存/时延/BitOps）；v1.3 主预算 = uniform-W6 字节（GR00T 862.9MB），以占 FP16 权存比例作可移植口径 |
| MCKP | Multiple-Choice Knapsack Problem；v1.3 二元形式退化为 0-1 背包（scipy milp 求精确解以校验贪心 gap） |
| 进化搜索 | Q-DiT Alg.1 式求解器：种群→代理评估→TopK→交叉变异，胜者投影回可行域（v1.2 已实现；v1.3 主路径为二元贪心+MCKP+扰动） |

## 附录：v1.1 修订记录（对照评审 20 条）

| # | 评审问题 | v1.1 处理 |
|---|---|---|
| 1 | skip 四义冲突 | 统一定义：skip=不量化保留 FP16（成本 2P+bias、失真 0）；0-bit 剪枝移入 P4（失真须实测）；b=16 因被 skip 支配而从搜索空间移除；全 skip 成本超预算，无退化 |
| 2 | D_i 与全局 D_action 混用 | 两级目标：层代理 Σ w_i·S_i（协议 1 原料，一阶可加）与全局 D_solver（协议 3，只做 TopK 裁决，不拆成 Σ）分离；w_i 显式定义（协议 2 探测 bit 散度归一化，均值 1） |
| 3 | d_action 无逐 bit 表 | 明确 P0 不提供 d_solver_i(b) 全表；w_i 是排序权重非逐 bit 失真；TopK 层逐 bit 补测列入 P4 |
| 4 | "6 次前向"归因错误 | §6.2.5 改为单层干预协议（116×6 次批量前向，逐层只改自身）；全模型同 bit 只用于协议 3 全局量 |
| 5 | γ 索引方向反了；"完整 rollout"命名错误 | D_solver 权重改为 γ^{k+1}（k=T−1 最大）并归一化；分母加 ε；"完整 rollout"→"单 chunk 内去噪轨迹"，长程关系改为假设（由 LIBERO 验证） |
| 6 | 共用高斯核矩阵错误 | 明确 CKA=D×D 协方差、CS=N×N Gram，只共用池化/接口 |
| 7 | 动作分布 CS 无样本支撑 | 单噪声配对=点对，不声称分布散度；条件分布 CS 列为 P4 可选扩展 |
| 8 | 线性 CKA 与高斯核 CS 混说 | 同 #6 |
| 9 | CS 三条边界过强 | "零重叠稳健"改为"无重叠时同样→∞，优势在 KDE 可估性"；对称性改为"公式对称、FP 固定带宽下估计有向"；i.i.d. 声明为启发式分数 |
| 10 | Q-DiT group ≠ DuQuant block | 明确 g=旋转块（只影响 R 矩阵，scale 恒 out×2）；per-group 量化尺度（out×ceil(in/g) scales）列为 P4 扩展；block_in/block_out 相等约束注明为当前假设 |
| 11 | 预算≠BitOps/显存 | Budget 重新定义为静态权存理论字节；3/6-bit 理想打包、不含激活/峰值/时延已声明；术语表同步 |
| 12 | b=16 语义不清 | 从候选集移除（被 skip 支配），语义声明写入 §3.1 |
| 13 | min-max 归一化未定义 | 固定为全部 (层,bit) 联合 min-max（跨层跨 bit 一次），max=min 置 0；代码同步 |
| 14 | cosine 尺度敏感表述错误 | 修正动机：CKA 的关键差异是集合级几何、跨维度、正交不变；并补充 CKA 整体缩放盲区 |
| 15 | "相似度取代 scale"过头 | 改为"scale 校正与相似度度量并存"，§1.3/§2.1 同步 |
| 16 | OHB 描述与论文不符 | 区分论文 per-layer 标量 β 与本仓库 per-head β 扩展；ATM/OHB 只对齐二阶统计量（std/RMS），不匹配均值 |
| 17 | L2 "on-manifold"过强 | 改为"仅保证 x_t 在随机条件输入的流轨迹上"，不能解决随机图像/LLM 侧问题 |
| 18 | per-step key 与 num_steps 无关过强 | 改为"坐标系无关但 key 集合取决于 num_steps，跨步数 fallback all，不保证等价" |
| 19 | 39% 显存类比错误 | 删除该类比 |
| 20 | 50 条统计不足 | 首里程碑改 ≥76%（38/50，可观测值）；结论级 3 种子×50 条 + mean±SE + bootstrap CI + 配对显著性；50 条仅作冒烟 |

## 附录：v1.2 修订记录（对照第二轮评审 11 条）

| # | 评审问题 | v1.2 处理 |
|---|---|---|
| 1 | `set_all_bits(16)` 与 FP16 skip 语义冲突，单层干预可能混入上游 16-bit 失真 | **参照协议重定义**：参照 R = 量化管线所有目标层 `weight_bits=0`（全精度权重路径，已核 `DuQuantLinear.forward` 分支；旋转/A8 保持）。单层干预 = 第 i 层设 b、其余 0——两侧共享 wrapper 与上游，归因无混淆。`weight_bits=16` 明确不是参照。probe 已改：参照 pass（all-0）只跑一次并缓存（输出 + 轨迹），w_i 与 CKA/CS 全部对 R 比较；纯 FP16 仅用于全局 D_solver 部署配对。运行命令 `--bits` 默认改为 `2,3,4,6,8` |
| 2 | TopK "重排"没有数学规则 | `select_final()` 公式化：d_min = min D_solver；平局集 T = {c: D_solver(c) ≤ d_min + 5%·max(d_min,ε)}；c* = argmin_{c∈T} L_proxy。字典序（任务指标优先，代理只破近似平局），已实现 + 单测 |
| 3 | 校准/部署 act 模式不一致风险 | probe 基准模式显式化（ATM/OHB 关、per-step 关、静态 act，写入 sensitivity.json meta）；`set_quant_env` 显式清空环境中的 ATM/OHB 变量；校准器加 `--act-dynamic` 开关，手册要求"校准与部署同 act 模式" |
| 4 | 量化成本不保证随 bit 单调下降（小层旋转矩阵开销） | 文档明确该事实 + 反例；贪心/进化对每个降档动作**过滤 Δbyte ≤ 0**（代码已有该 guard，文档补齐声明） |
| 5 | w_i 边界情况未定义 | w_i = n·d_i/Σd_j：Σ≤0 → 全 1；缺测取均值；归一化后 **winsorize [0.5, 2.0]**；权重方差（`d_solver_b4_std`）随表记录；n_rollout_obs=4 的小样本问题写入文档（P4 增样） |
| 6 | 轨迹索引 off-by-one | `return_trajectory` 现在返回 **T+1 个状态**（k=0 初始噪声 … k=T 最终动作）；权重 w_k ∝ γ^{k+1} 覆盖 T+1 项并归一化；"跨步数可比"降级为"近似可比（不同 T = 不同积分网格），实验固定 8 步" |
| 7 | S_i(b_i,g_i) 声明超前于实现 | 能力声明改为 **fixed-block (g=64) 的 action-weighted mixed-bit 搜索**；g∈{128,256} 逐层分配明确为 P4 |
| 8 | token 样本构成只是"记录"不够 | 文档明确 padding 未 mask、vision 主导、有效 N 跨层变化等偏差真实存在且仅被记录；按模态分层子采样列为 P4；子采样为确定性 stride（非前部偏置） |
| 9 | CS 取值范围/σ=0/跨层可比性 | D_CS 改为扩展实数 [0,+∞]；σ 下限 1e-3 写入文档（代码已防护）；新增跨层不可比性声明（联合 min-max 只重缩放、不消除估计器尺度差异，分数只支持序比较） |
| 10 | 二项 SE 不适用于 task-clustered 数据 | 结论级统计改为：per-task 成功率 + **task-level cluster bootstrap** + **task-level 配对置换检验**（episode-level McNemar 降级为参考）；理想化 SE≈3.5pp/CI≈±6.9pp 的数字写清楚 |
| 11 | LIBERO 反复调参 = benchmark leakage | **held-out 政策**：调参与开发只用 spatial/goal/object 三套件；libero_10/libero_90 仅在最终验收时跑；开关决策记录到 `runs/v2_decisions.md` |

## 附录：v1.3 修订记录（第三轮评审·范围收敛，对照实验报告与收敛评审）

| # | 评审/实验事实 | v1.3 处理 |
|---|---|---|
| 1 | 首轮实验：bit 维信号平坦（W6/W8 合计仅 0–1 层被选），W2 计划致 spatial 0%（±8.1 vs ±1.3），CKA/CS 失明 | **方法收缩为二元 W4/FP16 选层**（§0、§1.3、§3.1）；{2,3,6,8} 降为 audit-only（§3.1）；`min_bits=4` 定为**正式安全约束**，重开 W2/W3 需过六实验门槛（§1.3.2） |
| 2 | CS 对幅度爆炸失明，但现有 selftest 的 `CS(X,3.7X)>1e-2` 能过 → 不是简单核 bug | CS 自测电池（缩放阶梯分项断言 cross 项单调、均值漂移/旋转/outlier/covariance/permutation）+ 就地缩放验证 + 关闭准则（λ_cs=0，负结果照报）（§5.1.2） |
| 3 | W2 是可行性问题，加权和无法兜底 | 新增 D_rms/D_sat 硬护栏（vs 纯 FP16，τ = W4 候选 P99×1.5），**点火测试**为护栏自身验收（§5.1.3、§3.1 公式 3a/3b） |
| 4 | 归因参考与部署参考混用 | 双参照分工表：CKA/CS/w_i 用 R（归因），D_rms/D_sat/全局 D_solver 用纯 FP16（部署可行性）（§3.1） |
| 5 | 报告"w_i 0.0002–0.0010"实为 raw d_solver，无法确认权重参与搜索 | w_i 三段强制日志（raw/normalized/final 的 min/max/mean，mean≈1、[0.5,2]）；采样规范 8–16 obs × 2 噪声、中位数聚合、bootstrap 稳定性（§5.2、§6.3） |
| 6 | mixed-bit 求解器与实际结果（二元）不符 | 二元求解器为主路径：贪心 + scipy milp 0-1 背包（gap）+ 扰动近邻 + λ 扫描；FP16 mask Hamming 多样性 → Top10（§5.3、§6.3） |
| 7 | TopK 全局 D_solver 裁决未执行（报告"待做"） | 八步裁决管线公式化并列为 Phase 1 必做项；D_solver 必须报 per-obs std（§3.1、§5.1.4） |
| 8 | ATM/OHB 与选层方法边界不清 | **核心方法不依赖 ATM/OHB；部署版加 plan-specific 静态 ATM/OHB**；四配置静态消融先行；全 FP16 block 跳过；per-step 以 CV_t 门槛后置；2D 消融（§5.4） |
| 9 | 缺同预算/随机对照，选层价值未被证明 | 基线矩阵（11 配置）+ 两个必须通过的比较（v2 vs random mask 同层数、v2 vs uniform W6 同字节）；随机基线两阶段协议（§6.5） |
| 10 | 未验证分数能否区分 bit、排名是否稳定 | 指标有效性实验（gate 0）：20–30 代表层 × {2,4,6,8} × 指标 → Spearman vs d_solver、bit 单调性、W2 分离度、3 种子稳定性；四阶段路线图 + 逐阶段退出准则（§6.6、§6.6.1） |
| 11 | 预算口径与实验不符（文档 636.4MB vs 实际 862.9MB） | 主预算改为 uniform-W6 静态字节（862.9MB），并记录占 FP16 权存比例作可移植口径；v1.2 "636.4MB 默认"废止（§3.1） |
| 12 | 四套件各出 plan，跨套件一致性未定义 | 开发三套件 FP16 mask 两两 Jaccard ≥ 0.7 作为可复现门槛；部署用统一 plan（§6.5） |
| 13 | 随机图像 token 构成偏差仅"记录" | Phase 1 实施 padding mask + 模态分层子采样（vision/text/state 各自 cap）（§5.1.1） |

## 当前状态判定（v1.3）

v1.2 的度量与选择代码闭环已完成（probe 三协议、selector 贪心/进化/select_final、min-bits 护栏、per-step 校准器），并已产出首轮 LIBERO 结果（avg 87.0%、Long 76%，见 `docs/quantvla_v2_gr00t_experiment_report.md`）。该轮同时暴露了三个必须修正的问题：**CKA/CS 对低 bit 幅度爆炸失明、w_i 报告口径与定义脱节、缺乏同预算/随机对照**。

v1.3 据此将方法收敛为"**动作加权 W4/FP16 二元选层 + 幅度/饱和硬护栏 + 部署期 plan-specific 静态 ATM/OHB**"，并把下一步工作组织为四阶段（含 gate 0 指标有效性实验）。**v1.3 全部代码增项已于 2026-08-15 实现并通过 CPU 单测**：kernel_scores 自测电池（cross 项阶梯严格单调）、probe 护栏采集/主 W4/audit 子集/w_i 8×2 噪声规范/CS 就地验证、selector 二元模式 + 护栏硬过滤 + milp + 扰动/λ + Hamming 多样性 + w_i 三段日志 + bootstrap、`gr00t_metric_audit.py`、`gr00t_baselines.py`、calibrator plan-aware 全 FP16 block 跳过 + CV_t 统计。

**P0 正确性审查（2026-08-15）后的修复已全部落地并通过 CPU 回归**：权重不可变性 + bit 顺序不变性、A8 满批校准生命周期、D_solver 加权和与第二组噪声真正配对、metric audit 的 bit 复位顺序、run_quantvla.sh 默认关 ATM/OHB、baseline 字节匹配与真实部署语义、selector 的 missing-score/guard 继承/canonical 重打分/预算断言、新增 TopK 裁决工具 `gr00t_topk_scorer.py`、calibrator 真实 plan 加载与 α/β 拆分判定。细节见 `docs/quantvla_v2_p0_review_response.md`；受污染的旧 sensitivity/plan/ATM JSON 已移入 `checkpoints/packs/gr00t/deprecated_v1.2/` 作废。

剩余工作为**执行层**（GPU 实跑）：Phase 1 先跑 gate 0 指标审计与护栏点火验证，再按八步管线做 TopK D_solver 裁决与基线矩阵对照。**进入 Phase 1 的判据**：指标审计（分数区分 bit、W2 分离度）+ 护栏点火测试通过；**离开 Phase 1 的判据**：v2 ≥ uniform W6 且 ≥ random mask（task-level 配对检验）、跨 3 个校准种子 FP16 mask Jaccard ≥ 0.7。全部判据满足后，方法才具备论文级评测资格；文档定位：**Phase 1 执行前的最终设计稿**。
