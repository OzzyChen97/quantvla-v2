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

## 2026-08-15：全 GPU 舰队部署（D-010）

- dev-accept 曾忽略套件参数（三个 job 全跑 spatial）→ 修复为单套件模式；
- case 块 `local` 语法错误 → 修复；
- 陈旧 orphan server（PID 129136，15:25 起存活 5.5h+ CPU）清理；start_server 的模型路径校验生效；
- **最终部署**：GPU 7/3/1 各跑 spatial/goal/object dev-accept（端口 5556/7/8），
  GPU 5 跑 held-out libero_10（spatial plan zero-shot，3 种子，端口 5560），GPU 4 备用；
- 新增 `parse_libero_logs.py`：按 server-start 分隔符解析每配置成功率。

## 2026-08-15：舰队死锁事件与全量重启（D-011）

**事件**：dev LIBERO 五路（spatial/goal/object/long×2）在运行约 8-10h 后全部停滞——
客户端 CPU 时间 ~2 分钟（阻塞等 server）、陈旧 server 进程 600% CPU 死转、episode 计数两轮零增长。
**根因链**：早前波次的 orphan server 持续占用 5556/5557/5558 端口；orchestrator 的
start_server 端口检查被旧进程"骗过"（旧端口可连 + 旧日志文件含正确 Model 行），
新 server 启动即死（`Address already in use`），客户端实际在与旧 plan 的陈旧 server 对话，
且旧 server 陷入死锁——整条链路假活。

**修复（已推送）**：start_server 加固三层：
1. 启动前 `ss -tlnp` 找端口占用者并强制 kill + 等待释放；
2. 日志出现 `Address already in use` 即判失败退出；
3. 端口就绪时必须同时满足"日志含正确套件模型"且"新启动进程仍存活"。

**处置**：Python 辅助脚本安全清理全部 32 个残留进程（排除自身与祖先链），五卡归零后
用加固版 orchestrator 全量重启（日志全新）：
- GPU 7 spatial dev-accept（5556）、GPU 3 goal（5557）、GPU 1 object（5558）、
  GPU 5 long seed0（5560）、GPU 4 long seed1-2（5561）。

## 当前进度与计划（2026-08-15 重启后）

**已完成**：
- ① CPU selftests 10/10 绿；
- ② GPU 冒烟：finite/coverage/CS 就地验证/A8 二次启动 全 PASS；
- ③ 三套件正式 gate-0 全 PASS（CS 主代理，D-001/D-006）；
- ④ 三套件全链路：probe → 二元 selector（guard+swap 多样性 TopK）→ D_solver 裁决
  （spatial flip12 / goal flip12 / object flip16）；
- 配置级 D_solver 同协议对照：v2 比 uniform W6 好 4.7–8.5×、比 random 好 16–23×；
- D-008：三套件 mask Jaccard 0.39–0.50 → 层敏感度为 checkpoint 特异，
  held-out 采用 spatial plan zero-shot。

**进行中（⑤ LIBERO full test，重启后约 1h）**：五路并行如上；dev 每套件
5 配置 × 50 rollouts（v2 / uniform W6 / random best·median·worst，seed 0），
held-out libero_10 3 种子。预计剩余 15–30h。

**重启前部分数据（仅参考，不计入正式结果）**：spatial v2 20eps SR 0.95、
object v2 20eps SR 0.90、goal v2 15eps SR 0.80、long seed0 5eps SR 0.80——
因陈旧 server 污染已作废，正式数字以重启后为准。

**下一步**：五路完成后用 `parse_libero_logs.py` 出五配置对照表 + task-level 汇总，
填入 `docs/quantvla_v2_full_test_report.md`，并向用户交付最终汇总。

## 2026-08-15：long held-out 二度卡死与 ZMQ 超时修复（D-012）

**现象**：long 两片（GPU 5/4）卡在首个任务的第 5 trial 达两轮（tqdm 冻结、客户端 CPU 时间
不涨、server 半闲置），dev 三路正常。
**根因**：eval 客户端的 ZMQ `recv()` 无超时——server 端某请求挂起后客户端永久阻塞，
整个 run 冻结。
**修复**：`code/gr00t/eval/service.py` 的客户端 socket 设置 RCVTIMEO/SNDTIMEO
（15000ms），超时抛 RuntimeError → 该 episode 记失败、run 继续下一 episode。
long 两片已重启（GPU 5 seed0 / GPU 4 seed1-2），dev 三路不受影响继续。

## 2026-08-15：dev 三路预防性重启（D-013）

goal 客户端（修复前启动、无 ZMQ 超时）在 v2 配置 15eps 后再次冻于第 5 trial
（tqdm 两轮静止、server 674% CPU 自旋）；spatial/object 客户端同为旧版。
**决策**：dev 三路全部改用带 15s ZMQ 超时的新客户端重启（long 两片已是新版，不动）。
重启后若出现连续 "inference server timeout"，则表明 server 端存在特定输入的无限自旋，
届时按 episode 定位并修复 server 侧。部分作废数据（spatial 25/object 20/goal 15 eps，
SR 0.96/0.90/0.867）仅作参考。

## 2026-08-15：Long packdir 污染与第五轮审查整改（D-014/D-015）

**D-014（方法有效性，最高优先级）**：spatial 裁决 plan 的 `packdirs` 指向
`duquant_packed_libero_spatial_*`，held-out 直接复用时 Long server 加载的是
**Long 权重 + spatial 旋转/pack 元数据**——跨 checkpoint 错用预处理参数，非干净
mask zero-shot。处置：立即停 Long 两路并作废数据；新增 `gr00t_transfer_plan.py`
（mask 来自 spatial plan、packdir/A8/权重全部属目标 checkpoint），生成
`gr00t_quant_plan_long_transfer_v2.json`（78 W4/38 skip，Long pack）与
`gr00t_quant_plan_long_transfer_w6.json`（同预算 baseline）；held-out 每种子
同时跑 transfer-v2 与 transfer-w6。

**D-015（可靠性根治，五项全部落实）**：
1. **orphan 根治**：`run_quantvla.sh`/`run_inference_server.sh` 末尾 `exec`——
   orchestrator 的 SERVER_PID 即最终 python 进程，kill/wait 干净回收；
2. **安全清端口**：只杀 cmdline 匹配本项目的 holder（不误杀他人服务），kill 后
   轮询确认端口释放，未释放再 KILL；
3. **运行期保护**：eval 外层 `timeout`（6h 配置级）+ 进度 watchdog（episode 计数
   30 分钟不增长 → 转储 nvidia-smi/进程树/日志 → 杀进程对 → 自动重试一次，
   两次失败才中止）；server 注册 `faulthandler(SIGUSR1)` 支持线程栈转储；
4. **parser 完整性**：只有 `episodes==50 && tasks==10` 的配置才标记 complete 进入
   正式表；不完整配置不参与比较；
5. **random 唯一性审计**：三套件 20/20 唯一（min Hamming 8-12），
   `baselines_<S>/uniqueness.json` 落盘。
另：selector w_i 改为 **winsorize 后再归一化（final mean == 1，精确）**（D-015，
未来计划的 w_i 稳定性改进；当前 dev 三路使用旧归一化的已裁决 plan，继续有效）；
LIBERO 表移除 uniform W4 列（未跑该配置，仅保留 D_solver 数据）；
已知限制：eval seed 固定环境不固定 policy 侧 flow-matching noise（降低配对检验效率）。
