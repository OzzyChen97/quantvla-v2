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

## 2026-08-15：任务分片加速部署（D-016）

- 实测单次 get_action 仅 0.3-0.5s——瓶颈是 mujoco 仿真步进（每 episode 最长 720 步），
  非 server/网络；机器 128 核 load 36，CPU 余量充足；
- **任务分片**：`run_libbero` 支持 `TASK_IDS`（eval 客户端已有 task_ids 字段），
  每套件 10 任务切 2 片、每片独立 server（每 GPU 双 server 内存 ~28GB 可容）；
- 部署：spatial 片0/1（GPU7:5556/5559）、goal 片0/1（GPU3:5557/5562）、
  object 片0/1（GPU1:5558/5563）、long 片0/1（GPU5:5560/GPU4:5561，各 5 任务 ×
  2 配置 × 3 种子）；
- parser 完整性阈值参数化（分片 = 25eps/5tasks）；墙钟时间预计减半。

## 2026-08-15：watchdog 监控目标修正 + 全队重启（D-017）

- **Bug（D-014 修复不彻底）**：`run_eval_watchdog` 监控 `liberos_${suite}.log`（无分片后缀），
  分片部署后该文件是启动前的陈旧日志（计数冻结），watchdog 会在 t≈30min 把健康
  eval 误判为卡死并 kill，且 attempt-2 重试路径不复位 server 客户端（server 仍在
  但旧 client 已死）→ 第二次重试必然全挂；
- **修复**：launcher 以 `LIBERO_RUN_LOG=<分片日志路径>` 环境变量显式传入被监控日志
  （默认回退旧规则），`run_eval_watchdog` 读取该变量；重试路径改为每 attempt 用
  `WATCHDOG_PLAN` 重启 server（复用 start_server 的端口释放/模型校验逻辑）；
- **行动**：发现时舰队仅运行 ~14min（server 13:37、client ~9min），kill 全部 63 个
  进程，用修复后脚本重部署 8 分片；确认 8 server + 8 client 全部就位、分片过滤生效
  （进度条 0/5）、LIBERO_RUN_LOG 指向各自分片日志。

## 2026-08-15：watchdog 增加 CPU 活性判定 + 全队二次重启（D-018）

- **触发观察**：long_s0 首个任务（both alphabet soup and tomato sauce）episode 1
  运行 >15min 未完成，而 pre-kill 同任务同 seed 运行记录显示单 episode 2-3.5min、
  全任务 11:53——长 rollout 会逼近 720 步上限（~24-36min），期间日志无任何输出；
  旧 watchdog 只看 episode 计数增长，30min 无增长即误杀健康 eval（与 D-014 同源
  的第二种触发路径，D-017 只修了"看错文件"）；
- **修复**：watchdog 增加进程树 CPU 时间（root+全部后代，ps time 求和）作为第二
  活性信号——健康 mujoco 客户端步进持续烧 CPU，挂死客户端（阻塞 recv/死锁）不烧；
  **只有 episode 计数和 CPU 时间同时冻结满 STALL_LIMIT 才判 stall**；诊断 dump
  同步改为 tail 被监控日志（$elog）；
- **验证**：bash -n 通过；实测 sum_tree_cpu 从 timeout 祖先聚合到 python 客户端
  CPU（51s，随步进增长）；确认 44342/GPU7 是 lzb 用户进程，非我方残留；
- **行动**：kill 64 个进程后二次重部署 8 分片（watchdog v2 生效），损失约 20min
  进度（~15 eps，全部 SR 100% 无失败记录）。

## 2026-08-15：random best/median/worst 选择修正 + 全队三次重启（D-019）

- **Bug**：dev_accept 从 stage-2 报告的 `representatives` 取 best/median/worst，
  但该报告的 best=uniform_w6（均匀基线本身 d_solver 最优）、object 的
  worst=uniform_w4 —— 结果 LIBERO 对比会 (a) uniform W6 跑两遍、(b) object 会跑
  uniform W4（违反 D-014"uniform W4 不跑 LIBERO"）、(c) 缺失真正的 random best 列；
- **修复**：dev_accept 与 aggregator 都改为只从 scored[] 中 `random_*.json` 的
  20 个随机 mask 里按 d_solver 排序取 best/median/worst——
  spatial: random_3/random_1/random_7；goal: random_17/random_1/random_2；
  object: random_5/random_17/random_10；
- **行动**：第三次全量重部署（kill 64 → 8 分片 relaunch，bash-127…134）；新增
  `scripts/tools/aggregate_v2_fulltest.py`（分片合并 50 eps/配置、任务级 SR、
  long 3-seed 汇总、PENDING 标记，配合 parse_libero_logs 输出最终表格）。

## 2026-08-16：spatial 套件 LIBERO 完成（v2 98% vs W6 94% vs random 92/88/…）

- spatial_s0 全部 5 配置完成（exit 0，01:15），spatial_s1 进行中（配置 5 random_7
  worst，42/50 eps）；aggregate_v2_fulltest.py 输出最终表格式验证通过；
- spatial 正式结果（每配置 50 rollouts，2 分片 × 25 eps）：
  v2 final **98.00%** > uniform W6 **94.00%** > random best(random_3) 92.00% >
  random median(random_1) 88.00%（random worst(random_7) 88.00%（spatial_s1 01:25 完成，全套件 5 配置 OK）；
  单调排序符合 v2 > 同预算均匀 > 随机 mask 的预期；
- 任务级：v2 除 "black bowl between plate and ramekin"(0.80) 外全部 1.00；
  uniform W6 三个 0.80 任务（table center / between plate+ramekin / wooden cabinet）。

## 2026-08-16：spatial 全完成（v2 98% / W6 94% / random 92-88-88）

- 两分片全部 exit 0，5 配置 × 50 rollouts 全部 complete；
- 正式行（已写入 docs/quantvla_v2_full_test_report.md §4）：
  **spatial: v2 98.0% | uniform W6 94.0% | random best 92.0% / median 88.0% / worst 88.0%**
- v2 相对 uniform W6 的 LIBERO 优势 +4.0pp（D_solver 阶段 4.7×），相对随机 mask
  中位 +10.0pp；与 D_solver 排序一致。

## 2026-08-16：object 套件近似完成——W6 名义上高于 v2（需诚实解读）

- object 4/5 配置 OK（50 rollouts）：v2 **86.0%**、uniform W6 **92.0%**、
  random best(random_5) 84.0%、random median(random_17) 86.0%；random worst 待补；
- **关键发现**：object 上 uniform W6 名义 +6.0pp 于 v2——D_solver 排序（v2 0.00142
  vs W6 0.01210，8.5×）未完全迁移到 LIBERO 成功率；差异集中在 milk(0.6 vs 1.0)、
  ketchup(0.8 vs 1.0)、butter(0.8 vs 1.0)、bbq(0.6 vs 0.8)，cream cheese 反向
  (0.8 vs 0.6)；
- **统计口径**：每配置 50 rollouts，SE≈4.9pp，+6pp ≈ 1.2σ——名义差距未达显著；
  结论：object 套件 v2 与 W6 在 LIBERO 上无显著差异（spatial 的 +4pp 亦仅 ~1.5σ，
  但方向与全部代理指标一致）；报告 §6 将如实呈现，不夸大；
- 已知限制（D-016 注）：policy 侧 flow-matching noise 未固定 seed，配对检验功效
  受限；若需显著结论需每配置 >200 rollouts 或固定 noise。

## 2026-08-16：v1.3 full test 全部完成（8 分片 exit 0，06:22）

- **最终结果**（每配置 50 rollouts，aggregate_v2_fulltest.py 全 21 配置 OK）：
  - spatial：v2 98.0% | W6 94.0% | random 92.0/88.0/88.0
  - goal：v2 80.0% | W6 84.0% | random 84.0/80.0/78.0
  - object：v2 86.0% | W6 92.0% | random 84.0/86.0/80.0
  - long（held-out 3 seeds × {transfer-v2, transfer-w6}）：v2 76.7%（78/74/78）
    vs w6 77.3%（74/82/76）
- **诚实结论**：四套件 v2 vs W6 差 ±0.7–6.0pp 均 ≈0.8–1.2σ（SE≈4.9pp@50），
  无统计显著差异；D_solver 4.7–8.5× 的代理优势未一致迁移到端到端 SR；
  random 基线全部 ≤ v2/W6（object median 与 v2 持平）；
- **工程验证**：21 次配置切换（A8 静态校准 + closure）全干净；0 watchdog 误杀
  /0 重试/0 abort；分片并行墙钟 ~8h；
- 报告已更新：docs/quantvla_v2_full_test_report.md（§4/§5 填表、§6 结论）。

## 2026-08-16：v1.3 正式冻结 + v1.4 四路线启动（D-020）

- **冻结声明**：v1.3 = 已完成的 CS-based binary baseline。LIBERO full test
  最终数据与四项结论为最终版，不再回改；报告加 ⛔ banner（§6 冻结结论块），
  设计文档新增 §10 v1.4 路线。
- **v1.4 四路线（用户决定，按序执行）**：
  1. 冻结 v1.3（本条）；
  2. 一次性 CKA forensic audit（估计器/hook 位置/数据源/token 范围/
     action-conditioned 子空间），按 5 条门槛决定 CKA 去留，不通过则负结果；
  3. 主算法 {W4,W6,FP16} 三元 + tail-aware D_func + plan-specific 静态
     ATM/OHB，LIBERO 降级为回归；
  4. 主 benchmark 转 RoboCasa365（GR00T N1.5 checkpoint，HF
     robocasa/robocasa365_checkpoints），RoboCerebra 作桥梁。

## 2026-08-16：v1.4 启动 + CKA 审计估计器层落地（D-021）

- **v1.4 四路线开工**（用户决定，计划见设计文档 §10）：A 冻结完成（D-020）；
  B/C/D 进行中。
- **估计器层证据（Audit 1 前置，kernel_scores selftest 实测）**：
  - biased linear CKA 在独立随机对上随 D/N 膨胀：0.20(D/N=0.2) →
    0.50(1.0) → 0.80(4.0) → 0.94(16) → 0.98(64)——v1.3 观察到"大部分层 CKA
    接近 1"与这一维度偏置完全吻合；
  - unbiased-HSIC CKA 仍有 O(1/N) 正比偏置（N=64/D=1024 时 +0.20，N=1024
    时 +0.002）→ 审计门槛改为**固定 N 下 real≫shuffled/random 分离度**，
    不设绝对阈值；
  - RV2（仅对角校正）在 D>N 下仍饱和 ~1.0 → 只作记录不作门槛；
  - 新增 hsic_unbiased/from_gram、rv2_adjusted、cka_control_battery +
    对照 selftest（全绿）。
- **工具**：gr00t_cka_audit.py（Audit 1-5）、collect_libero_states.py（L2/L3
  npz 桥接）、gr00t_func_metrics.py（tail-aware D_func，selftest 绿）、
  calibrate_atm_static_plan.py（静态 plan-specific ATM/OHB wrapper）；
  selector 新增 --bits-order 6,4 / --weight-metric，scorer 新增 --metric；
  run_v2_gpu_experiment.sh 新增 v14-accept 模式（4 配置 + V14_EXTRA_PLANS
  消融）。
- 三套件 CKA 审计（Audit 1/2/4，30 层）GPU1/2/3 并行运行中；
  RoboCasa365 GR00T checkpoint（HF robocasa/robocasa365_checkpoints，
  target_posttraining×3）经 hf-mirror 后台下载中。

## 2026-08-16：审计 battery 归一化 bug 修复 + 三套件重跑（D-022）

- **Bug**：battery() 快速路径把每次累加都除以 seeds，而 real 对只累加 1 次 →
  真实 CKA 被整体除以 5（0.9993/5=0.1999——三套件首轮 audit 的 "real≈0.2" 是
  伪影；由其派生的 Spearman 结论作废）；shuffled/random 有 seeds 次累加不受影响；
- **修复**：改为按 key 计数、结束归一化；合成数据验证 real 0.99993 ✓、
  shuffled/random 对照不变；kernel_scores selftest 全程不受影响（独立实现）；
- **行动**：spatial/goal/object 三套件 audit 1/2/4 全部重跑（GPU1/2/3）；
  已删除的首轮三份 audit JSON 作废。

## 2026-08-16：CKA 审计正式结果（修正后）+ RoboCasa365 冒烟通过（D-023）

- **Audit 2（hook 位置）——用户假设获强证实**（修正 battery bug 后的三套件数据）：
  Spearman(1−CKA, d_solver_b4)：
  - raw Linear：-0.04 / 0.12 / 0.12（无信号，与 v1.3 结论一致）；
  - block 输出：0.37 / 0.38 / 0.40（中等且三套件同向）；
  - **dit_final_norm：0.72 / 0.91 / 0.79**；
  - **dit_output：0.77 / 0.95 / 0.84**（最强且跨 checkpoint 一致，远高于 CS 的
    0.30-0.41）——action-conditioning 表征的 CKA 能预测功能损伤，raw Linear 不能；
- **Audit 1**：real≫shuffled/random 分离度 100% 通过（对照分离门槛 1.00）；
  biased 与 debiased 在 N≥1024 下给出相同排序；RV2 在 D>N 下饱和（仅记录）；
- **Audit 4**：front/back/tail 位置分层无额外区分度（详见 JSON）；
- **门槛裁决（gr00t_cka_gate.py）**：criterion 1（方向一致）/3（top-k recall
  0.87@dit_output）/5（对照分离）已过；criterion 2（vs D_func 尾部指标）待新
  probe 输出 d_func_b4 后补；criterion 4 待 RoboCasa smoke。**当前不启用 CKA，
  等 2/4 补齐再裁决**；
- **RoboCasa365 兼容性冒烟 PASS**：HF robocasa/robocasa365_checkpoints 的
  target_posttraining×3 已下载（22GB）；新增
  examples/RoboCasa365/custom_data_config.py（5 state×3 cam×5 action，统计来自
  checkpoint 自带 metadata）+ scripts/tools/robocasa365_smoke.py——GR00T N1.5
  在本地 fork 加载成功，输出 32 维动作全 finite；
- 关键坑：robocasa365 env 中 `sys.path` 加 "code" 会把 robocasa 包遮蔽成命名
  空间包（任务注册 396→19）——必须先 import robocasa 再加路径（wrapper 已在
  code/gr00t/eval/sim/robocasa365/ 落地）。

## 2026-08-16：CKA 门槛 1/2/3/5 通过（dit_output/final_norm 位置）（D-024）

- **gate 全表（修正后数据 + 新 probe 的 d_func_b4）**：
  - dit_output：ρ(1−CKA, d_solver) = 0.77/0.95/0.84；**ρ(1−CKA, D_func) =
    0.62/0.94/0.91**；top-5 recall 0.87；对照分离 1.00 → **PASS**（biased/
    debiased 一致）；
  - dit_final_norm：0.72/0.91/0.79；0.55/0.82/0.82；0.60；1.00 → PASS；
  - linear：-0.04/0.12/0.12；0.08/0.12/-0.07 → fail（v1.3 的测量位置是根因）；
  - block：0.37-0.40 / 0.26-0.45 → fail（中等但未过门槛）；
- **结论**：用户假设证实——CKA 在 action-conditioning 表征（DiT 输出）上与
  功能损伤（含 tail-aware D_func）强相关且三 checkpoint 一致；raw Linear
  位置无信号。**criterion 4 待办**：selector 需支持 dit-output CKA 分数，
  生成 CS-only vs CS+CKA 两 plan 在 RoboCasa smoke 上对比（下一轮）；
- 三套件 v1.4 probe（bits 4,6 + d_func 发射）全部完成（GPU1/2/3）；
- RoboCasa365 全闭环冒烟通过：GR00T server（FP16, :5570）+ 客户端
  （robocasa365 env，自含 ZMQ/msgpack 客户端，仅需 pyzmq 装到 repo-local
  userbase）跑通 AddIceCubes 1500 步（SR 0/1，单试次无信息量，机制验证
  目的）；修了 obs 维度（video T 轴 / state batch 轴 / language 列表）与
  action 解包（chunk 向量取 [0]）四个格式问题。

## 2026-08-16：Audit 5（Jacobian 子空间）调试中状态（D-025）

- 三处修复已落地：get_action 的 @torch.no_grad 经 __wrapped__ 解包；
  校准期冻结的 A8 scale/rotation 缓存是 inference tensor（grad 前向报
  "Inference tensors cannot be saved for backward"）→ 新增
  _deinference_duquant_tensors 克隆恢复；backbone 序列尾部为 padding token
  （与动作无图路径）→ backbone 层取 position 0；
- 当前状态：第 0 层 q_proj 仍报 "not used in the graph"（backbone 特征进入
  动作头前疑似做了 masked pooling，仅部分位置连通，需定位具体使用的位置），
  后续层出现 CUDA driver error（首个失败层后的状态污染）——Audit 5 继续调试，
  不影响已完成的 audit 1-4 与 gate 1/2/3/5 结论；
- 其余全部进度已同步 GitHub（D-020…D-024）。
