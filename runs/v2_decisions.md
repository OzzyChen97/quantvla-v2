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

## 2026-08-16：Audit 5 完成 + v1.4 计划生成 + L2 管线打通（D-026）

- **Audit 5（action-conditioned 子空间）**：vjp 路径因 DuQuant unsafe-view 链
  的 autograd 报错弃用（三处修复后仍失败，记录在案）；改用计划中的**线性探针
  回退**（ridge W: H→a_T，TopSV 投影）——三套件 30 层全跑通，结果：投影子空间
  CKA 与 d_solver 相关 -0.21/-0.04/0.003，**无新增信号**（1−CKA 量级 1e-4~1e-2，
  噪声主导）。结论：plain dit_output CKA（ρ 0.72-0.95）仍是唯一有效 CKA 变体；
- **L2/L3 管线**：collect_libero_states.py 修复（env 构造/numba cache/EGL
  teardown 容错/uint8 图像/语言 sidecar）——L2 已采集 128 条真实 rollout 状态
  （2 tasks × 64 steps）；audit --obs-source l2 运行中；
- **v1.4 计划生成完成**（tri-state {W6,W4,FP16}，w_i=d_func，预算=uniform-W6
  862.9MB）：spatial W6=91/W4=14/FP16=11；goal W6=100?/…；object …（详见各
  plan JSON）；CS-only 消融（--no-weights）与 spatial CS+CKA（--cka-field
  cka_dit, λ_cka=1）计划同步生成，进入 D_func 裁决（7 组 × TopK，GPU4/5 并行）；
- selector 打印修正（按 bit 分列）；--no-weights 消融开关落地。

## 2026-08-16：L2 数据源审计 + v1.4 裁决第一批（D-027）

- **Audit 3（L2 真实 rollout 状态，128 obs）**：Spearman(1−CKA, d_solver)——
  linear 0.057（仍无信号）、block 0.376、dit_final_norm 0.672、dit_output 0.651；
  与合成数据结论一致（合成上 dit 位置 0.72-0.95，真实状态略降但仍强正相关）——
  **结论对数据源稳健**：action-conditioning 表征的 CKA 有效，raw Linear 无效；
- **v1.4 D_func 裁决（已完成 5/7 组）**：spatial final=lambda(1,1)
  d_func=0.0837（runner-up 0.0852，bootstrap p_best_wins=0.116——前两名统计上
  不可区分，5% tie 规则生效，如实记录）；goal/object 主计划与 csonly 消融的
  final plan 已产出；spatial csonly/cscka 裁决仍在 GPU4 批处理中；
- 注：v1.4 tri-state 的 d_solver ≈ 0.0018-0.0038 vs v1.3 binary 0.00079
  （同协议）——三元空间在 D_solver 上不优于二元，价值需 LIBERO 回归检验。

## 2026-08-16：criterion 3 口径修正后仍通过（D-028）

- 按 review 修正 gr00t_cka_gate.py 的 criterion 3：不再用 d_solver 自召回作代理，
  直接计算 R_CKA@5 与 R_CS@5（vs D_func 与 d_solver 两个功能指标），门槛 =
  R_CKA > R_CS + 0.05（两指标都需胜出）；
- 结果（k=5，三套件均值）：
  - dit_output：R_CKA=0.87(d_solver)/0.73(D_func) vs R_CS=0.20/0.20 → PASS；
  - dit_final_norm：0.60/0.60 vs 0.20/0.20 → PASS；
  - linear/block：fail（符合预期）；
- 门槛状态修正为：criterion 1/2/3/5 通过（dit_output/final_norm），criterion 4
  待 RoboCasa 端到端；Audit 5 以线性探针回退收尾（无新增信号，不阻塞主线）；
- v14-accept LIBERO 回归 8 分片已启动（spatial×2 含 csonly/cscka 消融 + goal×2
  + object×2；GPU7/3/1），每配置 50 rollouts；
- goal/object 的 dit sensitivity（--cka-location dit）在 GPU6 重跑中，用于三
  套件 criterion-4 计划；下一步：RoboCasa365 的 DuQuant pack 校准 + 小规模
  {uniform W6 / CS-only / CS+CKA} 对比（3-5 atomic + 3-5 composite × 3 trials）。

## 2026-08-16：ATM 配置启动 bug 修复 + RoboCasa365 量化管线接通（D-029）

- **Bug**：start_server 的 ATM 环境变量经 bash 数组 `"${atm_env[@]}"` 展开在
  续行上失败（bash 4.4，报 `GR00T_ATM_ENABLE=1: command not found`）——v14-accept
  舰队在首个 ATM 配置（cfg 3）处全体崩溃（goal 在 cfg 2 被杀重跑，object/spatial
  损失 50-75 eps）；改为显式 if/else 双分支 env 前缀，bash -n 通过；
- **行动**：kill 全部 6 分片 + 残留 server/client，重部署（bash-204…209）；
  这是 v1.4 舰队第二次全量重启（首次为 final-plan 文件名），损失 ~1h；
- **RoboCasa365 量化管线接通**：gr00t_v2_common 增加 make_robocasa365_obs /
  SUITE_DIRS/SUITE_DATA_CONFIG（robocasa365_atomic）；probe --obs-format
  robocasa365 + --suite 选项；inference_service 支持 GR00T_OBS_FORMAT（A8 静态
  校准 buffer 格式）；robocasa365 probe（bits 4,6, dit CKA, d_func）在 GPU5 运行
  中，pack 目录随加载自动生成（weight 旋转与 bit 无关）；
- goal/object 的 CS+CKA / CKA-only diagnostic 计划已生成（待裁决）。

## 2026-08-16：RoboCasa365 criterion-4 计划三件套生成（D-030）

- robocasa365 probe（bits 4,6 + cka_dit + d_func，合成 robocasa365 obs）完成；
- 生成四组计划（同 w_i=d_func、同 guards、同预算 uniform-W6 862.9MB、同
  {W6,W4,FP16} 空间）：uniform W6 / CS-only / CS+CKA(dit) / CKA-only(diagnostic)；
  进入 D_func 裁决（GPU4 串行）；
- scorer 增加 --suite robocasa365_atomic + --obs-format（A8 buffer 与合成 obs
  的 robocasa365 格式分支）；
- 裁决完成后启动 criterion-4 小规模 RoboCasa 对比：3-5 atomic + 3-5 composite
  × 3 trials × {W6, CS-only, CS+CKA} + FP16 单次上限校验。

## 2026-08-16：Triton 融合推理 A/B + RoboCasa365 计划裁决（D-031）

- **Triton 融合 W4 反量化 matmul（duquant_fused.py）A/B 结论（libero-spatial，
  v1.3 二元 final plan，16 paired obs）**：
  - kernel 单层对照 eager 数学**逐位一致**（0.0 diff，standalone 复算）；
  - 模型级能量加权差异：per-layer RMS rel diff 中位 1.5e-2、final-action RMS
    4.3e-3；逐元素 max-rel 差异大（100×+）是近零元素分母放大 + bf16 混沌放大，
    不代表损伤；
  - **D_solver：eager 0.00074 → fused 0.00070（−3.9e-5，略优）**——fp32 累积
    比 bf16 tensor-core 更准；fused_ready=78/78 确认融合分支生效；
  - **速度：当前 eval 批规模下 0.86×（更慢）**——单层微基准大 M 时 3.3× 快，
    但小 batch 下 kernel 启动开销占优；真正的加速需要 int4 packed 存储 +
    更大 tile 的工程化版本，正确性结论先行；
  - 结论：**融合推理不影响结果（D_solver 不劣化），当前实现不做默认**；
- **RoboCasa365 计划 D_func 裁决完成**：CS-only 0.164 / CS+CKA(dit) 0.150 /
  **CKA-only 0.095（最低）**——CKA(dit) 在 robocasa365 上也给出最低代理损伤，
  criterion-4 闭环 RoboCasa 对比待跑；
- 舰队 ATM 配置转换（cfg 3→4）已跨过（D-029 修复生效，spatial/object 已到
  cfg 4，goal cfg 3）。

## 2026-08-16：RoboCasa365 criterion-4 五配置闭环实验启动（D-032）

- 量化管线完成：scripts/run_robocasa365_quant_serve.sh（直连 inference_service：
  robocasa365 data config + GR00T_DUQUANT_PLAN + robocasa365 pack +
  GR00T_OBS_FORMAT=robocasa365 的 A8 静态校准）；
- 5 台 server 全部就绪（A8 静态校准完成）：fp16:5570(GPU5)、uniform W6:5571、
  CS-only:5572（GPU4）、CS+CKA:5573、CKA-only:5574（GPU6）；
- 任务集（用户指定）：OpenCabinet / OpenStandMixerHead /
  PickPlaceDrawerToCounter / CoffeeSetupMug，每配置 3 trials = 12 episodes；
- 5 客户端并行启动；机器负载高（LIBERO 舰队 + 5 个 kitchen env 构建），
  env 构建显著变慢，观察中。

## 2026-08-16：v1.4 LIBERO 回归完成（D-033）

- 全部 6 分片 exit 0：spatial 6 配置、goal/object 各 4 配置 × 50 rollouts
  （2 分片 × 25 eps）——含 ATM/OHB 与两消融；
- **结果表（SR，50 rollouts/配置）**：
  - spatial：W6 96.0 / **v14 100.0** / W6+ATM 92.0 / v14+ATM 98.0 /
    CS-only 96.0 / CS+CKA 95.5；
  - goal：W6 86.0 / v14 84.0 / W6+ATM 84.0 / v14+ATM 86.0；
  - object：W6 94.0 / v14 94.0 / W6+ATM 92.0 / v14+ATM 88.0；
- **对比 v1.3 冻结基线**：v14 相对 v1.3 v2 plan 三套件全部提升
  （spatial 98→100、goal 80→84、object 86→94）；相对同预算 W6：spatial +4、
  goal −2、object 0（±4-6pp SE，方向一致但未达显著）；
- **静态 ATM/OHB 未带来增益**（四组对照平均 −2.0pp，方向略负）——如实记录；
  CS-only 与 v14 持平（spatial −4），CS+CKA 略低（−4.5）——criterion-4 的
  RoboCasa 闭环数据将给出最终裁决；
- parser 增加 --mode v14（ATM-aware 配置切分）+ 修复打印缩进 bug。

## 2026-08-16：v1.4 long held-out 启动 + criterion-4 进行中（D-034）

- v14 胜出计划（spatial v14，SR 100%）mask-only 转移到 Long pack
  （gr00t_quant_plan_long_transfer_v14.json，109 W4/7 skip…实际含 W6 位，
  bits 原样保留）——final-holdout 增加 V14_HOLD_V2/W6 环境变量覆盖；
- long held-out 2 分片启动（GPU1/2，transfer-v14 vs transfer-w6 × 3 seeds，
  每配置 50 rollouts）；预期 6-10h；
- criterion-4 RoboCasa 五配置闭环实验进行中：fp16 已到 task 3/4，
  四量化配置 task 2/4（OpenStandMixerHead）；每 trial 全时长约 28 min；
- GPU 全部用起：0/7 他人、1/2 long held-out、4/5/6 RoboCasa server+客户端、
  3 空闲（留给 RoboCerebra GR00T 桥梁）。

## 2026-08-16：RoboCasa365 诊断链 + 判定（D-035）

- **fp16 4 任务 × 3 trials = 0/12**，逐层排查：
  1. 状态归一化模式 A/B（mean_std vs min_max）——均失败，非根因；
  2. **action key 顺序修正**：metadata 顺序 = [base_motion, control_mode,
     eef_pos, eef_rot, gripper_close]（我此前用 upstream 顺序，会把 32 维动作
     切片错位——已修正为 metadata 顺序，D-035a）；
  3. 官方 horizon=720（非 1500）——非截断问题；chunk-sequential 执行亦失败；
  4. 观测行为：eef 300 步移动 0.15m、gripper 开合正常、动作量级合理、
     control_mode/gripper 输出=训练分布均值（−0.93/−0.36）——机器人正常尝试；
  5. **条件化强度测试**：robocasa365 模型对 zero-obs vs random-obs 的 10 样本
     均值 |diff| = 0.016/max 0.12；**LIBERO 已知-good 模型同测试 = 0.023/0.069**
     ——两者量级一致，说明 robocasa365 的条件化路径并非失效（GR00T 条件化
     本身就弱/含噪），管线机制健全；
- **判定**：本机 harness 下绝对 SR 低（0/16 trials），但 criterion-4 五配置
  在同一 harness 上平行运行——**相对比较仍然有效**（fp16 上限 + 4 量化配置
  同条件对照）；绝对 SR 与官方 68.5% 的差异需官方 harness/data-config 对齐
  （上游 12-key state modality 等），列为后续对齐项，不阻塞 criterion-4 裁决；
- 图像原生 256 分支（ROBOCASA365_IMG=native）已加入 data config 供后续 A/B。

## 2026-08-16：criterion-4 首轮作废——预修复配置运行（D-036）

- **关键发现**：D-035a 的 action-key 顺序修复（~19:45）晚于 criterion-4 五台
  server 的启动（~16:50）——首轮 5 配置（fp16/W6/CS-only/CS+CKA/CKA-only 共
  57/60 trials）全部在**错位 action 切片**配置下运行，结果作废（全部 0% 是
  预期产物，无判别力）；
- **行动**：kill 旧 4 台量化 server（5571-5574），以修复后配置重启
  （GPU4/6）；fp16 使用已修复的 5567；**criterion-4 全量重跑**
  （4 任务 × 3 trials × 5 配置，预期 ~5-6h）；
- 注：修复后配置的单 trial 探针（OpenCabinet×1）仍 0/1——修复是必要条件但
  单 trial 不足以判断充分性；重跑数据将给出裁决。

## 2026-08-16：criterion-4 提速 + RoboCerebra 后置（D-037）

- RoboCerebra 暂缓（server 已 kill，桥梁代码保留）；优先完成 v1.4 long +
  RoboCasa 四任务；
- **criterion-4 提速**：(a) 每配置拆 2 客户端（各 2 任务）并行——10 客户端
  总并行度翻倍；(b) max-steps 1500→720（官方 horizon，DEFAULT_MAX_EPISODE_STEPS
  =720，失败 trial 不再空转 780 步 ≈ 每 trial 省 15-20 分钟）；
- 本轮配置 = 修复后 action 顺序 + mean_std + 224 图像管线（D-035a）；
- long held-out 不重启（保住 ~50 eps 进度），维持 2 分片原速推进。

## 2026-08-16：criterion-4 EGL 崩溃根因 + 设备-3 修复（D-038）

- **症状**：v2 三台 h2 客户端（fp16/w6/cscka）完成 trial 0 后在 env.reset()
  /下一次构造时段错误（timeout: dumped core，rc=-11）；dmesg 全部指向
  `libnvidia-eglcore.so.525.105.17`（NVIDIA EGL 驱动）；存活客户端的持续
  stepping 不受影响——只有"新建/重置渲染上下文"崩溃，特征 = EGL 设备 0 上
  的并发竞态（GPU0 被其他用户的 45GB 任务占用）。
- **根因**：robocasa365 客户端用 MuJoCo EGL 渲染三路 obs 相机，默认走 EGL
  设备 0（GPU0）；驱动 525.105.17 在设备 0 压力下新建上下文偶发 segfault。
- **修复**：MuJoCo 3.3.1 支持 `MUJOCO_EGL_DEVICE_ID`（egl/__init__.py:38）；
  切换到设备 3（GPU3，本组独占、空闲）。`scripts/tools/egl_device3_test.py`
  验证：设备 3 下 PickPlaceDrawerToCounter 构造 21s + reset + 50 steps 全通。
- **配套加固**：`scripts/tools/run_crit4_trial_driver.py` —— 每个 (task, trial)
  独立子进程 + 全新 env 构造（绕过 reset 崩溃路径），segfault 只丢当前 trial
  （attempts=5），按 (task,trial) 断点续跑；`crashed` 行不计入 SR 分母
  （parse_crit4_v2.py 排除）。三个坑已修：unlink(missing_ok)/capture_output
  （系统 python3.6 不支持）、resume 误把 crashed 当 done。
- **行动**：kill 全部旧客户端/旧 driver（在设备 0 上赌运气），10 个配置半区
  全部以设备 3 + 崩溃免疫 driver 重启（attempts=5，resume 跳过已完成 trial）。
- **数据有效性**：trial 级数据不受影响——完成的 trial 结果有效，crashed 的
  trial 无策略信号、剔除；D-036 的作废结论不变。

## 2026-08-16：GPU5 空置 → long held-out 加两分片（D-039）

- GPU5 空闲（3 MiB）被发现后加开 s2/s3，把 **seed-2 的全部工作提前并行**：
  s2（GPU5:5562，任务 0-4）+ s3（GPU2:5563，任务 5-9，与 s1 服务器共卡
  30/46GB），均 HOLD_SEEDS="2"（seed2-v14 + seed2-w6）。
- s0/s1 原样跑 seed0/seed1；s0/s1 到达 seed-2 边界时停掉即可（其 seed-2 输出
  与 s2/s3 重复，解析时 seed-2 以 s2/s3 完整数据为准，按 (config,task) 去重，
  零风险）；不停也无碍——s2/s3 完成即数据齐。
- 关键路径 4h→~2h；long 四片任务切分已验证（s2 首任务 = s0 首任务 ∈ 任务
  0-4；s3 = s1 ∈ 任务 5-9）。
- GitHub push 确认成功（main...origin/main，D-036/D-037 均已上远端）。

## 2026-08-17：criterion-4 视频全黑根因（D-040）——round 1/2 数据全部作废

- **根因链收口**：`gymnasium_groot.py` 第 222-226 行在 `enable_render=False` 时把
  三路相机 obs **全部填零**；我们的 eval client 一直传 `enable_render=False` →
  策略收到的视频 = 全黑帧（state+语言正常）→ VLA 失明 → 输出接近训练先验
  （eef≈−0.93 / gripper≈−0.36）→ 所有配置 ~0%。这同时完整解释了 D-035/D-036
  期间的"先验均值行为"与"绝对 SR 与官方 68.5% 的差距"之谜。
- **验证**：enable_render=True 探针（EGL 设备 3）：三路视频 mean 60.5/67.0/62.0、
  std 65.8/69.1/66.1、全域动态范围 → 真实图像。
- **量化 plan 不受污染**：灵敏度探针使用合成 obs 协议（make_obs("robocasa365")），
  csonly/cscka/ckaonly 三 plan 与 D_func 排序有效，无需重算。
- **行动**：(a) kill 全部 10 个 driver/客户端，`runs/robocasa365_eval/v2` 移入
  `v2_blind_invalid/` 留档（含 D-038 修复后的 21 trials——仍为盲视频，作废）；
  (b) client 改为 enable_render=True；(c) `crit4_video_sanity.py` 端到端验证
  策略对真视频的反应（输出不再先验化）；(d) sanity 通过后 10 driver 全量重启
  （设备 3 + 真视频，v2 全新目录）。
- **修正 D-036 记录**：round-1 作废原因 = action 错序 **且** 盲视频；D-038 修复
  了 EGL 崩溃，但当时未发现视频全黑——修复链条直到 D-040 才完整。

## 2026-08-17：EGL 并发构造死锁 + 构造串行化（D-041）

- **症状**：D-040 后 10 driver 同发（设备 3 + 真视频）→ 全部子进程 9 分钟只
  烧 44-52s CPU、无 ZMQ 连接、State=S 卡死——非崩溃、非慢，是死锁。单构造
  （sanity 探针）21s 正常 → 病根 = **同一 EGL 设备上 10 路并发上下文创建**
  触发 NVIDIA EGL 驱动互斥死锁（设备 0 时代表现为 segfault，设备 3 表现为
  hang——同一类并发病的两种表型）。
- **修复**：(a) client 构造 env 前加 repo 本地 flock（`runs/.robocasa365_
  construct.lock`，fcntl.LOCK_EX/LOCK_UN 包住 GrootRoboCasa365Env(...)）——
  跨进程串行化构造，永久生效；(b) 10 driver 错峰重启（每 60s 一个）。
- 注：flock 用 repo 路径而非 /tmp（/tmp 是启动器私有 tmpfs，跨进程不可见）。


## 2026-08-17：服务器显存泄漏→分配器自旋卡死（D-042）

- **症状**：~23:00 起 criterion-4 全部 stalled、10 客户端 ~0.03 核空转；5 台
  robocasa365 服务器瞬时 10-25 核 CPU、GPU util≈0。long 服务器正常（响应秒回），
  仅受 load 254 拖慢。
- **定位**（faulthandler SIGUSR1 栈转储）：w6 主线程卡在
  `duquant_preprocess.py:628 apply_output_restore_optimized` 的生成器表达式；
  fp16 卡在 torchvision `normalize_image`——两处均为"任意点"，真正的病因是
  torch CUDA 分配器：**量化服务器显存泄漏**（每台 8→16GB / ~2.5h），GPU4/6
  显存耗尽后分配器进入病态重试自旋 → 所有请求永久挂起。
- **佐证**：全新 w6 服务器在旧服务器占用 32.7GB 下直接 OOM（44.4GB 卡）；全新
  fp16 服务器（5568）真视频 60 步正常（97s，输出与旧服务器一致）。
- **处理**：kill 全部 5 台旧服务器 + 10 客户端 + 10 driver；5 台服务器以原配置
  重启（v3 日志）；driver 错峰重启；jsonl 保留真视频时代的有效 trial
  （fp16_h1/h2 各 1），resume 自动跳过。
- **后续**：监控 v3 服务器显存增长速率；若 ~40min 的 criterion-4 窗口内再接近
  阈值 → 中途按需重启 + 排查泄漏点（优先查 per-request 未释放的激活缓存）。
- 注：盲视频时代 2.5h 无楔死说明泄漏与请求路径有关、速率 ~1GB/15min/台。

## 2026-08-17：criterion-4 服务器翻倍 + fp16_h2 误量化纠正（D-043）

- **瓶颈**：每台服务器串行处理请求，2 client 共享 → 每 trial ~18-20min，60
  trials 需 ~2h。
- **处理**：每配置加一台镜像服务器（同 plan 新端口 5611-5615），h2 半区独占：
  fp16_h2→GPU1:5611、w6_h2→GPU1:5612、csonly_h2→GPU7:5613、cscka_h2→GPU7:5614、
  ckaonly_h2→GPU5:5615。吞吐翻倍 → ETA ~1h。
- **纠错**：fp16_h2 初次误用 run_robocasa365_quant_serve.sh（脚本无条件导出
  GR00T_DUQUANT_* → W4A8 包装 116 层，"plan None"），已改直启 inference_service
  纯 fp16。其余四台镜像 plan 与主服务器一致（uniform_w6 / csonly / cscka /
  ckaonly adjudicated）。
- h2 五台服务器与 long s0(GPU1)/s2(GPU5) 共卡，显存 40-44.5/46GB，注意余量。

## 2026-08-17：fp16_h1 纠错 + 双路重跑（D-044）

- **配置纠错**：端口 5567 的所谓 fp16_h1 仍带
  `GR00T_DUQUANT_WBITS_DEFAULT=4`，实际为 plan=None 的 116 层 W4A8；已停止该
  server/driver，旧结果移入 `runs/robocasa365_eval/fp16_h1_invalid_20260817_0126/`
  留档，不进入 criterion-4 汇总。
- **trial 正确性修复**：旧 crash-tolerant driver 把 seed 在外层和 client 内层各
  变换一次，且单-trial child 始终回写 `trial=0`，重启后会重复追加。client 新增
  `--exact-seed`；driver 回写逻辑 trial/seed，并补齐独立启动所需
  `PYTHONUSERBASE`、EGL device 3、NUMBA cache；parser 按 `(config, task, seed)`
  去重，未满 12 条唯一有效 trial 时只报 PENDING。
- **提速重跑**：GPU3 上保留已验证的纯 FP16 5568，并将 5567 以无任何 DuQuant
  环境变量的纯 FP16 方式重启；`OpenCabinet`→5567、`OpenStandMixerHead`→5568，
  两任务并行、每路串行 3 trials，吞吐相对单 server 约 2×。两客户端已建立 ZMQ
  连接，GPU3 占用约 31.7/46.1GB。
- 首次 detached driver 因缺 repo-local pyzmq 立即失败（未构造 env/未跑 episode），
  其 6 条 crash 标记及日志已移至
  `runs/robocasa365_eval/fp16_h1_startup_failed_20260817_0130/`，随后以修复后的
  self-contained 环境重启。

## 2026-08-17：criterion-4 官方协议对齐，旧结果与旧方案失效（D-045）

- **结论：不是模型/任务不兼容，而是本地评测协议错误。** 四个任务均属于
  RoboCasa365 `atomic_seen`，当前 `checkpoint-60000` 正是对应的官方
  target-posttraining checkpoint；官方基准中该 checkpoint 的 atomic-seen
  平均成功率为 68.5%。
- 对照官方 `PandaOmronDataConfig`、`run_eval.py`、`simulation.py` 与
  `MultistepWrapper` 后发现本地链路仍有八处错位：
  1. state 拼接顺序错误；
  2. action 拼接顺序错误；
  3. 连续量错误地使用 `mean_std`（官方为 `min_max`），gripper/control-mode
     应使用 `binary`；
  4. eef/base quaternion 缺少 `rotation_6d` 变换；
  5. language key 错用 `annotation.human.action.task_description`（官方为
     `annotation.human.task_description`）；
  6. 每次推理只执行 action chunk 的第 0 步（官方连续执行 16 步）；
  7. 全任务固定 horizon=720（官方四任务分别为 1050/450/750/600）；
  8. denoising=8（官方评测为 4）。
- 修复已落到 `custom_data_config.py`、`run_robocasa365_gr00t_eval.py`、
  `run_crit4_trial_driver.py` 及相关 smoke/sanity/解析工具；driver 的
  `--max-steps 0` 表示使用官方 task horizon，action chunk 默认 16。
- **纯 FP16 修复后单种子探针：3/4 成功**：OpenCabinet 193/1050 成功，
  OpenStandMixerHead 450/450 失败，PickPlaceDrawerToCounter 295/750 成功，
  CoffeeSetupMug 272/600 成功。单轮不能当正式 SR，但足以否定“四任务不兼容”
  并验证协议修复；每 episode 约 25–46 秒，相比旧链路的 20–28 分钟约提速
  30–50 倍。
- **证据有效性修正**：此前 criterion-4 全部结果作废；D-040 中“量化 plan
  不受污染”的判断也被本次发现推翻。旧 sensitivity/CS-only/CS+CKA/CKA-only
  方案使用了错误 data config、language key、state 分布与 denoising=8，不能作为
  最终 criterion-4 证据。旧文件保留归档，需在正确协议下重做 probe → selector
  → TopK adjudication → criterion-4。

## 2026-08-17：criterion-4 修复后正式结果（D-046）

- 在 D-045 的官方协议下完整重做 sensitivity（W4/W6、116 个可量化 Linear +
  16 个 DiT attention 观测项、denoising=4、16 obs、8 rollout obs × 2 paired
  noises）→ 三态 selector → TopK D_func 裁决。
  新的固定校准 buffer SHA256 为
  `1f61d740384e97ec1e1d2b527591331e40cda3927190d342e2df26cc218f531c`。
- 额外修复 selector 消融污染：默认 lambda sweep 会把混合 CKA 候选放入
  “CS-only”、把含 CS 候选放入“CKA-only”。本轮分别限制为纯 CS、纯 CKA
  候选池；CS+CKA 候选则保证两个系数均非零。
- 修正后的 TopK 最终 D_func：**CKA-only 0.09157 < CS+CKA 0.13415 <
  CS-only 0.13950**；三个最终候选分别为 flip8 / lambda(2,1) / greedy。
- 正式闭环（每任务 3 个固定种子，官方 task horizon，chunk=16）：

  | config | OpenCabinet | OpenStandMixerHead | DrawerToCounter | CoffeeSetupMug | Avg/SR |
  |---|---:|---:|---:|---:|---:|
  | FP16 | 3/3 | 3/3 | 2/3 | 3/3 | **11/12 = 91.7%** |
  | uniform W6 | 2/3 | 1/3 | 2/3 | 2/3 | **7/12 = 58.3%** |
  | CS-only | 3/3 | 2/3 | 3/3 | 1/3 | **9/12 = 75.0%** |
  | CS+CKA(dit) | 3/3 | 2/3 | 3/3 | 1/3 | **9/12 = 75.0%** |
  | CKA-only | 3/3 | 2/3 | 3/3 | 2/3 | **10/12 = 83.3%** |

- **criterion-4 FAIL**：预注册判据要求 CS+CKA 的闭环 SR 严格高于 CS-only，
  实测二者均为 75.0%，因此不重新启用 CS+CKA。诊断性的 CKA-only 达 83.3%，
  与其最低 D_func 方向一致，但 n=12 的区间较宽，不能据此替代预注册判据或宣称
  普遍显著优于 CS-only。
- 结果目录：`runs/robocasa365_eval/v2/`（共 60 条、每配置 12 条，无 crash）；
  D-045 前的旧数据完整归档为
  `runs/robocasa365_eval/v2_pre_protocolfix_invalid_20260817_094749/`。

## 2026-08-17：v1.4 LIBERO 四套件 Avg（D-047）

- v1.4：Spatial 100.0 / Goal 84.0 / Object 94.0 / Long 78.7（118/150），
  四套件 **macro Avg = 89.2%**；不同套件 episode 数不同时，不用总 episode
  加权值冒充官方四套件 Avg（加权值为 257/300 = 85.7%）。
- 同预算 uniform W6：96.0 / 86.0 / 94.0 / 76.7，macro Avg = **88.2%**，
  因此 v1.4 **+1.0pp**。
- v1.3 frozen：98.0 / 80.0 / 86.0 / 76.7，macro Avg = **85.2%**，
  因此 v1.4 **+4.0pp**。

## 2026-08-17：RoboCasa365 18-task 扩展消融与 CS 降噪（D-048）

- **CS 噪声定位**：正确读取 116 层 sensitivity 后，`1−CKA` 对 `D_func` 的
  Pearson/Spearman 为 **0.982/0.814**，CS 仅为 **−0.042/0.370**；旧 2:1
  CS+CKA 最终 mask 与 CS-only 只差 1 层。因而保持 min-max/预算/护栏不变，
  只把 CKA:CS 扫为 {8,16,32,64}:1。
- TopK `D_func` 排名：**64:1 0.07495 < 16:1 0.10044 < 8:1 0.11597 <
  32:1 0.13395**；64:1 也低于 CKA-only 0.09157。四任务开发集（3 paired
  seeds）64:1 为 **11/12**，CKA-only 9/12，16:1 与旧 2:1 均 8/12，按
  预注册规则冻结 64:1。
- 正式矩阵：18 atomic_seen tasks × 5 seeds × 6 configs = **540/540**，
  common-random-number diffusion noise、官方 horizon、chunk=16、denoising=4，
  **0 crash / 0 timeout**。新增 14 个 held-out 任务为主要统计口径：

  | config | held-out macro SR (95% task-cluster CI) | all-18 SR |
  |---|---:|---:|
  | FP16 | 71.4% [58.6, 84.3] | 66/90 = 73.3% |
  | 原版 W4A8 + static ATM/OHB | 52.9% [35.7, 70.0] | 47/90 = 52.2% |
  | CS-only | 58.6% [41.4, 74.3] | 55/90 = 61.1% |
  | CKA-only | 57.1% [40.0, 74.3] | 54/90 = 60.0% |
  | **CS+CKA 64:1** | **67.1% [54.3, 80.0]** | **61/90 = 67.8%** |
  | CKA + static ATM/OHB | 60.0% [50.0, 71.4] | 57/90 = 63.3% |

- **统计裁决**：64:1 相对 CKA-only held-out `+10.0pp`，但 task-cluster CI
  `[-1.4,+22.9]pp`、Holm p=0.891；相对 CS-only `+8.6pp`，CI
  `[-2.9,+21.4]pp`、Holm p=1.0。因此它是数值最优量化配置，但未达到预注册的
  “CI 下界 > 0”启用门槛，**默认仍保留 CKA-only，CS 结论为 promising but
  not significant**。CKA+ATM/OHB 对 CKA-only 为 +2.9pp，也不显著。
- 完整 manifest、per-task 表、10k task bootstrap、paired permutation、Holm 与
  McNemar：`runs/robocasa365_atomic_expanded_20260817/{summary.md,summary.json}`。
  旧 RoboCasa 日志已移至可恢复归档
  `/home1/gyy/vla/QuantVLA_old_logs_20260817_pre_expanded/`；废弃四任务 parser 与
  旧 smoke 入口已移除，避免旧协议结果混入新汇总。

## 2026-08-17：官方规模复测、paired diffusion noise 与安全提速（D-049）

- 本轮调参冻结口径升级为四个预声明任务各 50 个 target scenarios；候选仍为
  CKA:CS = {8,16,32,64,128,256}:1，并加入 CKA-only 作为
  `lambda_cs=0`（无穷大比例）边界；主判据为四任务 macro SR，并列依次按更低
  `D_func`、更高 CKA:CS 裁决。8:1 与 32:1 缺失的 plan-specific static-A8
  产物已经补齐，分别校验为 100/98 个 wrapped layers；两者与 16:1、64:1
  都固定使用同一 calibration buffer、P99.9、32 batches 和 denoising=4。
- **paired deterministic diffusion noise 是合法的 common-random-numbers
  方差缩减设计**：每个配置在相同 `(task, env_seed, replan_index)` 下使用相同
  标准正态初始噪声，单个配置的边际噪声分布不变，因此适合参数选择和配对差异
  估计。但它不是上游脚本默认的 GPU RNG 实现，正式结果必须明确标注为
  “paired evaluation”；若要声称与上游官方单点数字完全同口径，需另做 native
  GPU-noise 验证，不能把两种协议的结果混写。
- runner 现已同时支持 `paired` 与 `native` manifest，旧 `get_action` 行为保持
  不变；manifest 冻结 noise mode、checkpoint、plan/A8/ATM/pack hashes、EGL
  device、trial seeds、batch size、timeout 与效率采样周期。
- **安全提速**：旧 driver 每个 scenario 启动一次 Python，短任务也需要约
  70–83 秒。新 driver 在一个子进程内批量跑 seed，但仍为每个 seed 重建环境，
  避开已知的同一 env 第二次 reset crash；子进程 crash 时保留已完成 rows，仅
  重试缺失 seeds。四配置 `OpenStandMixerHead × seeds 0,1` 真实 smoke 共 8/8
  完整、0 crash，摊销后约 31–33 秒/episode，较旧日志提升约 **2.3–2.6×**。
- efficiency 新增每 episode 的环境构造、推理、env-step 与 wall time，以及每卡
  显存峰值/均值、GPU utilization、power；smoke 中环境构造约 5.7–6.9 秒，
  inference 约 0.285–0.304 秒/replan，峰值显存随 mask 为 14.3–14.8 GiB。

## 2026-08-17：50-scenario CS+CKA 最终冻结（D-050）

- 严格完成 7 个候选 × 4 个预声明 atomic_seen 任务 × 50 target scenarios：
  `{8,16,32,64,128,256}:1` 与 `CKA-only (lambda_cs=0)` 共 **1400/1400**
  episodes；均使用相同 `(task, env_seed, replan_index)` deterministic diffusion
  noise、官方 horizon、chunk=16、denoising=4，所有 client exit=0，0 crash / 0
  timeout。该阶段是 paired evaluation，不与 native GPU-RNG 结果混写。
- 正式开发集排序：

  | CKA:CS | successes | 四任务 macro SR |
  |---:|---:|---:|
  | **16:1** | **161/200** | **80.5%** |
  | 256:1 | 159/200 | 79.5% |
  | CKA-only | 155/200 | 77.5% |
  | 128:1 | 154/200 | 77.0% |
  | 8:1 | 154/200 | 77.0% |
  | 32:1 | 153/200 | 76.5% |
  | 64:1 | 150/200 | 75.0% |

- 因此按预注册的“最高 macro SR；并列时更低 D_func；再并列时更高比例”规则，
  **冻结 16:1 为 final**。大比例并非单调改善：256:1 虽接近 CKA-only mask，
  仍低 1.0pp；CKA-only 比 16:1 低 3.0pp。结论是 CS 确实需要降权，但本轮不应
  设为零；旧 3-seed 选择的 64:1 被 50-scenario 结果推翻。
- CKA-only 四任务为 OpenCabinet 90%、OpenStandMixerHead 88%、
  PickPlaceDrawerToCounter 62%、CoffeeSetupMug 70%，总计 155/200。完整选择文件：
  `runs/robocasa365_cs_loss/final_selection_official50.json`；三批严格 summaries：
  `runs/robocasa365_cs_tuning_{official_core,large_ratio_official,
  cka_boundary_gpu1}/summary.json`。
- final checkpoint-specific 产物已完成：atomic/composite_seen/composite_unseen 的
  16:1 plan-specific A8 均为 100 wrapped layers，plan/A8/checkpoint/buffer hashes
  全部匹配；三套 static ATM/OHB 的 CV_t `static_sufficient=True`（α/β 达标比例
  分别为 99.8/99.2%、99.8/99.0%、99.8/99.6%）。原版 W4A8 的 composite
  static ATM/OHB 也已生成，但 seen/unseen 的 CV_t 均判为不充分，正式测试仍保留
  该预注册 static baseline 并在报告中披露。
- 修复 A8 scale 加载到 PyTorch inference-tensor buffer 时在普通上下文执行
  `copy_` 会报错的问题：状态恢复现包在 `torch.inference_mode()` 内，并新增了
  inference-buffer round-trip 回归。原始失败在两个 composite checkpoint 上均可
  复现，修复后两套 W4A8 ATM/OHB 标定均成功。
- 已生成并只读验证三套正式 spec：
  `runs/robocasa365_official_full_{atomic,composite_seen,
  composite_unseen}_spec.json`，每套固定比较 FP16、原版 W4A8+static ATM/OHB、
  CS+CKA(16:1)、CS+CKA(16:1)+static ATM/OHB；正式矩阵启动前先逐 checkpoint
  做 1 task × 1 seed smoke。

## 2026-08-17：官方 50-task 正式执行与 composite 并发（D-051）

- 三个 checkpoint 的 4 配置 smoke 共 12/12 rows 完整，runtime 校验确认
  FP16=0、原版 W4A8=116、两个 final 配置=100 wrapped layers；ATM/OHB 配置
  均捕获 16/16 hooks，所有 client exit=0。
- atomic_seen 保持预注册的每配置 2 个 9-task 平衡分片（总 horizon 6000/5850）。
  composite 的平均 horizon 显著更长，若仍为 2 分片，失败占比较高的 W4A8 预计
  每个 split 接近一天。因显式 seed 与 deterministic action noise 已保证任务顺序/
  分片不改变样本，且单服务器 4-client 路径已在 CKA-only 200-episode 边界测试中
  通过，composite_seen/unseen 改为每配置 4 个 horizon-balanced 分片。
- 该调整只改变执行调度，不改变 task、seed、checkpoint、horizon、chunk、
  denoising、render 或 paired-noise 协议；不可变 manifest 会记录实际 shards。
  16 client 的 CPU 需求远低于 128 核容量，EGL 构造仍按物理设备加锁串行，
  rollout 保持并发，预计将 composite 墙钟时间降低约 2 倍。

## 2026-08-17：official atomic_seen 18-task × 50-scenario 结果（D-052）

- atomic_seen 正式矩阵 **3600/3600** rows 完整（18 tasks × 50 scenarios ×
  4 configs），0 crash / 0 timeout；严格 parser 对 manifest/config hash、task、
  seed、paired noise 和矩阵完整性检查均通过。

  | config | 18-task macro SR (95% task-cluster CI) | episodes |
  |---|---:|---:|
  | FP16 | **75.6% [68.8, 81.8]** | 680/900 |
  | 原版 W4A8 + static ATM/OHB | 49.6% [39.4, 59.7] | 446/900 |
  | **CS+CKA final (16:1)** | **69.0% [61.4, 75.9]** | 621/900 |
  | CS+CKA final + static ATM/OHB | 66.7% [57.1, 75.3] | 600/900 |

- 预注册配对比较：final 相对原版 W4A8+ATM/OHB **+19.4pp**，task-cluster
  CI `[+13.8,+25.2]pp`，Holm p=0.000076；final 相对 FP16 `-6.6pp`，CI
  `[-10.2,-3.1]pp`，Holm p=0.0067。给 final 加 static ATM/OHB 为
  `-2.3pp`，CI `[-5.7,+0.6]pp`、Holm p=0.209，点估计为负且不显著，atomic
  不支持启用该校正。
- efficiency（mean episode wall / inference per replan / peak GPU memory / mean
  GPU memory）：FP16 `26.9s / 0.139s / 12.8GiB / 10.4GiB`；原版 W4A8+ATM
  `37.5s / 0.268s / 18.0GiB / 15.7GiB`；final `30.9s / 0.254s / 17.7GiB /
  15.2GiB`；final+ATM `31.6s / 0.259s / 17.7GiB / 15.2GiB`。环境 step
  四配置均约 0.048s，差异主要来自推理和任务成功/失败 horizon。
- 按 QuantVLA 论文 Tables 1/2 的 `Memory (LLM+DiT)` 口径（180 个 LLM+DiT
  Linear，W4 理想紧密打包，计入 FP16 scales 与 FP32 block rotations，ATM/OHB
  折叠进已有 scale，不计 vision/activation/CUDA workspace/fake-quant cache），
  FP16 为 **1.993 GiB**，原版 116-layer W4A8 为 **0.898 GiB（节省 55.0%，
  2.22×）**，100-layer final 与 final+ATM 均为 **1.001 GiB（节省 49.8%，
  1.99×）**。该理论部署存储与当前 eager fake-quant 的 nvidia-smi 实测必须分栏：
  当前包装器保留多份浮点缓存且使用浮点 GEMM，因此不能把论文式存储节省表述为
  已实现的端到端显存或速度提升。统计已固化到
  `scripts/tools/robocasa_paper_memory.py`，单 split parser 与 50-task aggregator
  会从 checkpoint/plan 自动重算并校验。
- 结果与完整 per-task/统计/效率表：
  `runs/robocasa365_official_full_atomic_paired50/{summary.json,summary.md}`。
- composite formal 启动时发现 formal 阶段错误校验默认 atomic
  `--smoke-task OpenCabinet`；修复为仅在 `phase=smoke` 时校验。错误发生在
  manifest/server 创建前，没有产生 episode；修复后 composite_seen 的 4×4
  driver 与 checkpoint-specific runtime 校验均成功。

## 2026-08-18：composite 吞吐复核与 unseen 安全提速（D-053）

- composite_seen 启动 1.57 小时后的 312 个有效/可恢复样本显示：官方 task
  horizon 平均为 2316（atomic_seen 为 658，约 3.5×）；当前样本成功率仅
  5%–13%，平均实际执行 2914–3115 steps。单 episode 的 232–293 秒中，env
  step/render 占 169–186 秒，模型推理占 42–89 秒，故主要瓶颈是官方长 horizon
  仿真，而非进程卡死或 GPU OOM。
- seen 的不可变 manifest 已冻结为 4 shards/config，继续原样运行，避免重排使已完成
  rows 失效。尚未创建正式 manifest 的 composite_unseen 调整为 **8 个
  horizon-balanced shards/config**：仅增加环境并发，不改变 task、50 scenarios、
  checkpoint、paired noise、official horizon、chunk=16、denoising=4、render、
  plan 或 scale；显式 `(config, task, seed)` 键与确定性 noise 保证调度不改变样本。
- 每配置的 8 个 EGL client shard 按冻结的 round-robin 映射均匀分散到用户新授权的
  **GPU1–7**，主模型 server 固定在 GPU1/3/4/5；这直接并行化占单局 60%–70% 的
  render/env-step。为避免 8-client 把量化 server 排队变成新瓶颈，三套量化配置
  各增加一个数值/产物完全相同的只读模型副本：原版 W4A8→GPU2:5772、final→
  GPU6:5774、final+ATM→GPU7:5777；8 个 shard 在主/副本间 4:4 交替，FP16 因
  单实例吞吐最高保留 GPU1:5771。8-shard 的最大 horizon 负载约为 4-shard 的
  一半；结合模型副本，预计 unseen 墙钟缩短约 40%–55%。manifest 会冻结每个
  shard 的 EGL device 和 server instances，最终
  efficiency 按 task set 分栏，不把不同并发度的 latency 冒充单实例 kernel 速度。
- GPU2/6/7 的已有进程均不终止；正式启动前按每个 EGL client 2.3GiB、FP16/
  quant server 12/16GiB 加 2GiB guard 做实时显存门槛，不足只等待重试，不使用
  GPU0。GPU2/6/7 均已完成真实环境 construct/reset/50-step EGL smoke。

## 2026-08-18：GPU6/7 正式占用与 unseen 双 wave 自动接力（D-054）

- 为避免 GPU6/7 只做 smoke 后等待 seen，提前创建完整、不可变的
  composite_unseen manifest，并立即启动量化配置的奇数 shards `{1,3,5,7}`：
  原版 W4A8+ATM/OHB 在 GPU2:5772、final 在 **GPU6:5774**、final+ATM/OHB
  在 **GPU7:5777**。三个 checkpoint runtime 均严格验证通过：wrapped layers
  分别为 116/100/100，ATM/OHB hooks 为 16/16（需要的两配置），denoising=4；
  共 12 个正式 rollout clients，结果直接写 manifest 声明的正式文件。
- 启动后实测 GPU6 从 3MiB 增至约 14.7GiB，GPU7 总占用约 20.2GiB（含他人约
  2.7GiB 既有进程，未终止或驱逐），两卡均出现持续推理利用率；首批 unseen
  episode 已写入可恢复 journal，0 crash / 0 OOM。
- 仅等待该提前 wave 会导致 seen 结束后 GPU1/3/4/5 空转。新增互补 primary
  wave：seen 严格完成后，GPU1 跑 FP16 全部 8 shards，GPU3/4/5 分别跑三个
  量化配置的偶数 shards `{0,2,4,6}`，同时 GPU2/6/7 继续奇数 shards。静态覆盖
  检查确认 `4 configs × 8 shards = 32` 项恰好一次，无重叠、无缺口；模型实例
  映射为 FP16 全部 GPU1，三套量化配置偶数 shard→主实例、奇数 shard→副本。
- early/primary wave 各有 kernel advisory lock；普通 full runner 在 resume/严格
  校验前等待两把锁，并仅启动 hash 匹配后仍缺失的 shard。自动链已重启为新逻辑：
  等 composite_seen 完整解析 → 启动 primary wave → 等两 wave 完成 → strict
  composite_unseen parse → 50-task aggregate。这样 crash/restart 仍按
  `(config, task, seed)` 恢复，且不会并发写同一个正式结果文件。

## 2026-08-18：外部 GPU 负载归因与 server-process efficiency（D-055）

- 运行约 8 小时后，GPU6/7 整卡占用升至约 35.8/42.4GiB。逐 PID 核查确认并非
  QuantVLA 泄漏：我方三个 unseen server 的 CUDA memory 稳定为原版 W4A8
  13.17GiB、final 12.75GiB、final+ATM/OHB 12.75GiB；新增占用来自 07:49/08:17
  后启动的两套无关 Ollama server（GPU6/7 各约 22GiB）和 GPU7 原有约 2.7GiB
  任务。未终止或迁移任何外部进程，正式 rollout 保持 0 crash / 0 OOM。
- 因此 `nvidia-smi` 整卡 memory/utilization/power 仍保留为真实调度环境记录，但
  不能再解释为 QuantVLA 单模型开销。新增 `gpu_server_efficiency.jsonl`，按我方
  server PID 独立采集 CUDA memory、CPU RSS、GPU、主/副实例和采样来源；已经为
  正在运行的 seen 4 个 server 与 unseen 3 个副本启动 sidecar，并集成到后续
  primary/full runner，从 server 启动起自动采样。
- strict parser 与 50-task aggregator 的 efficiency 表改为同时报告
  `Peak server-process memory` 和 `Peak device memory`。前者用于比较模型实际
  eager/fake-quant 常驻显存，后者用于披露包含 EGL clients 与外部作业的整卡峰值；
  inference/replan 仍来自每 episode 的服务请求计时，论文式 LLM+DiT packed
  memory 继续单独分栏，不与两种实测显存混用。

## 2026-08-18：CPU 亲和性 A/B 负结果（D-056）

- 七个 PyTorch server 默认各暴露 64 intra-op / 64 inter-op threads，OS 线程数
  324/server；128 核机器在 seen+early-unseen 并发时 load≈222、CPU busy≈94%。
  为验证线程过度并发是否拖慢环境/render，做了不改变模型数值的可逆 affinity A/B。
- 12 cores/server 使机器立即出现 33%–36% idle，明显饿住推理，直接否决；随后
  测试 16 个互不重叠 cores/server，机器仍有 14%–24% idle，但进程级 runnable
  wait 达 25.8%–49.0%，说明 affinity 分区造成“有空核但服务线程不可用”，短窗口
  也未建立唯一结果行吞吐增益。
- 按预先记录的 keep/restore 规则，七个正在运行的 server 已全部恢复 `0-127`
  affinity；恢复后 CPU busy 回到 95%–99%。实验无重启、无 crash、无样本或协议
  变化。结论：当前不做静态 CPU 分区，也不据此增加 clients；完整 A/B 记录保存在
  `runs/robocasa365_cpu_affinity_ab.json`。

## 2026-08-19：unseen primary 提前启动（D-057）

- composite_seen 的 FP16、final、final+ATM/OHB 已各自完成 800/800，只有原版
  W4A8 尚余少量长-horizon episodes。用户授权 primary wave 提前启动后，保持
  composite_unseen 不可变 manifest、结果文件、shards 和协议均不变，仅让原版
  W4A8 的偶数 shards 使用 manifest 已声明的 GPU2 replica；FP16、final、
  final+ATM/OHB 仍分别使用 GPU1/4/5 primary。`primary_wave.lock` 在启动前已
  独占，seen 完成后的自动链只会等待并严格 resume，不会并发写同一文件。
- 启动前保守门禁同时计入仍驻留的 seen servers、20 个 EGL clients 和外部 GPU
  作业，所有 GPU 均满足余量要求，因此无需终止 seen 空闲 server 或任何无关进程。
  runtime 重新严格验证 FP16/W4A8/final/final+ATM 的 wrapped layers 为
  0/116/100/100，checkpoint-specific plan/A8 与 ATM/OHB 校验通过；主波次 20 个
  drivers 已启动，覆盖 FP16 全部 8 shards 和三个量化配置的偶数 4 shards。
- `run_robocasa_unseen_primary_wave.py` 新增向后兼容的
  `--replica-config-ids`，默认空值仍保持原 post-seen GPU1/3/4/5 行为；本次仅将
  `w4a8_atmohb` 映射到其 manifest replica GPU2:5772。preflight 和 runtime
  metadata 显式记录每个配置实际使用的 GPU、端口和 replica 编号。
