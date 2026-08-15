# QuantVLA LIBERO 完整复现结果

日期: 2026-08-10 | 硬件: 8× NVIDIA H20 | 协议: GR00T 官方 run_libero_eval.py (10 任务 × 5 trials = 50 条/套件, denoising=8, headless)
配置: GR00T N1.5 posttrain + DuQuant W4A8 (block=64, calib=32, ls=0.15, LLM全量+DiT MLP = 116层) + ATM + OHB

## 主结果 (成功率, 每套件 50 条)

| 套件 | 论文 QuantVLA (Table 2) | 我们量化 R1 | 我们量化 R2 | 我们 FP16 | 论文 FP16 |
|---|---|---|---|---|---|
| Spatial | 96.0% | 96.0% | 96.0% | 96.0% | 92.0% |
| Object | 92.0% | 94.0% | 90.0% | 94.0% | 92.0% |
| Goal | 90.0% | 88.0% | 88.0% | 88.0% | 86.0% |
| Long | 74.0% | 70.0% | 70.0% | 80.0% | 76.0% |
| **Avg** | **88.0%** | **87.0%** | **86.0%** | **89.5%** | **86.5%** |

## 结论

1. **复现成功**：量化版结果与论文 Table 2 (+QuantVLA W4A8, 96/92/90/74) 高度一致,
   全部差异 (±2-4%) 在 50 条样本的统计误差内 (σ≈4%)。
2. **量化精度损失小**：同协议下量化 vs FP16 损失约 2.5-3.5% (论文: 88 vs 86.5, 量化略优)。
3. **Long 套件最敏感**：长序列累积误差是量化主要损失点 (70% vs FP16 80%)，与论文模式一致。
4. 论文 Table 1 (无 ATM/OHB 消融): 116层方案 avg 82.5% → 加 ATM+OHB 后 88% (Table 2)，
   验证了 ATM/OHB 的核心贡献（+5.5%）。

## 资源与复现
- 模型: /home/wubohan/llms/vla/models/ (GR00T-N1.5-3B + 5× libero posttrain)
- 数据: /home/wubohan/llms/vla/data/ (5× LIBERO LeRobot + RoboCasa 24子集 + 教程数据)
- 环境: groot_test (torch 2.5.1+cu124, flash-attn 2.7.1.post4 自编译 sm_90) + libero_test (torch 2.9.0+cu128)
- 评估日志: /tmp/eval_fp16_*.log (FP16), /tmp/eval_*.log (量化)
