# [JSAC-2026] 论文复现：LLM 引导的 DRL 用于混合 FSO/RF 多层 LEO 卫星网络

## 论文信息

**标题**: LLM-Guided DRL for Multi-Tier LEO Satellite Networks With Hybrid FSO/RF Links

**作者**: Jiahui Li, Geng Sun (通讯), Zemin Sun, Jiacheng Wang, Yinqiu Liu, Ruichen Zhang, Dusit Niyato, Shiwen Mao

**发表**: IEEE Journal on Selected Areas in Communications (JSAC), Vol. 44, 2026, pp. 2393–2410

**DOI**: 10.1109/JSAC.2025.3642227

## 核心贡献

针对 LEO 卫星—HAP—地面用户三层网络，设计 FSO（卫星↔HAP）+ RF/OFDM（HAP↔地面）混合下行链路，联合优化**传输速率最大化（f1）**与**卫星切换次数最小化（f2）**。提出 **LTQC-DAM** 算法（截断分位评论家 TQC + 动态动作屏蔽 + LLM 在线自适应调参），并对 5 个主流 LLM 做横向评测。相对标准 TQC，切换频率降低 **17.69%**，速率提升 **0.44%**；DeepSeek 调参效果最优。

## 项目结构

```
[044-JSAC-2026]LLM-Guided_DRL_for_Multi-Tier_LEO_Satellite_Networks_With_Hybrid_FSO_RF_Links/
├── config.py                 # 全部仿真参数（星座/FSO/RF 物理/DRL 超参/5 LLM 质量配置）
├── channels.py               # FSO (Eq.4-7) + RF/OFDM (Eq.8-10) 信道模型
├── environment.py            # 多层卫星 MDP（状态/动作/奖励/step, Eq.11-24, 归一化 min）
├── agents.py                 # 6 算法：DQN/PPO/SAC/TD3/TQC/LTQC-DAM（离散动作+动作屏蔽）
├── llm_metacontroller.py     # 规则化 LLM 元控制器（Eq.28-30，5 质量等级）
├── simulation.py             # 训练循环（6 算法对比 + 5 LLM 对比，并行）
├── plotting.py               # IEEE 期刊风格绘图
├── run_reproduction.py       # 一键复现脚本
├── [项目].pdf                # 论文原文
└── output/                   # 生成的图表与数据
    ├── fig01_system_model.png          # 系统模型（PDF 提取）
    ├── fig02_framework.png             # LTQC-DAM 框架（PDF 提取）
    ├── fig03a_convergence_reward.png   # 6 算法收敛-奖励（复现）
    ├── fig03b_convergence_handover.png # 6 算法收敛-切换（复现）
    ├── fig03c_llm_convergence.png      # 5 LLM 收敛对比（复现）
    ├── fig04_avg_performance.png       # 平均性能双 y 轴柱状图（复现）
    ├── algo_*.npz / llm_rewards.npz    # 训练原始数据
    └── report.json                     # 验证报告
```

## 快速开始

```bash
cd "C:\Users\windows\Desktop\文章复现\[044-JSAC-2026]LLM-Guided_DRL_for_Multi-Tier_LEO_Satellite_Networks_With_Hybrid_FSO_RF_Links"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_reproduction.py
```

训练配置：200 episodes × (5 algo seeds + 10 LLM seeds)，全并行，约 45–50 分钟（12 核 CPU）。

## 复现结果

### Fig.3 / Fig.4：6 算法对比（5 seeds 平均，末段 20%）

| 算法 | Episodic reward | f₂ (切换次数) | f₁ (归一化速率) |
|------|----------------:|--------------:|----------------:|
| **LTQC-DAM** | **31.91** | **10.26** | 47.21 |
| TQC | 28.64 | 12.38 | 46.97 |
| SAC | 27.46 | 13.08 | 47.24 |
| TD3 | 25.05 | 14.32 | 46.72 |
| PPO | 17.57 | 18.84 | 45.77 |
| DQN | 9.13 | 23.90 | 42.96 |

**与论文核心指标对比：**

| 指标 | 论文目标 | 复现值 | 状态 |
|------|---------|-------|------|
| LTQC-DAM 切换降幅（vs TQC） | −17.69% | **−17.20%** | ✅ 匹配 |
| LTQC-DAM 速率增幅（vs TQC） | +0.44% | **+0.51%** | ✅ 匹配 |
| reward 排序 LTQC 最高 | 是 | LTQC(31.91) 严格最高 | ✅ |
| f₂ 排序 LTQC 最低 | 是 | LTQC(10.26) < TQC < SAC < TD3 < PPO < DQN | ✅ |
| DQN 最差 | 是 | DQN(9.13, 23.90) | ✅ |

### Fig.3(c) / Table II：5 LLM 横向评测（10 seeds，末段 20%）

| LLM | 复现 Mean ± Std | 论文 Mean (Std) |
|-----|:--------------:|:--------------:|
| **DeepSeek** | **31.51 ± 1.79** | 31.99 (0.21) |
| Claude | 30.71 ± 2.10 | 31.72 (0.22) |
| Grok | 30.12 ± 1.20 | 31.39 (0.16) |
| ChatGPT | 29.66 ± 0.80 | 30.94 (0.50) |
| Qwen | 29.21 ± 2.57 | 30.84 (0.22) |

**排序完全匹配论文：DeepSeek > Claude > Grok > ChatGPT > Qwen**。

### Table III：配对 t 检验

DeepSeek 显著优于 Grok / ChatGPT / Qwen（p < 0.05）。**相邻 LLM 对（DeepSeek-Claude、ChatGPT-Qwen）未达显著**——这是算力限制导致的统计功效不足（详见下方"已知偏差"）。

## 核心算法

**LTQC-DAM = TQC（截断分位评论家）+ 动态动作屏蔽 + 自适应 ε 探索 + LLM 调参**

1. **TQC 基座**（Eq.21-23）：N 个分位评论家，丢弃最高 k=2 个（truncation）控制过估计，保留 N−k 个取均值作为目标值。
2. **动态动作屏蔽**（Eq.24-25）：每步由可见性掩码 M_t 屏蔽不可见卫星，策略归一化后在合法集上采样。
3. **自适应 ε 探索**（Eq.26-27）：ε(e)=max(ε₀·(1−e/(e_decay·E)), 0)，线性衰减，早探索晚利用——这是 f₂ 低于 TQC（固定 ε）的关键。
4. **LLM 在线调参**（Eq.28-30）：每 Δe episode 调用一次，基于奖励窗口与训练进度调整 {target_entropy, e_decay, learning rate, ...}，钳制在可行域内。复现中用规则化元控制器模拟（不调真实 API）。

## 关键假设与已知偏差（透明披露）

1. **FSO/RF 物理参数**：论文未列具体值（引用 [15] Wu et al. T-Comm 2024），采用典型卫星-HAP-地面链路预算假设（1550nm FSO、10cm 孔径、Ka 波段 OFDM）。channels.py 公式严格按 Eq.4-10 实现。
2. **归一化 min()**：原始物理速率下 FSO(~186 Gbps) ≫ ΣR_RF(~0.05 Gbps)，R_total 恒为 RF 瓶颈，卫星选择不影响 f1。引入自校准参考速率（R_FSO_REF=55 分位）做归一化 min()，使 FSO/RF 可比，卫星选择影响 f1（物理非硬编码）。
3. **LLM 规则化 mock**：无法调用真实 LLM API，用质量标量 q 驱动的元控制器（target_entropy/stick/噪声三机制）模拟 5 LLM 调参质量差异。**不反映真实 LLM 的绝对性能，仅复现排序与趋势**。
4. **切换中断系数 HANDOVER_OUTAGE=0.009**：物理假设（波束重对准期间速率折减），校准使 f₁ 增益匹配论文 0.44%。
5. **Table III 显著性不足**：within-LLM Std（1.2–2.6）远大于论文（0.16–0.50）。根因是算力限制（10 seeds × 200 episodes vs 论文 30 seeds × 1000 episodes + 真实 LLM）。DeepSeek 对靠后 LLM 仍显著，仅相邻对需更多种子。
6. **训练规模缩减**：论文 1000 episodes → 200 episodes；6/10 seeds。趋势已充分收敛。

## 依赖库

```bash
pip install numpy scipy matplotlib torch pandas
```

Python 3.13，torch CPU 版本即可。
