# [045-TWC-2026] 分层强化学习驱动的密集 LEO 星座切换复现（真实 RL 训练）

## 论文信息

**标题**: Learning When and Where to Handover: A Hierarchical Reinforcement Learning Framework for Dense LEO Satellite Constellations

**作者**: Di Zhao, Yiqun Wang, Bin Song (通讯), Yejun Zhou, Pengfei Qin

**发表**: IEEE Transactions on Wireless Communications (TWC), Vol. 25, 2026, pp. 12787–12801

## 复现方法：前向物理仿真 + 真实 RL 训练（PyTorch，CPU）

本复现构建前向物理仿真器（真实轨道几何+信道+负载动态），并在其上**真实训练论文的分层 RL**：
- **空间层 DQN**（`dqn_spatial.py`，Eq.30-39）：FC(256,128,128,64)+掩码输出，replay buffer+target 网络+ε-greedy，lr=1e-3。真实训练。
- **时间层 PPO+GRU**（`ppo_temporal.py`，Eq.16-29）：GRU(256,256,128) actor + FC critic，PPO clip 0.2/GAE/γ=0.99，lr=3e-4。真实训练。
- 训练用简化 env（20用户/200s，~2.5s/episode），评估在 100用户/600s 完整 env。

复用模块：`constellation.py`(Walker-delta+开普勒/J2)、`channel.py`(FSPL/莱斯/阴影/SINR/Shannon)、`environment.py`(Poisson业务/负载)、`handover.py`(qos_utility Eq.20/触发逻辑/成败判定，含 policy_fn/temporal_fn 钩子)。

## 项目结构

```
[045-TWC-2026].../
├── constellation.py / channel.py / environment.py / handover.py   # 前向物理仿真
├── rl_env.py            # SpatialEnv(DQN) + TemporalEnv(PPO+GRU)
├── dqn_spatial.py       # 空间层 DQN（Eq.30-39）
├── ppo_temporal.py      # 时间层 PPO+GRU（Eq.16-29）
├── train_rl.py          # run_fig4(PPO 3配置) + run_fig5(DQN 4配置)
├── eval_rl.py           # 训练后策略评估 Fig.6/7/9（+分层 MO-HRL）
├── simulation.py / plotting.py / run_reproduction.py
├── output/              # fig01-03场景 + fig04/05训练曲线 + fig06/07/09评估 + rl_ckpt/
└── README.md
```

## 快速开始

```bash
cd "C:\Users\windows\Desktop\文章复现\[045-TWC-2026]..."
python run_reproduction.py     # 全流程：训练(DQN+PPO) + 评估 + 出图，约2-4小时 CPU
# 或分阶段：python train_rl.py（Fig.4/5）→ python eval_rl.py（Fig.6/7/9）
```

无 GPU 时在 12 核 CPU 上训练；`train_rl.py --quick` 用 100 episode 快速验证收敛。

## 复现结果（真实训练涌现）

### Fig.4 — 时间层 PPO+GRU 训练收敛（3 奖励配置）✓ 真收敛
| 配置 | 早期→末期 reward | 增益 |
|------|-----------------|------|
| case2（仅必要切换奖励）| 2893→2918 | +0.9% |
| case23（+改善切换奖励）| 3058→3186 | +4.2% |
| case234（+乒乓惩罚，完整）| 2736→3009 | **+10.0%** |

三曲线全上升（学习发生）。**case234 学习增益最大**（与论文"完整奖励学到最优切换时刻"定性一致）；PPO 学到的强抑制乒乓：切换次数 1773→345。诚实偏差：绝对 reward 排序 case23>case234（Case4 惩罚压低 case234 基线），非逐点匹配论文。

### Fig.5 — 空间层 DQN 训练收敛（4 权重配置）✓ 真收敛
| 权重 (L,H,τ) | 早期→末期 reward |
|--------------|-----------------|
| (0.6,0.2,0.2) | 629→778 |
| (0.2,0.6,0.2) | 600→823 |
| (0.2,0.2,0.6) | 583→746 |
| (0.4,0.3,0.3) | 621→794 |

四曲线全上升（+149~+223），真实训练收敛。

### Fig.6/7/9 — 训练后策略评估（涌现）
用训练后 DQN（+分层 PPO+GRU for MO-HRL）在 100用户完整 env 上评估：
- Fig.6 成功率 vs 信号/负载/服务时间：趋势方向正确（a/c 增、b 减），动态范围 0.33-0.99。
- Fig.7 纬度区成功率：MO 排序正确。
- Fig.9 可见卫星数→速率：递减趋势正确，量级 197-246 Mbps（论文 235-250 ballpark）。

Fig.6/7/9 的 5 算法对比统一用训练后的空间层 DQN 选目标（时间层触发用规则），保证 5 算法评估口径一致；时间层 PPO+GRU 的真实训练成果由 Fig.4 体现（学到切换次数 1773→345，强抑制乒乓）。`run_reproduction.py` 另跑分层 MO-HRL（PPO+GRU+DQN）作控制台验证，不另存图。

## ⚠️ 诚实保真度边界

论文未公开种子/代码，**即使真实训练也不可能逐点一致**：
1. Fig.4/5 复现"reward 上升到平台"的**收敛形状与学习增益**；绝对 reward 值与 case 排序可能不同（reward 标度/env/随机种子差异）。
2. Fig.6/7/9 用训练后策略评估，趋势与算法排序正确，绝对数值不精确匹配论文。
3. 时间层 PPO+GRU 在 20用户简化场景训练、100用户完整场景评估，存在训练-评估用户数失配（分层 MO-HRL 在中高纬多可见星场景泛化偏差偏大），已诚实标注。
4. 目标=**真实训练 + 定性一致**（收敛动态 + 算法分离 + 趋势），非逐点一致。所有 reward/transition 由前向物理仿真涌现，无查表/硬编码输出。

## 演进说明（方法论纠错记录）
本复现经历三版，体现 RL 论文复现的方法论纠错：
1. **v1 端点锚定插值**（已弃用）：硬编码论文输出值查表——输出即输入、不构成验证，**非复现**。
2. **v2 ψ-贪心规则代理**：前向物理仿真 + 贪心决策，结果涌现，但无训练曲线、算法分离弱。
3. **v3 真实 RL 训练**（当前）：在 v2 仿真器上真实训练 PPO+GRU + DQN，复现 Fig.4/5 训练曲线 + 训练后策略评估。

## 依赖库

```bash
pip install numpy matplotlib scipy torch
```
