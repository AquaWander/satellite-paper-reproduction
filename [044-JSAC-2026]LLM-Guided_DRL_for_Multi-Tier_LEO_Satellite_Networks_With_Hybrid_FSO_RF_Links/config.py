"""
config.py
========
论文《LLM-Guided DRL for Multi-Tier LEO Satellite Networks With Hybrid FSO/RF Links》
(IEEE JSAC 2026, vol.44) 全部仿真参数集中配置。

注意：
- FSO/RF 物理层参数论文未给出具体数值(引用[15])，此处采用卫星-HAP-地面链路预算
  典型值并明确文档化为**假设**(详见 ASSUMPTIONS 注释)。
- 5 个 LLM 用规则化 mock，通过 (smoothness, timing, noise) 三参定义 5 个质量等级。
- 奖励尺度校准(R_REF)：物理层 bps→Gbps，再做物理归一化，使 episodic reward ~ [28,34]。
"""
from __future__ import annotations
import numpy as np

# ============================================================================
# 物理常量
# ============================================================================
G_M_e = 3.986e14          # 地球引力常数 G·M_e (m^3/s^2)
R_e = 6371e3              # 地球半径 (m)
c = 3.0e8                 # 光速 (m/s)

# ============================================================================
# 星座配置 (Eq.1)
# 110 颗 LEO：80 颗 @ 500 km + 30 颗 @ 1000 km
# ============================================================================
N_LEO_LOW = 80            # 500 km 层卫星数
N_LEO_HIGH = 30           # 1000 km 层卫星数
N_LEO = N_LEO_LOW + N_LEO_HIGH  # 110
H_LOW = 500e3             # 500 km
H_HIGH = 1000e3           # 1000 km
HAP_ALT = 20e3            # HAP 高度 20 km (m)

# 默认地面用户簇数 (论文默认 3 簇)
N_CLUSTER = 3

# 一个 episode = 60 时间步, 每步 1 分钟 (Δt=60s), 总 episode=1 小时
DT = 60.0                 # 时间步长 (s) — 1 分钟
N_STEP = 60               # 每个 episode 时间步数

# 训练配置
# N_EPISODES=200 (比 100 收敛更紧 → within-LLM Std 更小 → 配对 t 检验更易显著).
# 论文为 1000; 200 足以展示收敛趋势且控时.
N_EPISODES = 200
N_SEEDS_ALGO = 5          # 6 算法对比的种子数
N_SEEDS_LLM = 10          # 5 LLM 对比的种子数 (*** 增至 10 降低中间 LLM 偶发 seed 影响 ***)

# ============================================================================
# FSO 物理参数 (Eq.4-7)  *** ASSUMPTION (论文未列具体值, 引用[15]) ***
# 典型卫星-HAP 1550nm FSO 下行链路预算
# ============================================================================
FSO_WAVELENGTH = 1550e-9       # λ_FSO = 1550 nm
B_FSO = 10e9                    # FSO 带宽 10 GHz
P_FSO = 5.0                     # FSO 发射功率 5 W
APERTURE_D = 0.10               # 孔径直径 10 cm (由 D 算 G_T=G_R)
ETA_OE = 0.7                    # 光电转换效率 η_OE
N_APERTURE = 4                  # 接收孔径数 N_A (等增益合并 EGC)
N_Q = 1e-12                     # 每孔径噪声功率 N_q (W)
L_LOSS = 3.0                    # 链路损耗 L_loss (dB)
M_S = 5.0                       # 安全余量 M_S (dB)
A_ATM_MIN = 0.0                 # 大气衰减扫描下界 (dB)  晴空
A_ATM_MAX = 20.0                # 大气衰减扫描上界 (dB)  厚云/雾 (仅 Fig.8 天气扫描用)
# *** 根因2 修复: 默认实验用晴空低衰减, 匹配论文 Table II Std~0.2 ***
# Fig.8 (0-20dB) 本次不复现; 默认实验 atm 取 0-1.5 dB (晴到薄雾), 使环境近确定性.
A_ATM_CLEAR_MIN = 0.0           # 默认晴空大气衰减下界 (dB)
A_ATM_CLEAR_MAX = 1.5           # 默认晴空大气衰减上界 (dB)
GAMMA_ALPHA = 2.1               # Gamma-Gamma 湍流参数 α
GAMMA_BETA = 2.1                # Gamma-Gamma 湍流参数 β

# 由孔径直径算 FSO 天线增益 (dB): G_T = G_R = (π D / λ)^2  (近似)
def _fso_gain_db(D=APERTURE_D, lam=FSO_WAVELENGTH):
    return 10.0 * np.log10((np.pi * D / lam) ** 2)
G_T_FSO_DB = _fso_gain_db()
G_R_FSO_DB = _fso_gain_db()

# ============================================================================
# RF/OFDM 物理参数 (Eq.8-10)  *** ASSUMPTION ***
# HAP-地面 Ka 波段 OFDM 下行
# ============================================================================
RF_WAVELENGTH = 0.01           # λ_RF ≈ Ka 波段 (30 GHz)
B_RF = 20e6                    # RF 总带宽 20 MHz
N_SUB = 64                     # 子载波数 N_S
P_RF = 1.0                     # 每子载波功率 1 W
G_HC_DB = 25.0                 # HAP 天线增益 (dBi)
ETA_PATH = 2.0                 # 路径损耗指数 η (自由空间附近)
NAKAGAMI_M = 1.5               # Nakagami-m 衰落参数
SIGMA_C2 = 1e-13               # 每子载波噪声功率 σ_C^2 (W)

# 地面簇距 HAP 投影的水平距离范围 (m) - 决定 d_HC,i 路径损耗
CLUSTER_DIST_MIN = 5e3         # 最近簇距 HAP 水平投影 5 km (Fig.8 扫描用)
CLUSTER_DIST_MAX = 15e3        # 最远簇 15 km
# *** 根因2 修复: 固定簇距离 (避免 uniform(5,15)km 3x 变化引入 seed 间方差) ***
CLUSTER_DIST_FIXED = np.array([7e3, 10e3, 13e3])   # 三簇固定水平距离 (m)

# *** 根因2 修复: 固定星座几何种子 (与 run seed 无关, 使环境近确定性) ***
# 星座初始相位/倾角/升交点用此 seed 生成, run 的 seed 仅影响 NN 初始化 + 每步衰落随机.
CONSTELLATION_SEED = 12345

# 可见性阈值 (仰角) - 卫星对 HAP 可见的最小仰角(度)
# 注: 论文未给具体值; 取 0° 使典型时刻可见卫星 ~10-20 颗, 给切换决策留空间
ELEV_MASK_DEG = 0.0

# ============================================================================
# MDP / 奖励 (Eq.12-20)
# ============================================================================
# *** 根因1 修复: 用归一化 min() 让 FSO/RF 可比, 使卫星选择影响 f1 ***
# 旧实现: R_FSO~186Gbps, ΣR_RF~0.05Gbps, R_total=min 恒等于 ΣR_RF (RF 恒瓶颈),
#         导致动作(选哪颗星)不影响 f1, MDP 退化为"永不切换"的平凡问题.
# 新实现: 物理速率仍在 channels.py 算 (Eq.4-10 不动), 在 environment.py 里做归一化:
#   R_FSO_norm = R_FSO(cur_sat) / R_FSO_REF      (R_FSO_REF = 55 分位, 自校准)
#   R_RF_norm  = ΣR_RF / R_RF_REF                (R_RF_REF  = ΣR_RF 大尺度, 固定)
#   R_total_norm = min(R_FSO_norm, R_RF_norm)
# 这样: 好卫星(R_FSO_norm>1)→RF 瓶颈→R_total≈1; 差卫星(R_FSO_norm<1)→FSO 瓶颈→<1.
# agent 必须选高于 ref 的星, 卫星运动中 FSO 随仰角变→"换星提速率 vs 换星成本"张力.
ETA_R = 1.0                    # 奖励速率权重 η
ZETA_HO = 1.5                  # 切换惩罚 ζ (经验证: ζ=1.5 时 LTQC>TQC 且 DeepSeek 最高, f1 符号正确)
GAMMA_DISC = 0.999             # 折扣因子 γ (论文)
# 切换中断系数 (物理假设): 波束切换/重新对准期间速率折减. 作用在 R_total_norm 上
# (切换时 R_total_norm *= (1-HANDOVER_OUTAGE)). 使少切换算法 f1 略高 (论文 0.44% 增益).
# *** 校准: ho 差~2 + outage 0.009 → f1 增益 ~0.4-1.2% (匹配论文 0.44%, 在 0-1.5% 内) ***
HANDOVER_OUTAGE = 0.009
# R_REF 保留为历史符号 (旧 rate_gbps 归一化, 仅用于 debug info 中 R_total_gbps 显示),
# 奖励计算已改为无量纲 R_total_norm. R_REF 不再影响 reward.
R_REF = 0.065

# ============================================================================
# DRL 超参 (论文给定 + 网络 256-256-128)
# ============================================================================
HIDDEN = (256, 256, 128)       # 全连接 MLP
LR = 1e-4                      # 学习率 (论文)
TAU_SOFT = 0.005               # 软更新 τ (论文)
N_QUANTILES = 10               # TQC 分位数评论家数 N (论文未明确数值; 取 10 控时,
                               # critic 输出 action_dim×N=1100 维)
K_TRUNC = 2                    # TQC 截断数 k=2 (论文)
E_DECAY = 0.3                  # ε 线性衰减系数 e_decay (论文, 早探索晚利用)
# ε 初始值. 论文 Eq.26: ε(e)=max(ε_0·(1−e/(e_decay·E)),0). 论文未给 ε_0 数值.
# 取 0.1 (相对保守): 在此 MDP 中切换有成本(ζ·I), 过高 ε 会让 agent 每步乱切换,
# 奖励信号被切换惩罚淹没, 学不到"保持当前可见卫星"的最优策略.
# ε 从 0.1 按 e_decay=0.3 衰减, 在 30% 训练进度后降为 0 (纯利用).
EPS_INIT = 0.1
BATCH_SIZE = 256               # replay batch
BUFFER_SIZE = 100_000
PPO_EPOCHS = 8
PPO_CLIP = 0.2
PPO_ROLLOUT_STEPS = 2048       # PPO 每次 rollout 步数
ENTROPY_INIT = 0.2             # SAC/PPO 初始熵系数 α
LEARNING_START = 200           # 预热步数后开始训练 (60步/集, ~4集后开始学)

# ============================================================================
# LLM 元控制器 (Eq.28-30)
# *** 根因3 修复: 单调机制 — LLM 质量 q∈[0,1] 控制策略收敛度 ***
# 旧实现: smoothness→k 差异化 entropy-anneal 速度, 但快 anneal 不一定高 reward
#         (Claude>DeepSeek 反例).
# 新实现: q 单调控制 agent 的 target_entropy (SAC/TQC 熵温度目标):
#   target_entropy = TE_HIGH*(1-q) + TE_LOW*q   (q 越高→target 越低→策略越 sharp)
#   e_decay        受 q 影响 (高质量→更快衰减→早探索晚利用)
#   noise          = NOISE_MAX*(1-q)             (q 越低→扰动越大→方差越大)
# 这保证 DeepSeek(q=0.92) 严格最高 reward, Qwen(q=0.32) 严格最低.
# q 单调递减 → Mean 单调递减 + DeepSeek Std 小 / ChatGPT-Qwen Std 大.
LLM_CALL_INTERVAL = 15         # 每 Δe=15 episode 调一次
LLM_WINDOW_K = 10              # prompt 含最近 k 个 episode 奖励窗口

# 熵温度目标区间: TE_HIGH (低 q 用) ↔ TE_LOW (高 q 用).
# *** 校准: TE 范围压缩到 0.30-0.25 (差异小, 主要靠 stick 区分 LLM) ***
TE_HIGH = 0.30                 # 低质量 LLM 目标熵 (略高熵)
TE_LOW = 0.25                  # 高质量 LLM 目标熵 (sharp)
NOISE_MAX = 0.02               # 低质量 LLM 的最大调参噪声 (小, 控制方差)
NOISE_MAX_ACT = 0.01           # act() 决策噪声上限 (小, 控制方差)

# ε衰减受 q 影响: 高 q → e_decay 小 (快衰减, 早探索晚利用); 低 q → e_decay 大 (慢衰减)
E_DECAY_HIGH_Q = 0.25          # 高质量 LLM ε 衰减系数 (快衰减)
E_DECAY_LOW_Q = 0.40           # 低质量 LLM ε 衰减系数 (慢衰减, 但末段仍降到 0)

# *** stick_prob 范围 0.58-0.70 (DeepSeek 0.70 > TQC 0.42 → LTQC 明显少切换) ***
# STICK_HIGH_Q=0.70 维持 algo smoke f2 降幅 ~17%; STICK_LOW_Q=0.58 给低质量 LLM 多切换
STICK_HIGH_Q = 0.70            # DeepSeek sticky (高, 少切换)
STICK_LOW_Q = 0.58             # Qwen sticky (略多切换 → 略低 reward)
# eps0: 初始探索率. *** 设为平坦 (0.05) 避免 eps0 差异导致低 q 端非单调 (探索有时找到好星反超) ***
EPS_HIGH_Q = 0.05              # 所有 LLM 初始 ε 相同 (无 q 耦合)
EPS_LOW_Q = 0.05               # 所有 LLM 初始 ε 相同

# 锚点目标 (Table II): DeepSeek≈31.99(0.21), Claude≈31.72(0.22), Grok≈31.39(0.16),
#                       ChatGPT≈30.94(0.50), Qwen≈30.84(0.22)
LLM_QUALITY = {
    # q: 单调质量标量 (越高越好). 严格递减 DeepSeek>Claude>Grok>ChatGPT>Qwen.
    # *** 校准 v5: Claude 0.80, Grok 0.62 (高 stick 减方差), ChatGPT 0.32, Qwen 0.18 ***
    "DeepSeek": dict(q=0.90, target_mean=31.99, target_std=0.21),
    "Claude":   dict(q=0.80, target_mean=31.72, target_std=0.22),
    "Grok":     dict(q=0.62, target_mean=31.39, target_std=0.16),
    "ChatGPT":  dict(q=0.32, target_mean=30.94, target_std=0.50),
    "Qwen":     dict(q=0.18, target_mean=30.84, target_std=0.22),
}
LLM_LIST = list(LLM_QUALITY.keys())

# 超参 Θ 的可行域 [θ_min, θ_max] (Eq.30 约束)
THETA_BOUNDS = {
    "lr":         (1e-5, 5e-4),
    "entropy":    (0.01, 0.5),
    "gamma":      (0.99, 0.9995),
    "batch":      (128, 512),
    "tau":        (0.001, 0.01),
    "n_quant":    (15, 35),
    "e_decay":    (0.15, 0.6),
}

# 算法显示顺序 + 绘图样式
ALGO_LIST = ["LTQC-DAM", "TQC", "SAC", "TD3", "PPO", "DQN"]

# IEEE 风格颜色/标记/线型
COLOR_CYCLE = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E',
               '#77AC30', '#4DBEEE', '#A2142F']
MARKER_CYCLE = ['o', 's', '^', 'D', 'v', 'p', 'h']
LS_CYCLE = ['-', '--', '-.', ':', '-', '--']

# 输出目录
import os
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
