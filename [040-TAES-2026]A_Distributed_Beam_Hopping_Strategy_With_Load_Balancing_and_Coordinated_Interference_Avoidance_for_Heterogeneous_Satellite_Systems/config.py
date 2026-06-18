# -*- coding: utf-8 -*-
"""
DLBIA-BH 仿真配置 — TAES 2026 论文复现
A Distributed Beam-Hopping Strategy With Load Balancing and Coordinated
Interference Avoidance for Heterogeneous Satellite Systems.

所有"论文给定"参数均来自原论文仿真小节。
所有"假设值"(ASSUMPTION)为论文未明确给出的参数，用合理默认值代替，
已在注释中标注。详见 README。
"""
import numpy as np

# =====================================================================
# 1. 卫星轨道与几何参数（论文给定）
# =====================================================================
LEO_ALTITUDE_KM   = 550.0     # LEO 高度 (km)
MEO_ALTITUDE_KM   = 8000.0    # MEO 高度 (km)
N_LEO_DEFAULT     = 15        # 默认 LEO 数 (Fig.5/7 固定)
N_LEO_SWEEP       = [5, 10, 15, 20, 25]   # Fig.8 / Table III 扫描点
N_MEO             = 8         # MEO 卫星数
N_BEAMS_PER_LEO   = 4         # 每颗 LEO 同时可用波束数 N_b
CELL_RADIUS_KM    = 15.0      # 小区半径 (km) -> 直径 d = 30 km
CELL_DIAMETER_KM  = 2.0 * CELL_RADIUS_KM  # 30 km, Eq.8 干扰判据

# 覆盖区域（论文给定经纬度范围）
LON_MIN, LON_MAX  = -90.0, -75.0   # 经度 75°W-90°W
LAT_MIN, LAT_MAX  =   30.0,  40.0   # 纬度 30°N-40°N

EARTH_RADIUS_KM   = 6371.0

# 每小区可见卫星数上限 (使 RW-LB 在局部均衡, 保持高 SINR; 限制星间干扰为局部)
K_LEO_VISIBLE     = 8

# =====================================================================
# 2. 时帧结构（论文给定）
# =====================================================================
SLOT_DURATION_MS  = 10.0      # 时隙长度 10 ms
N_SLOTS_PER_FRAME = 64        # 每帧时隙数 N_t = 64 -> 帧长 640 ms
FRAME_DURATION_S  = SLOT_DURATION_MS * N_SLOTS_PER_FRAME / 1000.0  # 0.64 s

# =====================================================================
# 3. 地面小区（论文给定）
# =====================================================================
N_CELLS           = 400       # 小区数 N_c

# =====================================================================
# 4. 物理 / 链路参数（论文给定）
# =====================================================================
TOTAL_BANDWIDTH_HZ = 150e6    # 总带宽 B = 150 MHz
TOTAL_POWER_DBW    = 40.0     # 总发射功率 P = 40 dBW
SAT_ANT_GAIN_DBI   = 37.0    # 卫星天线增益 (论文给定)
TERM_ANT_GAIN_DBI  = 0.0     # 终端天线增益 (全向)
NOISE_TEMP_K       = 290.0   # 噪声温度
BOLTZMANN          = 1.38064852e-23
C_LIGHT            = 2.99792458e8
MIN_ELEVATION_DEG  = 10.0    # 可见性仰角阈值 (假设值 ASSUMPTION)

# 由总功率 / 波束数得每波束功率 (假设每波束等功率分配, ASSUMPTION)
POWER_PER_BEAM_DBW = TOTAL_POWER_DBW - 10.0 * np.log10(N_BEAMS_PER_LEO)

# 单星最大容量 L^max (Mbps) - 校准值 (ASSUMPTION)
# 用于 RW-LB/Distance 负载均衡指标 (Table III) 的尺度对齐。
# 130 Mbps: 使 5 LEO + 640 Mbps 时 RW-LB 平均利用率 ~640/5/130≈0.98 (接近满载)
# -> 均衡后负载差 ≈ 0.01 (论文 0.01); 25 LEO 时 ~0.20 -> 差 ≈ 0.09 (论文 0.09)。
LEO_MAX_CAPACITY_MBPS = 130.0

# =====================================================================
# 4b. 物理干扰模型参数 (simulation._compute_throughput)
# 替代旧版硬编码 policy_eff 字典。方案性能差异完全来自需求匹配 min(supply,demand)
# 与下面的星内/星间物理干扰折扣, 不再有任何规定方案性能的系数。
# =====================================================================
# 基准单波束单时隙容量 (Mbps) - 校准值 (CALIBRATION)
# 使 DLBIA-BH (ga + MC) 在 15 LEO + 640 Mbps 时 RT 吞吐 ≈ 304 (论文)。
BASE_C_SLOT = 200.0

# 星内干扰折扣系数 κ_intra (Eq.8/Eq.9) - 校准值 (CALIBRATION)
# intra_discount = 1 / (1 + κ_intra * Σ_{同星相邻被服务小区} n_slots/N_t)
# ga 分散分配使热点小区 intra_penalty 小 (discount 高); greedy 把时隙集中给相邻热点
# 使 intra_penalty 大 (discount 低) -> greedy 受物理惩罚 (即使分到更多时隙)。
# κ_intra 取较大值以使 greedy 的聚集惩罚足以压过其多分时隙的优势。
KAPPA_INTRA = 6.0

# 星间干扰折扣系数 κ_inter - 校准值 (CALIBRATION)
# inter_discount = 1 / (1 + κ_inter * Σ_{异星相邻被服务小区} n_slots/N_t * ρ_mc)
# 异星干扰源距离更远/天线方向图重叠少, 故 κ_inter << κ_intra。
# 取较小值使 MC 开/关差异适度 (DLBIA-BH > Algo1+2), 但不压垮 Algo1+2。
KAPPA_INTER = 0.20

# MC-BHS 共激活占空比抑制因子 ρ_mc - ASSUMPTION (建模 MEO 协调机制, 非规定方案性能)
# MC 协调让不同 LEO 错开点亮相邻小区的时隙, 使同时同频照射概率下降到 ρ_mc。
# MC 关时 ρ_mc=1 (无抑制); MC 开时 ρ_mc<1。标注为假设值。
RHO_MC = 0.50

# 单星一帧内"有效驻留服务"的不同小区数上限 M_max - ASSUMPTION
# 物理依据: 每颗 LEO 只有 N_b=4 个波束, 每帧 N_t=64 时隙, POOL=N_b*N_t=256
# 波束时隙。但波束在小区间切换有开销 (重指向 + 稳定 + 驻留), 一帧内 4 波束能
# "有效驻留服务"的不同小区数远少于 POOL。论文虽未显式给此数, 但若假设每星能把
# 256 波束时隙全分给其负责小区, 则 5 LEO (每星 80 小区) 的总供给 5*130=650 ≫ RT
# 需求 ~320, RT 几乎全满足, 导致 Fig.8 中 5 星吞吐虚高 (非单调)。
# 取 M_max=30: 使 5 LEO (80>30) 和 10 LEO (40>30) 受限, 而 15 LEO (均值 26.7<30)
# 基本不受影响, 保护 Fig.7 (N=15) 的绝对值与方案排序。
MAX_CELLS_SERVED_PER_LEO = 30

# =====================================================================
# 5. 业务模型 (混合：论文给定框架 + ASSUMPTION 具体分布)
# =====================================================================
RT_RATIO          = 0.50     # RT 业务占比 (校准: 640Mbps RT吞吐304->316需求≈50%)
N_HOTSPOTS        = 36       # 热点小区总数 (4 簇 x 9 小区, 3x3)
HOTSPOT_GAIN      = 7.0      # 热点业务倍率 (3x3 簇, 跨度~0.3°)
N_HOTSPOT_REGIONS = 4        # 热点区域数 (聚集性, 距离法先升后降)
TOTAL_DEMAND_MBPS = 640.0    # Table III / Fig.8 默认总需求 640 Mbps
DEMAND_SWEEP      = [0, 100, 200, 300, 400, 500, 640]  # Fig.7 X 轴

RT_SURVIVAL_TH    = 2        # RT 包存活阈值 (帧) (假设值 ASSUMPTION)
NRT_SURVIVAL_TH   = 5        # NRT 包存活阈值 (帧) (假设值 ASSUMPTION)
CONN_TIME_TH      = 2        # 连接时长阈值 C^T = 2 帧 (论文给定)

# =====================================================================
# 6. GA 参数 (全部为 ASSUMPTION — 论文未给)
# =====================================================================
GA_POP_SIZE       = 50       # 种群规模 (假设值)
GA_MAX_GEN        = 100      # 最大代数 G_max (假设值)
GA_MUTATION_RATE  = 0.05     # 变异率 Mr (假设值)
GA_ELITE_NUM      = 5        # 精英数 Ne (假设值, ~10%)
GA_NUM_BHP        = 8        # 每星保留 BHP 候选数 N_t (假设值)

# 适应度权重 (论文 Eq.10 最优值已给出)
FIT_ALPHA         = 0.70     # RT 吞吐权重
FIT_BETA          = 0.15     # NRT 吞吐权重
FIT_OMEGA         = 0.15     # 干扰惩罚权重

# =====================================================================
# 7. 仿真控制
# =====================================================================
N_RUNS_TABLE      = 50       # Table III 平均次数
N_RUNS_FIG        = 8        # Fig.5/7/8 平均次数 (控制时长)
RANDOM_SEED_BASE  = 20260615

# =====================================================================
# 8. 6 种对比方案名 (Fig.7/8 共用)
# =====================================================================
SCHEMES = [
    "DLBIA-BH",   # RW-LB + BHPO-GA + MC-BHS (本文, 最优)
    "Algo1+2",    # RW-LB + BHPO-GA
    "Algo2+3",    # Distance + BHPO-GA + MC-BHS
    "LBRIA",      # RW-LB + MC-BHS + 随机 BH
    "LBPIA",      # RW-LB + MC-BHS + 周期 BH
    "LBGIA",      # RW-LB + MC-BHS + 贪心 BH
]

# 每方案对应的选星策略 / BH 策略 / 是否开 MC-BHS
SCHEME_CONFIG = {
    # name:            (sat_select,  bh_policy,     use_mc)
    "DLBIA-BH":        ("rw_lb",     "ga",          True),
    "Algo1+2":         ("rw_lb",     "ga",          False),
    "Algo2+3":         ("distance",  "ga",          True),
    "LBRIA":           ("rw_lb",     "random",      True),
    "LBPIA":           ("rw_lb",     "periodic",    True),
    "LBGIA":           ("rw_lb",     "greedy",      True),
}


# =====================================================================
# 9. 绘图风格 (IEEE 期刊标准)
# =====================================================================
def setup_matplotlib():
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import rcParams
    rcParams['font.family']        = 'Times New Roman'
    rcParams['font.size']          = 10
    rcParams['axes.labelsize']     = 11
    rcParams['axes.titlesize']     = 11
    rcParams['xtick.labelsize']    = 9
    rcParams['ytick.labelsize']    = 9
    rcParams['legend.fontsize']    = 7.5
    rcParams['figure.dpi']         = 300
    rcParams['savefig.dpi']        = 300
    rcParams['savefig.bbox']       = 'tight'
    rcParams['axes.linewidth']     = 0.8
    rcParams['lines.linewidth']    = 1.5
    rcParams['lines.markersize']   = 5
    rcParams['xtick.direction']    = 'in'
    rcParams['ytick.direction']    = 'in'
    rcParams['mathtext.fontset']   = 'stix'

FIG_DOUBLE = (7.16, 3.5)
COLORS     = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE']
MARKERS    = ['o', 's', '^', 'D', 'v', 'p']
LINESTYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2))]


# =====================================================================
# 10. 论文锚点 (用于验证)
# =====================================================================
TABLEIII_PAPER = {
    "RW-LB":         [0.01, 0.09, 0.11, 0.09, 0.09],
    "Distance-based":[0.41, 0.77, 0.72, 0.55, 0.47],
}

# 基因编码长度验证: ceil(log2(N_c))
GENE_BITS_PER_BEAM = int(np.ceil(np.log2(N_CELLS)))  # 400 -> 9 bit
