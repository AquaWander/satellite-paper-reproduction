# -*- coding: utf-8 -*-
"""
业务模型 — RT/NRT 需求生成。

- 总需求 D_total (Mbps) 按以下方式分摊到 400 小区:
  * 创建 N_HOTSPOT_REGIONS 个强聚集的热点区域, 每个区域含若干相邻小区
    (共同承载很高的业务量, 约为均值 * HOTSPOT_GAIN 倍)。
  * 强聚集热点使 Distance 选星在 10-15 LEO 出现峰值 (热点集中到单星):
    LEO 少  -> 单星覆盖宽, 热点被分散 (差适中)
    LEO 增  -> 多个相邻热点小区选同一颗最近卫星 -> 单星严重过载 (差增大)
    LEO 多  -> 热点被多星分担 -> 缓解 (差回落)
- 每小区业务再按 RT_RATIO 拆分为 RT / NRT。
- 跨 64 时隙均匀分布 (即每时隙需求 = 单帧需求 / N_SLOTS)。

返回: D_rt (N_c, N_t Mbps), D_nrt (N_c, N_t Mbps), hotspot_idx
"""
import numpy as np
import config as C


def _hotspot_clusters(side, rng):
    """生成 N_HOTSPOT_REGIONS 个聚集热点簇。

    簇位置: 在网格 4 个象限 (准确定位), 加小随机扰动, 使:
    - N=5:  4 簇各落到不同卫星 (覆盖宽) -> 分担 (Distance 适中 0.4)
    - N=10: 卫星密集到每簇上方 -> 单星过载 (Distance 峰值 0.7)
    - N=25: 单元更小, 簇被切到多星 -> 缓解 (Distance 回落)
    每簇为 3x3 (9 小区), 跨度 ~0.3°。
    返回 list of cell index。
    """
    n_regions = C.N_HOTSPOT_REGIONS
    # 4 个象限的中心 (网格坐标), 加小扰动
    quadrants = [
        (side // 4, side // 4),
        (side // 4, 3 * side // 4),
        (3 * side // 4, side // 4),
        (3 * side // 4, 3 * side // 4),
    ]
    seeds = []
    for i in range(n_regions):
        base = quadrants[i % 4]
        rr = int(base[0] + rng.integers(-1, 2))
        cc = int(base[1] + rng.integers(-1, 2))
        rr = max(1, min(side - 2, rr))
        cc = max(1, min(side - 2, cc))
        seeds.append((rr, cc))

    hotspot_idx = []
    for (r0, c0) in seeds:
        # 3x3 簇 (跨度 ~0.3°): N=10 单星承载峰值, N=25 切分缓解
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                rr, cc = r0 + dr, c0 + dc
                if 0 <= rr < side and 0 <= cc < side:
                    hotspot_idx.append(rr * side + cc)
    # 去重
    hotspot_idx = sorted(set(int(x) for x in hotspot_idx))
    return hotspot_idx


def generate_traffic(geo, total_demand_mbps, seed=0):
    """
    生成一帧的业务需求矩阵。

    Returns
    -------
    D_rt  : (N_c, N_t) Mbps, 每小区每时隙 RT 需求
    D_nrt : (N_c, N_t) Mbps, 每小区每时隙 NRT 需求
    hotspot_idx : list, 热点小区索引
    """
    rng = np.random.default_rng(seed)
    n = C.N_CELLS

    if total_demand_mbps <= 0:
        z = np.zeros((n, C.N_SLOTS_PER_FRAME))
        return z, z, []

    side = int(np.sqrt(n))
    hotspot_idx = _hotspot_clusters(side, rng)

    # 非热点均值, 使总均值 = total_demand
    # total = n_hot*HOTSPOT_GAIN*mu + (n-n_hot)*mu = mu*(n_hot*(G-1)+n)
    n_hot = len(hotspot_idx)
    mu = total_demand_mbps / (n_hot * (C.HOTSPOT_GAIN - 1.0) + n)

    # 每小区期望业务 (Mbps, 每帧总和), 加泊松扰动
    cell_mean = np.full(n, mu)
    cell_mean[hotspot_idx] *= C.HOTSPOT_GAIN
    cell_demand = rng.normal(cell_mean, 0.30 * cell_mean)
    cell_demand = np.clip(cell_demand, 0.0, None)
    # 重新归一使总和精确等于 total_demand
    cell_demand *= total_demand_mbps / cell_demand.sum()

    # 拆分 RT / NRT (每帧, Mbps)
    rt_demand = cell_demand * C.RT_RATIO
    nrt_demand = cell_demand * (1.0 - C.RT_RATIO)

    # 跨时隙均匀 -> 每时隙 Mbps
    D_rt = np.tile((rt_demand / C.N_SLOTS_PER_FRAME)[:, None], (1, C.N_SLOTS_PER_FRAME))
    D_nrt = np.tile((nrt_demand / C.N_SLOTS_PER_FRAME)[:, None], (1, C.N_SLOTS_PER_FRAME))

    return D_rt, D_nrt, list(hotspot_idx)
