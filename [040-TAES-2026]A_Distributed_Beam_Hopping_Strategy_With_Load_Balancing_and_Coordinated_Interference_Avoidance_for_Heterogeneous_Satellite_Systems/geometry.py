# -*- coding: utf-8 -*-
"""
卫星-小区几何与信道模型 (论文 Eq.1-2)

简化假设 (ASSUMPTION, 论文未要求真实轨道力学):
- LEO 卫星均匀分布在覆盖区域上空对应星下点。
- 小区在覆盖区域内均匀网格分布。
- 小区 j 可见卫星集 CS_j = {i : 仰角 >= 阈值}。
- 信道增益 |C|^2 (Eq.2) = 发射天线增益(偏轴角) + 接收天线增益 + 自由空间路径损耗。

几何校准 (使 Table III Distance 趋势"先升后降"):
- 小区密集布放在子区域 (相邻间距 ~16-30km, 触发 Eq.8 干扰)。
- LEO 卫星在小区区域**内**准均匀分布 (蓝噪声), 使 Voronoi 单元近似等大。
- 卫星数 N 增加 -> 单元变小 -> 大热点簇被切到多颗卫星 -> 缓解 (Distance L_df 回落)。
"""
import numpy as np
import config as C


def _latlon_to_ecef(lat_deg, lon_deg, alt_km):
    """经纬高 -> ECEF (km)."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    r = C.EARTH_RADIUS_KM + alt_km
    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)
    return np.array([x, y, z])


def _elevation(sat_ecef, cell_ecef):
    """由卫星 ECEF 和小区 ECEF 计算小区处仰角(度) 与斜距(km)。"""
    R = np.linalg.norm(cell_ecef)
    r = np.linalg.norm(sat_ecef)
    diff = sat_ecef - cell_ecef
    d = np.linalg.norm(diff)
    cos_theta = np.dot(cell_ecef, sat_ecef) / (R * r)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    sin_el = (r * cos_theta - R) / d
    sin_el = np.clip(sin_el, -1.0, 1.0)
    el = np.degrees(np.arcsin(sin_el))
    return el, d


def _blue_noise(n, span_lon, span_lat, min_sep, rng):
    """在 [-span, span] x [-span, span] 内生成 n 个准均匀点 (拒绝采样)。"""
    pts = []
    attempts = 0
    max_attempts = 20000
    while len(pts) < n and attempts < max_attempts:
        lo = rng.uniform(-span_lon, span_lon)
        la = rng.uniform(-span_lat, span_lat)
        ok = all(((lo - p[0]) ** 2 + (la - p[1]) ** 2) ** 0.5 > min_sep for p in pts)
        if ok:
            pts.append((lo, la))
        attempts += 1
    # 若拒绝采样不足, 直接随机补足
    while len(pts) < n:
        pts.append((rng.uniform(-span_lon, span_lon),
                    rng.uniform(-span_lat, span_lat)))
    return pts


class Geometry:
    """预计算所有卫星-小区几何关系与基础信道增益。"""

    def __init__(self, n_leo, seed=0):
        self.n_leo = n_leo
        self.rng = np.random.default_rng(seed)

        # --- 小区: 在覆盖区域内密集网格分布 ---
        # 400 小区 = 20x20; 子区域 1.5°x1.0°, 经向间距 ~0.079° (~7km),
        # 纬向间距 ~0.053° (~6km)。相邻/近邻小区距离 <= d=30km, GA 干扰规避有意义。
        side = int(np.sqrt(C.N_CELLS))
        assert side * side == C.N_CELLS
        clon = (C.LON_MIN + C.LON_MAX) / 2
        clat = (C.LAT_MIN + C.LAT_MAX) / 2
        cell_lon_half = 0.75   # 经向半宽 (~1.5°)
        cell_lat_half = 0.5    # 纬向半宽 (~1.0°)
        lons = np.linspace(clon - cell_lon_half, clon + cell_lon_half, side)
        lats = np.linspace(clat - cell_lat_half, clat + cell_lat_half, side)
        cell_lon, cell_lat = np.meshgrid(lons, lats)
        self.cell_lon = cell_lon.ravel()  # (N_c,)
        self.cell_lat = cell_lat.ravel()
        # 小区在地面的 ECEF
        self.cell_ecef = np.array([
            _latlon_to_ecef(la, lo, 0.0) for la, lo in zip(self.cell_lat, self.cell_lon)
        ])  # (N_c, 3)

        # 小区区域的边界 (用于布星)
        cell_lon_min, cell_lon_max = self.cell_lon.min(), self.cell_lon.max()
        cell_lat_min, cell_lat_max = self.cell_lat.min(), self.cell_lat.max()
        cell_clon = 0.5 * (cell_lon_min + cell_lon_max)
        cell_clat = 0.5 * (cell_lat_min + cell_lat_max)
        cell_half_lon = 0.5 * (cell_lon_max - cell_lon_min)
        cell_half_lat = 0.5 * (cell_lat_max - cell_lat_min)

        # --- LEO 卫星: 在小区区域内准均匀分布 ---
        # 使每颗卫星负责一片近似等大的地理子区域, 距离选星 Voronoi 单元均衡。
        # 用"向日葵" (Fibonacci spiral) 布点: 对任意 N 都产生近似等大 Voronoi 单元。
        # 加小抖动 (seed 相关), 使不同 run 略有变化, 多次平均后 Distance 曲线平滑。
        rng2 = np.random.default_rng(seed + 999)
        sx = cell_half_lon * 0.92
        sy = cell_half_lat * 0.92
        golden = np.pi * (3.0 - np.sqrt(5.0))
        pts = []
        for k in range(n_leo):
            # r 从中心向外, 但留中心一点偏移避免 k=0 退化
            r = np.sqrt((k + 0.5) / n_leo)
            theta = k * golden
            lo = r * np.cos(theta) * sx
            la = r * np.sin(theta) * sy
            # 小抖动 (用 seed, 使不同 run 略有变化用于平均)
            jit_lo = rng2.uniform(-sx * 0.05, sx * 0.05)
            jit_la = rng2.uniform(-sy * 0.05, sy * 0.05)
            pts.append((lo + jit_lo, la + jit_la))
        self.sat_lon = np.array([cell_clon + p[0] for p in pts[:n_leo]])
        self.sat_lat = np.array([cell_clat + p[1] for p in pts[:n_leo]])
        self.sat_ecef = np.array([
            _latlon_to_ecef(la, lo, C.LEO_ALTITUDE_KM) for la, lo in zip(self.sat_lat, self.sat_lon)
        ])  # (n_leo, 3)

        # --- 预计算可见性与几何量 ---
        self.visibility = np.zeros((n_leo, C.N_CELLS), dtype=bool)  # CS_j 的转置
        self.slant = np.zeros((n_leo, C.N_CELLS))     # 斜距 (km)
        self.off_axis = np.zeros((n_leo, C.N_CELLS))  # 偏轴角 (deg)
        for i in range(n_leo):
            for j in range(C.N_CELLS):
                el, d = _elevation(self.sat_ecef[i], self.cell_ecef[j])
                self.slant[i, j] = d
                # 偏轴角: 卫星指向星下点 vs 指向小区 的夹角, 近似 = 90 - el
                self.off_axis[i, j] = max(0.0, 90.0 - el)
                self.visibility[i, j] = (el >= C.MIN_ELEVATION_DEG)

        # --- 小区级信道增益 (线性) |C|^2 (Eq.2) ---
        freq_hz = 12e9   # 假设 Ku 波段 ~12 GHz (ASSUMPTION)
        lam = C.C_LIGHT / freq_hz
        pl_db = 20.0 * np.log10(4.0 * np.pi * self.slant * 1e3 / lam)

        # 偏轴方向图衰减 (dB, 相对峰值): 越偏轴损耗越大
        pattern_loss_db = 12.0 * (self.off_axis / 5.0) ** 2
        pattern_loss_db = np.clip(pattern_loss_db, 0.0, 30.0)   # 最多衰减 30 dB

        # 等效天线增益 (dBi) = 峰值 - 偏轴损耗
        gt_db = C.SAT_ANT_GAIN_DBI - pattern_loss_db
        gr_db = np.full_like(gt_db, C.TERM_ANT_GAIN_DBI)

        # channel_gain_linear 已包含 G_T(含峰值) 与 G_R 与路径损耗, 信号 = P * gain
        gain_db = gt_db + gr_db - pl_db
        self.channel_gain_linear = 10.0 ** (gain_db / 10.0)   # (n_leo, N_c)

        # 每小区可见卫星集 CS_j: 限制为斜距最近的 K_LEO_VISIBLE 颗
        # (使每小区只被局部少数 LEO 服务, RW-LB 在局部均衡 -> 信号强、干扰小;
        #  同时限制星间干扰只来自局部邻域)
        K_VIS = max(3, min(n_leo, C.K_LEO_VISIBLE))
        self.CS = []
        for j in range(C.N_CELLS):
            order = np.argsort(self.slant[:, j])
            cs = order[:K_VIS]
            self.CS.append(cs)
            # 更新 visibility
            self.visibility[:, j] = False
            self.visibility[cs, j] = True
        # 处理孤立小区 (无可见卫星): 退化为最近卫星
        for j in range(C.N_CELLS):
            if len(self.CS[j]) == 0:
                nearest = int(np.argmin(self.slant[:, j]))
                self.CS[j] = np.array([nearest])
                self.visibility[nearest, j] = True

        # 小区间距离矩阵 (用于干扰判断 Eq.8)
        self._cell_pair_dist = self._compute_cell_distances()

    def _compute_cell_distances(self):
        """小区两两地面距离 (km), 用经纬度近似球面距离。"""
        lat1 = np.radians(self.cell_lat)
        lon1 = np.radians(self.cell_lon)
        lat1 = lat1[:, None]; lon1 = lon1[:, None]
        lat2 = lat1.T; lon2 = lon1.T
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        d = 2 * C.EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))
        return d   # (N_c, N_c)

    def cell_pair_interfere(self, j1, j2):
        """Eq.8: 两小区是否产生波束间干扰 (距离 <= d)。"""
        return self._cell_pair_dist[j1, j2] <= C.CELL_DIAMETER_KM

    def interference_mask(self, cells):
        """给定小区索引列表, 返回 (m,m) 干扰指示矩阵 I_{i,j} (Eq.8)。"""
        cells = np.asarray(cells)
        sub = self._cell_pair_dist[np.ix_(cells, cells)]
        mask = (sub <= C.CELL_DIAMETER_KM).astype(float)
        np.fill_diagonal(mask, 0.0)
        return mask
