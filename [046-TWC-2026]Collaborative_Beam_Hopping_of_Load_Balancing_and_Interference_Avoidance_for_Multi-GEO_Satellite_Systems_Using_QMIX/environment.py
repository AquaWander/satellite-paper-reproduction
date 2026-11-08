"""
environment.py — 多GEO卫星跳波束仿真环境
实现: 小区布局, 流量生成, 波束分配, 天线增益, 干扰模型, SINR, 容量, 吞吐量, 延迟, 负载
"""
import numpy as np
import config as cfg


def itu_antenna_gain(phi_deg, g_max_dbi, phi3_deg, ls_db):
    """
    ITU-R S.672-4 天线增益模式 (dB)
    :param phi_deg: 偏离主瓣角度 (度)
    :param g_max_dbi: 最大增益 (dBi)
    :param phi3_deg: 半功率波束宽度 (度)
    :param ls_db: 旁瓣限制 (dB)
    :return: 增益 (dB)
    """
    phi = max(phi_deg, 1e-6)
    a = cfg.ANTENNA_A  # 3.16
    b = cfg.ANTENNA_B  # 6.32
    if phi <= phi3_deg:
        g = g_max_dbi - 12.0 * (phi / phi3_deg) ** 2
    elif phi <= a * phi3_deg:
        g = g_max_dbi - 12.0 - 12.0 * np.log10(phi / phi3_deg)
    elif phi <= b * phi3_deg:
        g = g_max_dbi - 12.0 - 12.0 * np.log10(a) - 10.0 * np.log10(phi / (a * phi3_deg))
    else:
        g = g_max_dbi - 12.0 - 12.0 * np.log10(a) - 10.0 * np.log10(b / a) \
            - 20.0 * np.log10(phi / (b * phi3_deg))
    # 限制旁瓣
    g = max(g, ls_db)
    return g


class GEOBeamHoppingEnv:
    """多GEO卫星跳波束仿真环境"""

    def __init__(self, n_cells=cfg.N_CELLS, n_beams=cfg.N_BEAMS,
                 n_sat=cfg.N_SATELLITES, seed=0):
        self.n_cells = n_cells
        self.n_beams = n_beams
        self.n_sat = n_sat
        self.rng = np.random.RandomState(seed)

        # 小区网格布局 (n_cells = rows*cols)
        self.n_rows = int(np.sqrt(n_cells))
        self.n_cols = int(np.ceil(n_cells / self.n_rows))
        # 小区中心坐标 (归一化到单位正方形, 中心在网格上)
        self.cell_pos = np.zeros((n_cells, 2))
        idx = 0
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if idx < n_cells:
                    self.cell_pos[idx] = [c + 0.5, r + 0.5]
                    idx += 1
        # 小区间角度分离 (度) - 由位置差估计
        # 卫星到地面距离产生的角分离
        self._compute_angles()

        # 卫星位置 (角度, 度) - 3颗GEO卫星
        self.sat_angles = np.array([-3.5, 0.0, 3.5])  # 相对角位置 (度)

        # 状态
        self.traffic = np.zeros((self.n_sat, self.n_cells))  # 流量需求 (Mbps)
        self.alloc = np.zeros((self.n_sat, self.n_cells), dtype=int)  # 小区是否被服务
        self.subband = np.zeros((self.n_sat, self.n_cells), dtype=int)  # 子频带分配
        self.served_data = np.zeros((self.n_sat, self.n_cells))  # 已服务数据量
        self.delay = np.zeros((self.n_sat, self.n_cells))  # 各小区排队延迟
        self.t = 0

    def _compute_angles(self):
        """计算小区间的角分离 (度)"""
        # 简化: 相邻小区角分离约 1度 (由波束宽度和覆盖范围决定)
        # 这里用小区网格间距近似
        self.angle_sep = np.zeros((self.n_cells, self.n_cells))
        for i in range(self.n_cells):
            for j in range(self.n_cells):
                d = np.linalg.norm(self.cell_pos[i] - self.cell_pos[j])
                self.angle_sep[i, j] = d * 1.0  # 每格约1度

    def generate_traffic(self, mean_base=250.0):
        """生成泊松流量需求
        :param mean_base: 基准流量均值 (Mbps)
        """
        # 每个小区流量: Poisson, λ 在 [50,500] 均匀随机
        for s in range(self.n_sat):
            lam = self.rng.uniform(cfg.TRAFFIC_LAMBDA_MIN,
                                   cfg.TRAFFIC_LAMBDA_MAX, self.n_cells)
            self.traffic[s] = self.rng.poisson(lam)
        return self.traffic

    def scale_traffic(self, scale):
        """按比例缩放流量"""
        self.traffic = self.traffic * scale

    def reset(self, mean_base=250.0):
        """重置环境"""
        self.generate_traffic(mean_base)
        self.alloc = np.zeros((self.n_sat, self.n_cells), dtype=int)
        self.subband = np.zeros((self.n_sat, self.n_cells), dtype=int)
        self.served_data = np.zeros((self.n_sat, self.n_cells))
        self.delay = np.zeros((self.n_sat, self.n_cells))
        self.t = 0
        return self.get_state()

    def get_state(self):
        """返回全局状态: 所有小区的流量需求"""
        return self.traffic

    def get_obs(self, s):
        """返回卫星 s 的局部观测"""
        return self.traffic[s]

    def set_action(self, action_sat, s):
        """设置卫星 s 的波束分配动作
        :param action_sat: 长度为 n_cells 的0/1向量, 指出哪些小区被服务
        """
        self.alloc[s] = action_sat

    def beam_gain_sat_to_cell(self, s, k_cell, v_cell, tx=True):
        """计算卫星 s 在小区 k (发射) 对小区 v (接收) 的增益
        :return: 增益 (dB)
        """
        # 小区 k 是卫星波束指向的小区, 小区 v 是观测点
        phi = self.angle_sep[k_cell, v_cell]
        if tx:
            return itu_antenna_gain(phi, cfg.G_TRAN_MAX_DBI, cfg.HPBW_DEG,
                                    cfg.ANTENNA_LS_DB)
        else:
            return itu_antenna_gain(phi, cfg.G_REC_MAX_DBI, cfg.HPBW_DEG,
                                    cfg.ANTENNA_LS_DB)

    def compute_interference(self, s, k, v):
        """
        计算小区 v 接收来自卫星 s 波束 k 的信号时受到的干扰
        干扰 = 星内同小区重复 + 星间同小区 + 邻频同频 (简化自论文 Eq.29-32)
        :return: 干扰功率 (W)
        """
        # 有用信号功率 (W)
        p_sig = cfg.P_BEAM_W * 10**(self.beam_gain_sat_to_cell(s, k, v)/10) \
            * 10**(-cfg.PATH_LOSS_DB/10) * 10**(cfg.G_REC_MAX_DBI/10)

        # 干扰功率: 其他有源波束对同一卫星其他小区, 或其他卫星的干扰
        p_int = 0.0
        # 星内干扰: 同卫星其他波束指向其他小区, 对小区 v 的泄漏
        for k2 in range(self.n_cells):
            if self.alloc[s, k2] == 1 and k2 != k:
                g = 10**(self.beam_gain_sat_to_cell(s, k2, v)/10)
                p_int += cfg.P_BEAM_W * g * 10**(-cfg.PATH_LOSS_DB/10) \
                    * 10**(cfg.G_REC_MAX_DBI/10)
        # 星间干扰: 其他卫星的波束指向小区 v 或邻近
        for s2 in range(self.n_sat):
            if s2 == s:
                continue
            for k2 in range(self.n_cells):
                if self.alloc[s2, k2] == 1:
                    # 卫星间角分离
                    ang = abs(self.sat_angles[s] - self.sat_angles[s2])
                    g_tx = itu_antenna_gain(ang, cfg.G_TRAN_MAX_DBI, cfg.HPBW_DEG,
                                            cfg.ANTENNA_LS_DB)
                    g_rx = 10**(self.beam_gain_sat_to_cell(s2, k2, v)/10)
                    p_int += cfg.P_BEAM_W * 10**(g_tx/10) * 10**(-cfg.PATH_LOSS_DB/10) * g_rx
        return p_sig, p_int

    def compute_sinr(self, s, k, v):
        """计算 SINR (Eq.6)"""
        p_sig, p_int = self.compute_interference(s, k, v)
        # 噪声功率
        k_boltz = 1.38e-23
        t_noise = 290.0
        p_noise = k_boltz * t_noise * cfg.SUB_BW_HZ
        sinr = p_sig / (p_int + p_noise)
        return sinr

    def compute_capacity(self, s, k):
        """计算容量 (Eq.4): C = δ · B · log2(1 + γ)"""
        # 简化: 只考虑被服务的小区
        c = np.zeros(self.n_cells)
        for v in range(self.n_cells):
            if self.alloc[s, v] == 1:
                sinr = self.compute_sinr(s, v, v)
                c[v] = cfg.SUB_BW_HZ * np.log2(1 + sinr)
        return c  # bits/s

    def step(self):
        """推进一个时隙, 更新服务数据、延迟"""
        # 容量计算
        for s in range(self.n_sat):
            cap = self.compute_capacity(s, s)  # 每小区容量 (bits/s)
            cap_mbps = cap / 1e6  # Mbps
            # 吞吐量 = min(容量, 剩余流量需求) (Eq.15)
            throughput = np.minimum(cap_mbps, self.traffic[s])
            self.served_data[s] += throughput
            # 更新剩余流量和延迟
            remaining = np.maximum(0, self.traffic[s] - throughput)
            self.traffic[s] = remaining
            # 延迟: 未服务或服务不足的小区延迟增加
            self.delay[s] += (remaining > 0).astype(float)
        self.t += 1
        return self.t

    def get_metrics(self):
        """返回各指标"""
        throughput = self.served_data.copy()  # Mbps
        delay = self.delay.copy()
        load = self.served_data.copy()  # 负载 = 已服务数据量
        return throughput, delay, load

    # ========== 论文公式指标 ==========

    def compute_objective(self):
        """Eq.28: F_t = ω_T·T - ω_D·D - ω_L·L"""
        # 平均吞吐量 (T)
        T = np.mean(self.served_data)
        # 延迟偏移 (D) - Eq.19-20
        D = self._delay_offset()
        # 负载偏移 (L) - Eq.24-26
        L = self._load_offset()
        return cfg.W_T * T - cfg.W_D * D - cfg.W_L * L

    def _delay_offset(self):
        """Eq.19-20: 卫星内最大最小延迟差"""
        offsets = []
        for s in range(self.n_sat):
            d = self.delay[s]
            offsets.append(np.max(d) - np.min(d))
        return np.mean(offsets)

    def _load_offset(self):
        """Eq.24-26: 卫星间负载偏移"""
        loads = np.zeros(self.n_sat)
        for s in range(self.n_sat):
            loads[s] = np.sum(self.served_data[s])
        return np.max(loads) - np.min(loads)

    def satisfaction_rate(self):
        """Eq.49: 平均流量满意度率 r_v = d_v / T_v"""
        rates = []
        for s in range(self.n_sat):
            for v in range(self.n_cells):
                total = self.served_data[s, v]
                demand = self.served_data[s, v]  # 已服务即满足
                # 满意度 = min(1, 服务/需求) 这里用服务率
                if total > 0:
                    rates.append(min(1.0, total / (total + self.traffic[s, v])))
                else:
                    rates.append(0.0)
        return np.mean(rates) if rates else 0.0

    def normalized_load_offset(self):
        """归一化负载偏移"""
        loads = np.zeros(self.n_sat)
        for s in range(self.n_sat):
            loads[s] = np.sum(self.served_data[s])
        span = np.max(loads) - np.min(loads)
        total = np.sum(loads)
        return span / total if total > 0 else 0.0