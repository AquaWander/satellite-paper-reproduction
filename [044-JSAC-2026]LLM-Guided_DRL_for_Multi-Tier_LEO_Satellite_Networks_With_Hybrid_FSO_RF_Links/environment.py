"""
environment.py
===========
多层 LEO-HAP-地面 用户簇 MDP 环境。

物理层:
- 轨道动力学 (Eq.1): ω_i(t) = ω_init_i + (t·ϖ_i mod τ_i)   取模
- 可见性掩码 (Eq.24): M_t[i]=1 若卫星 i 在时刻 t 对 HAP 可见(仰角≥mask)
- FSO 速率 (Eq.7): 选定服务卫星→HAP 的 FSO SNR→速率
- RF 速率 (Eq.10): HAP→各簇 OFDM 速率
- 流守恒 (Eq.11): Σ R_RF ≤ R_FSO
- 瓶颈比例分配 (Eq.14): R'_RF_i = R_RF_i/Σ_j R_RF_j · R_FSO

MDP (Eq.18-20):
- 状态 s[t] = {t, 当前服务卫星 idx, 累计切换数, 各簇累计数据, 各簇信道条件}
- 动作 a[t] = 选择下一时刻服务卫星 idx (离散动作空间, 维度=N_LEO)
              子载波/用户分配由环境按 Eq.14 比例公平规则确定性处理
- 奖励 r[t] = η·R_total_norm(t) − ζ·I(切换)   (R_total=min{R_FSO, ΣR_RF}, Eq.13)

实现要点: numpy 预计算整个 episode 的可见性 + 信道轨迹(每种子确定但不同)。
"""
from __future__ import annotations
import numpy as np

import config as C
import channels as ch


# ============================================================================
# 轨道动力学 (Eq.1)
# ============================================================================
class Constellation:
    """110 颗 LEO 卫星轨道; 预计算每个时间步的位置与对 HAP 可见性."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        # 高度数组: 前 80 @ 500km, 后 30 @ 1000km
        self.altitudes = np.concatenate([
            np.full(C.N_LEO_LOW, C.H_LOW),
            np.full(C.N_LEO_HIGH, C.H_HIGH),
        ])
        self.orbit_radius = self.altitudes + C.R_e           # H_i = h_i + R_e
        # 角速度 ϖ_i = sqrt(G·M_e / H_i^3)
        self.angular_vel = np.sqrt(C.G_M_e / self.orbit_radius ** 3)
        # 周期 τ_i = 2π/ϖ_i
        self.period = 2.0 * np.pi / self.angular_vel
        # 初始相位 ω_init_i (每个卫星随机 [0, 2π))
        self.omega_init = rng.uniform(0, 2 * np.pi, size=C.N_LEO)
        # 轨道倾角 (随机分布, 30°~60° 典型 LEO 极/斜轨)
        self.inclination = np.deg2rad(rng.uniform(30, 60, size=C.N_LEO))
        # 升交点赤经 (随机)
        self.raan = rng.uniform(0, 2 * np.pi, size=C.N_LEO)

        # HAP 位置 (固定, 赤道上空 + 中纬度, 这里取中纬度 30°N)
        self.hap_lat = np.deg2rad(30.0)
        self.hap_lon = 0.0

    def satellite_position(self, sat_idx, t):
        """
        简化轨道位置 (球面坐标): 计算 LLA.
        ω_i(t) = ω_init_i + (t·ϖ_i mod τ_i)   (Eq.1, 取模)
        """
        theta = self.omega_init[sat_idx] + \
                np.mod(t * self.angular_vel[sat_idx], self.period[sat_idx])
        inc = self.inclination[sat_idx]
        raan = self.raan[sat_idx]
        # 在轨道平面内的位置 (球面近似)
        lat = np.arcsin(np.sin(inc) * np.sin(theta))
        lon = raan + np.arctan2(np.cos(inc) * np.sin(theta), np.cos(theta))
        return lat, lon, self.orbit_radius[sat_idx]

    def elevation_to_hap(self, sat_idx, t):
        """卫星对 HAP 的仰角 (度). 用球面几何近似."""
        lat_s, lon_s, r_s = self.satellite_position(sat_idx, t)
        # HAP 到地心距离
        r_h = C.R_e + C.HAP_ALT
        # 球面角距 ψ (HAP-地心-卫星)
        dlat = lat_s - self.hap_lat
        dlon = lon_s - self.hap_lon
        cos_psi = (np.sin(self.hap_lat) * np.sin(lat_s) +
                   np.cos(self.hap_lat) * np.cos(lat_s) * np.cos(dlon))
        cos_psi = np.clip(cos_psi, -1.0, 1.0)
        psi = np.arccos(cos_psi)
        # 仰角公式: sin(el) = cos(ψ) − (R_e/H_i) / sqrt(1 − (R_e/H_i)^2 sin^2(ψ))
        # 标准卫星-地面仰角
        Re_over_r = C.R_e / r_s
        denom = np.sqrt(1.0 - (Re_over_r * np.sin(psi)) ** 2)
        sin_el = (np.cos(psi) - Re_over_r) / denom if denom > 0 else -1.0
        sin_el = np.clip(sin_el, -1.0, 1.0)
        el = np.arcsin(sin_el)
        return np.rad2deg(el)

    def visibility_mask_all(self, t):
        """
        Eq.24: M_t[i] = 1 若卫星 i 在时刻 t 对 HAP 可见(仰角 ≥ mask).
        返回 bool 数组 (N_LEO,)
        """
        mask = np.zeros(C.N_LEO, dtype=bool)
        for i in range(C.N_LEO):
            mask[i] = self.elevation_to_hap(i, t) >= C.ELEV_MASK_DEG
        # 保证至少有 1 颗可见: 若全不可见, 取仰角最高的几颗
        if not mask.any():
            els = np.array([self.elevation_to_hap(i, t) for i in range(C.N_LEO)])
            top = np.argsort(els)[-3:]
            mask[top] = True
        return mask


# ============================================================================
# MDP 环境
# ============================================================================
class SatelliteHAPEnv:
    """
    一个 episode (60 步 × 1 min) 的 MDP.
    动作: 选择下一时刻服务卫星 idx (离散, N_LEO)
    奖励: η·R_total_norm − ζ·I(切换)
    """

    def __init__(self, seed: int):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        # *** 根因2 修复: 星座几何用固定 seed, 与 run seed 无关 → 环境近确定性 ***
        # run 的 seed 仅影响: torch NN 初始化, 每步 Nakagami/Gamma-Gamma 衰落随机, LLM rng.
        const_rng = np.random.default_rng(C.CONSTELLATION_SEED)
        self.constellation = Constellation(const_rng)

        # *** 根因2 修复: 固定簇距离 (不用 uniform, 避免 seed 间 3x 方差) ***
        self.cluster_d_hc = np.array(C.CLUSTER_DIST_FIXED, dtype=float)
        # 每簇的固定天线增益 R_i (dB), 假设 12 dBi 终端天线
        self.r_i_db = np.full(C.N_CLUSTER, 12.0)

        # *** 根因2 修复: 默认晴空低衰减 (0-1.5 dB), 0-20dB 仅 Fig.8 用 ***
        self.atm_atten = self.rng.uniform(C.A_ATM_CLEAR_MIN, C.A_ATM_CLEAR_MAX,
                                          size=C.N_STEP)
        # 缓存: 卫星-步 的仰角(惰性填充, 避免每步重复算)
        self._elev_cache = {}

        # 预计算每步可见性掩码 + 每颗可见卫星到 HAP 的 FSO 距离 (用斜距)
        # 距离近似: d_sh ≈ sqrt(H_i^2 − R_e^2 sin^2(ψ)) ... 这里用轨道高度+斜距近似
        self.visibility = np.zeros((C.N_STEP, C.N_LEO), dtype=bool)
        # 服务卫星的 FSO 斜距(到 HAP) 与 RF 各簇斜距轨迹在 step 内即时算

        self._precompute()

        # MDP 状态变量
        self.t = 0
        self.cur_sat = None      # 当前服务卫星 idx
        self.n_handover = 0
        self.cum_data = np.zeros(C.N_CLUSTER)  # 各簇累计归一化数据

    def _precompute(self):
        """预计算整 episode 的可见性掩码与所有 (步,卫星) 的仰角.

        *** 根因1 修复: 同时预计算 R_FSO_grid[timestep, satellite] (大尺度 FSO 速率) ***
        用该步 atm + 大尺度增益 (h_l), 不含随机湍流 (gamma_gamma_sample 用固定值 1).
        然后自校准参考速率:
          R_FSO_REF = percentile(R_FSO_grid[visibility], 55)  (物理非硬编码)
          R_RF_REF  = ΣR_RF (大尺度, 固定簇距下确定)
        使约 45% 可见星 FSO 高于 ref, 55% 低于 ref → 选星影响 f1.
        """
        N, S = C.N_LEO, C.N_STEP
        self.elev_grid = np.zeros((S, N))
        # *** 根因1: 预计算 R_FSO_grid (大尺度, 无湍流随机) ***
        self.R_FSO_grid = np.zeros((S, N))
        for k in range(S):
            t_phys = k * C.DT
            for i in range(N):
                self.elev_grid[k, i] = self.constellation.elevation_to_hap(i, t_phys)
            self.visibility[k] = self.elev_grid[k] >= C.ELEV_MASK_DEG
            if not self.visibility[k].any():
                # 退化: 取仰角最高的 3 颗保证可见
                top = np.argsort(self.elev_grid[k])[-3:]
                self.visibility[k, top] = True
            # 计算该步每颗可见卫星的大尺度 FSO 速率 (无湍流, h_a=1)
            a_atm_k = self.atm_atten[k]
            for i in range(N):
                if self.visibility[k, i]:
                    d_sh = self._fso_distance(i, k)
                    # 大尺度 FSO SNR: h_a=1 (无湍流), 用 N_A 孔径但 h_SH_q 全等于 h_l
                    h_l_db = ch.fso_static_gain_db(d_sh, a_atm_k)
                    h_l_lin = 10.0 ** (h_l_db / 10.0)
                    # h_EGC = N_A * h_l (等增益合并, 无湍流时各孔径相同)
                    h_egc = C.N_APERTURE * h_l_lin
                    gamma_h = C.P_FSO * (C.ETA_OE ** 2) * (h_egc ** 2) / \
                              (C.N_APERTURE * C.N_Q)
                    self.R_FSO_grid[k, i] = ch.fso_rate(gamma_h)   # bps

        # === 自校准参考速率 (物理非硬编码) ===
        vis_rates = self.R_FSO_grid[self.visibility]
        # 55 分位: 约 45% 可见星 FSO 高于 ref, 55% 低于 ref
        self.R_FSO_REF = float(np.percentile(vis_rates, 55))

        # R_RF_REF: 大尺度 ΣR_RF (固定簇距下确定, 无 Nakagami 随机; g=1)
        n_per = C.N_SUB // C.N_CLUSTER
        n_alloc = np.full(C.N_CLUSTER, n_per, dtype=int)
        rem = C.N_SUB - n_alloc.sum()
        if rem > 0:
            nearest = int(np.argmin(self.cluster_d_hc))
            n_alloc[nearest] += rem
        sum_R_RF_large = 0.0
        for i in range(C.N_CLUSTER):
            c_db = ch.rf_large_scale_db(self.cluster_d_hc[i], self.r_i_db[i])
            c_lin = 10.0 ** (c_db / 10.0)
            # g=1 (无 Nakagami 随机), |h|^2 = c_lin^2
            sum_R_RF_large += ch.rf_rate(n_alloc[i], c_lin)
        self.R_RF_REF = float(sum_R_RF_large)

    def _fso_distance(self, sat_idx, t_step):
        """卫星到 HAP 的 FSO 斜距 (m). 用仰角+球面几何."""
        k = min(t_step, C.N_STEP - 1)
        el = np.deg2rad(self.elev_grid[k, sat_idx])
        h_orbit = self.constellation.altitudes[sat_idx]
        # 斜距近似: d = sqrt((R_e+h)^2 − (R_e·cos(el))^2) − R_e·sin(el)
        rs = C.R_e + h_orbit
        d = np.sqrt(rs ** 2 - (C.R_e * np.cos(el)) ** 2) - C.R_e * np.sin(el)
        return float(max(d, 1.0))

    def reset(self):
        self.t = 0
        self.n_handover = 0
        self.cum_data = np.zeros(C.N_CLUSTER)
        # 初始服务卫星: 第 0 步可见集合中仰角最高的
        vis = self.visibility[0]
        cands = np.where(vis)[0]
        if len(cands) == 0:
            cands = np.arange(C.N_LEO)
        els = self.elev_grid[0, cands]
        self.cur_sat = int(cands[np.argmax(els)])
        return self._get_state()

    def _get_state(self):
        """
        Eq.18: s[t] = {t, s_t(当前卫星), N_t(累计切换), D(t)(各簇累计数据), H(t)(信道条件)}
        编码为固定维向量喂入 MLP:
        - 1 维 时间归一化 t/N_STEP
        - 1 维 当前卫星轨道半径归一化 (区分 500km/1000km 两层)
        - 1 维 当前卫星 FSO 速率归一化 (R_FSO_grid[cur]/R_FSO_REF) *** 根因1 关键 ***
        - 1 维 累计切换数归一化
        - N_CLUSTER 维 各簇累计数据归一化
        - N_CLUSTER 维 各簇当前 RF 信道条件 (大尺度 dB 归一化)
        - N_CLUSTER 维 当前各簇距 HAP 水平距离归一化
        总维: 4 + 3*N_CLUSTER

        注: 当前卫星 FSO 归一化速率让 agent 从状态直接判断"当前星好不好",
        是否值得切换到更好的可见星 (可见星集合由 action_mask 给出, agent 通过
        探索学习各可见星的 FSO 质量 → 选高于 ref 的星).
        """
        t_norm = self.t / C.N_STEP
        cur_radius_norm = (self.constellation.orbit_radius[self.cur_sat] - C.R_e) / C.H_HIGH
        # 当前卫星 FSO 归一化速率 (大尺度, 无湍流)
        k_step = min(self.t, C.N_STEP - 1)
        cur_rfso_norm = self.R_FSO_grid[k_step, self.cur_sat] / self.R_FSO_REF
        ho_norm = self.n_handover / C.N_STEP
        data_norm = self.cum_data / max(self.cum_data.max() + 1e-6, 1.0)
        # 各簇 RF 大尺度增益(dB) 与距离
        c_db = np.array([ch.rf_large_scale_db(self.cluster_d_hc[i], self.r_i_db[i])
                         for i in range(C.N_CLUSTER)])
        c_db_norm = (c_db + 200.0) / 200.0   # 简单归一化 (典型 dB 范围 ~-150~-50)
        dist_norm = self.cluster_d_hc / C.CLUSTER_DIST_MAX
        state = np.concatenate([
            [t_norm, cur_radius_norm, cur_rfso_norm, ho_norm],
            data_norm, c_db_norm, dist_norm
        ]).astype(np.float32)
        return state

    @property
    def state_dim(self):
        return 4 + 3 * C.N_CLUSTER

    @property
    def action_dim(self):
        return C.N_LEO

    def action_mask(self):
        """Eq.24: 当前步可见掩码 (合法动作集)."""
        return self.visibility[self.t].copy()

    def step(self, action: int):
        """
        执行动作: 选择下一时刻服务卫星.
        action: 卫星 idx (已被 agent 通过 mask 约束在可见集合内)
        """
        # 切换计数 (Eq.20 indicator)
        switched = (action != self.cur_sat)
        if switched:
            self.n_handover += 1
        prev_sat = self.cur_sat
        self.cur_sat = int(action)
        self.t += 1

        # === 物理层速率 (channels.py Eq.4-10, 严格不动) ===
        t_phys = self.t * C.DT
        # FSO: 当前服务卫星→HAP (含随机湍流, 用于物理真实性)
        d_sh = self._fso_distance(self.cur_sat, self.t)
        a_atm = self.atm_atten[min(self.t, C.N_STEP - 1)]
        h_sh_arr = ch.fso_channel_gain_linear(d_sh, a_atm, self.rng,
                                              n_aperture=C.N_APERTURE)
        gamma_h = ch.fso_snr(h_sh_arr)
        R_FSO = ch.fso_rate(gamma_h)  # bps  (Eq.7)

        # RF: HAP→各簇; 子载波全分配 (Eq.17 等式 Σn_i=N_S, 按簇间等量基线+微小扰动)
        # 这里: 子载波均分给 3 簇 (每个 n_i = N_S // N_CLUSTER, 余数给最近簇)
        n_per = C.N_SUB // C.N_CLUSTER
        n_alloc = np.full(C.N_CLUSTER, n_per, dtype=int)
        rem = C.N_SUB - n_alloc.sum()
        if rem > 0:
            # 余数给距 HAP 最近的簇(信道最好)
            nearest = int(np.argmin(self.cluster_d_hc))
            n_alloc[nearest] += rem

        R_RF = np.zeros(C.N_CLUSTER)
        for i in range(C.N_CLUSTER):
            h_hc_i = ch.rf_channel_gain(self.cluster_d_hc[i], self.r_i_db[i], self.rng)
            R_RF[i] = ch.rf_rate(n_alloc[i], h_hc_i)   # bps (Eq.10)

        # 流守恒 (Eq.11) & 瓶颈比例分配 (Eq.14) — 物理量, 用于 R_total_gbps 调试
        sum_R_RF = R_RF.sum()
        if sum_R_RF <= R_FSO:
            R_total_phys = sum_R_RF                    # FSO 不瓶颈
            R_RF_alloc = R_RF.copy()
        else:
            R_total_phys = R_FSO                       # FSO 瓶颈
            # Eq.14: 按比例分配
            R_RF_alloc = R_RF / sum_R_RF * R_FSO

        # 累计数据 (各簇)
        self.cum_data += R_RF_alloc

        # *** 根因1 修复: 归一化 min() 让 FSO/RF 可比, 使卫星选择影响 f1 ***
        # R_FSO_norm = R_FSO(cur_sat) / R_FSO_REF  (用预计算的大尺度 grid 值, 无湍流随机)
        # 用 grid 值而非瞬时 R_FSO 是为了让 agent 能从状态推断卫星好坏 (确定性映射),
        # 否则湍流随机会让同一卫星同一时刻速率乱跳, agent 学不到稳定策略.
        k_step = min(self.t, C.N_STEP - 1)
        R_FSO_for_norm = self.R_FSO_grid[k_step, self.cur_sat]
        R_FSO_norm = R_FSO_for_norm / self.R_FSO_REF
        R_RF_norm = sum_R_RF / self.R_RF_REF
        R_total_norm = min(R_FSO_norm, R_RF_norm)
        # 钳位到合理范围 (避免极端湍流/衰落导致超出)
        R_total_norm = float(np.clip(R_total_norm, 0.0, 1.1))

        # === 切换中断 (作用在 R_total_norm 上) ===
        # 切换时 R_total_norm *= (1 - HANDOVER_OUTAGE). 少切换算法 f1 略高 (论文 0.44% 增益).
        if switched:
            R_total_norm = R_total_norm * (1.0 - C.HANDOVER_OUTAGE)

        # === 奖励 (Eq.13, Eq.20): r = η·R_total_norm − ζ·I(switch) ===
        reward = C.ETA_R * R_total_norm - C.ZETA_HO * (1.0 if switched else 0.0)

        done = (self.t >= C.N_STEP)
        info = {
            "R_FSO_gbps": ch.rate_to_reward_gbps(R_FSO),
            "R_RF_sum_gbps": ch.rate_to_reward_gbps(sum_R_RF),
            "R_total_gbps": ch.rate_to_reward_gbps(R_total_phys),
            "R_total_norm": R_total_norm,    # 无量纲归一化速率 (用于 f1 统计)
            "f1_increment": R_total_norm,    # 归一化速率贡献 (用于 f1 累加)
            "handover": int(switched),
            "n_handover": self.n_handover,
        }
        return self._get_state(), float(reward), done, info
