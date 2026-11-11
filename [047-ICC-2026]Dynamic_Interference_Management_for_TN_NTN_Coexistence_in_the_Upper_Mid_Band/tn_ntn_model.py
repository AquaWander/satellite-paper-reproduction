"""TN–NTN时变物理环境：逐站动作、上下行干扰和吞吐量。

固定一整次过境的布局和链路随机量，支持各方法同种子公平对照。
无真实Frankfurt坐标时采用显式标注的合成布局。
"""
from __future__ import annotations

from dataclasses import asdict
import numpy as np
from scipy.special import j1
from config import CFG, SimConfig

C = 299_792_458.0


def db_to_linear(x):
    return np.power(10.0, np.asarray(x) / 10.0)


def linear_to_db(x):
    return 10.0 * np.log10(np.maximum(x, 1e-30))


def fspl_db(distance_m, frequency_hz):
    """距离米、频率Hz，自由空间损耗dB。"""
    return 20 * np.log10(np.maximum(distance_m, 1.0)) + 20 * np.log10(frequency_hz) + 20 * np.log10(4 * np.pi / C)


def slant_distance_m(elevation_deg, altitude_m=600e3, earth_radius_m=6371e3):
    """Eq.(1)减号项位于根号外；10–170°视作过境扫描角。"""
    s = np.sin(np.deg2rad(elevation_deg))
    return np.sqrt((earth_radius_m * s) ** 2 + altitude_m ** 2 + 2 * altitude_m * earth_radius_m) - earth_radius_m * s


def uma_path_loss_db(distance_m, frequency_hz, los, h_bs=25.0, h_ut=1.5):
    """原文Eq.(4) LOS断点分段，NLOS用38.901常见UMa形式。

    >5km链路延续对数斜率，属超出标准标定范围的明确外推假设。
    """
    d2 = np.maximum(np.asarray(distance_m), 10.0)
    d3 = np.sqrt(d2 * d2 + (h_bs - h_ut) ** 2)
    fc = frequency_hz / 1e9
    bp = 4 * max(h_bs - 1, 0.1) * max(h_ut - 1, 0.1) * frequency_hz / C
    near = 32.4 + 20 * np.log10(d3) + 20 * np.log10(fc)
    far = 32.4 + 40 * np.log10(d3) + 20 * np.log10(fc) - 10 * np.log10(bp * bp + (h_bs - h_ut) ** 2)
    pl_los = np.where(d2 < bp, near, far)
    nlos = 13.54 + 39.08 * np.log10(d3) + 20 * np.log10(fc) - 0.6 * (h_ut - 1.5)
    return np.where(np.asarray(los, dtype=bool), pl_los, np.maximum(pl_los, nlos))


def los_probability(distance_m):
    d = np.maximum(np.asarray(distance_m), 1.0)
    return np.minimum(18.0 / d, 1.0) * (1 - np.exp(-d / 63.0)) + np.exp(-d / 63.0)


def aperture_radius_m(peak_gain_dbi, frequency_hz=12e9, efficiency=0.65):
    return C / frequency_hz / (2 * np.pi) * np.sqrt(db_to_linear(peak_gain_dbi) / efficiency)


def bessel_gain_db(off_axis_deg, peak_dbi, frequency_hz=12e9, efficiency=0.65, back_attenuation_db=55.0, rear_model='relative_floor'):
    """Eq.(3)圆孔径Bessel图；孔径由增益/效率推导，T1/T2逐用户独立。

    原式只定义前半球，背面衰减为工程假设；数值下限仅防log(0)。
    """
    theta = np.abs(np.asarray(off_axis_deg))
    if np.any(~np.isfinite(theta)) or np.any(theta > 180):
        raise ValueError('离轴角须为有限数且绝对值不超过180度')
    peak = np.asarray(peak_dbi)
    radius = aperture_radius_m(peak, frequency_hz, efficiency)
    x = 2 * np.pi * frequency_hz / C * radius * np.sin(np.deg2rad(theta))
    factor = np.ones(np.broadcast_shapes(np.shape(x), np.shape(peak)), dtype=float)
    np.divide(2 * j1(x), x, out=factor, where=np.abs(x) > 1e-10)
    if rear_model == 'relative_floor':
        pattern = np.where(theta <= 90.0, factor ** 2, db_to_linear(-back_attenuation_db))
    elif rear_model == 'symmetric':
        # 仅作为未知后半球的敏感性：sin(theta)关于90度对称，非宣称真实背瓣。
        pattern = factor ** 2
    else:
        raise ValueError('rear_model需为relative_floor或symmetric')
    return peak + linear_to_db(np.maximum(pattern, 1e-12))


def ut_gain_db(off_axis_deg, peak_dbi, cfg=CFG):
    """用户接收图的两种显式假设；不将卫星Eq.(3)误称为UT规定。

    wbes_envelope在6°以上遵循ETSI/官方离轴增益上限包络，6°以下用同一
    对数式延拓并以用户33/17dBi峰值封顶；主瓣完成方式不是ETSI原规定。
    """
    if cfg.ut_antenna_model == 'bessel':
        return bessel_gain_db(off_axis_deg, peak_dbi, cfg.frequency_hz,
            cfg.aperture_efficiency, cfg.antenna_back_attenuation_db, cfg.antenna_rear_model)
    if cfg.ut_antenna_model == 'wbes_envelope':
        theta = np.abs(np.asarray(off_axis_deg))
        if np.any(~np.isfinite(theta)) or np.any(theta > 180):
            raise ValueError('离轴角须为有限数且绝对值不超过180度')
        envelope = np.where(theta < 48, 40 - 25 * np.log10(np.maximum(theta, 1e-12)), -2.0)
        return np.minimum(np.asarray(peak_dbi), envelope)
    raise ValueError('ut_antenna_model需为bessel或wbes_envelope')


def ascent_array_gain_db(horizontal_deg, downward_deg, tilt_deg):
    """独立闭式16×16、半波长UPA，逐角度核验官方gain_5g至1e-11dB。

    8dBi单元、H/V HPBW80°/65°来自固定官方版本。真实几何给出zenith，
    其weighting正tilt朝上，因此本接口的正下倾以负号传入；主服务束phi=0。
    """
    phi, down, tilt = np.broadcast_arrays(horizontal_deg, downward_deg, tilt_deg)
    theta = 90.0 + down
    th, ph, ti = np.deg2rad(theta), np.deg2rad(phi), np.deg2rad(-tilt)
    qv = 0.5 * (np.cos(th) - np.sin(ti))
    qh = 0.5 * np.sin(th) * np.sin(ph)
    def normalized_sum(q):
        denominator = np.sin(np.pi * q)
        value = np.ones_like(q)
        np.divide(np.sin(16 * np.pi * q), 16 * denominator, out=value,
                  where=np.abs(denominator) > 1e-12)
        return value
    array_power = 256 * normalized_sum(qv) ** 2 * normalized_sum(qh) ** 2
    element = 8.0 - np.minimum(np.minimum(12 * (phi / 80.0) ** 2, 30.0)
                              + np.minimum(12 * ((theta - 90.0) / 65.0) ** 2, 30.0), 30.0)
    return element + linear_to_db(array_power)


def gnb_gain_db(horizontal_deg, downward_deg, tilt_deg, cfg=CFG):
    """主profile用Eq.(5)；官方兼容profile用实际服务束阵列，单位为度。"""
    if cfg.gnb_antenna_model == 'ascent_array':
        return ascent_array_gain_db(horizontal_deg, downward_deg, tilt_deg)
    if cfg.gnb_antenna_model != 'parabolic':
        raise ValueError('gnb_antenna_model需为parabolic或ascent_array')
    ah = np.minimum(12 * (np.asarray(horizontal_deg) / cfg.gnb_horizontal_hpbw_deg) ** 2, cfg.gnb_front_back_db)
    av = np.minimum(12 * ((np.asarray(downward_deg) - np.asarray(tilt_deg)) / cfg.gnb_vertical_hpbw_deg) ** 2, cfg.gnb_sidelobe_db)
    return cfg.gnb_peak_gain_dbi - np.minimum(ah + av, cfg.gnb_front_back_db)


def _angle_deg(vectors_a, vectors_b):
    dot = np.sum(vectors_a * vectors_b, axis=-1)
    den = np.linalg.norm(vectors_a, axis=-1) * np.linalg.norm(vectors_b, axis=-1)
    return np.rad2deg(np.arccos(np.clip(dot / np.maximum(den, 1e-30), -1, 1)))


class CoexistenceEnv:
    """obs=global5+flatten(local[n_gnb,4])；action[n_gnb,4]。

    action前三列归一化[-1,1]的DL/UL/tilt，末列为0/1站级活动开关。
    每站活动开关共同控制其下行和调度上行，是单控制单元/站的抽象。
    """
    global_dim = 5
    local_dim = 4

    def __init__(self, cfg: SimConfig = CFG, seed=0, density=None, t1_fraction=None):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.fixed_density = density
        self.fixed_t1_fraction = t1_fraction
        self.obs_dim = 5 + cfg.n_gnb * 4
        self.action_shape = (cfg.n_gnb, 4)
        self._ready = False

    def reset(self, seed=None, density=None, t1_fraction=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        c = self.cfg
        if c.propagation_model not in ('icc_los', 'uma_mixed'):
            raise ValueError('propagation_model需为icc_los或uma_mixed')
        if c.ut_antenna_model not in ('bessel', 'wbes_envelope'):
            raise ValueError('ut_antenna_model需为bessel或wbes_envelope')
        if c.gnb_antenna_model not in ('parabolic', 'ascent_array'):
            raise ValueError('gnb_antenna_model需为parabolic或ascent_array')
        if c.noise_temperature_k <= 0:
            raise ValueError('noise_temperature_k必须为正')
        d = density if density is not None else self.fixed_density
        f = t1_fraction if t1_fraction is not None else self.fixed_t1_fraction
        self.density = float(self.rng.choice(c.density_values) if d is None else d)
        self.t1_fraction = float(self.rng.choice(c.mix_values) if f is None else f)
        if self.density <= 0 or not 0 <= self.t1_fraction <= 1:
            raise ValueError('density应为正数，t1_fraction应位于[0,1]')
        self.n_ntn = max(1, int(round(self.density * c.area_side_m ** 2)))
        if c.bs_coordinates_csv:
            self.gnb_xy = np.loadtxt(c.bs_coordinates_csv, delimiter=',', skiprows=1)
            if self.gnb_xy.shape != (c.n_gnb, 2) or not np.all(np.isfinite(self.gnb_xy)):
                raise ValueError('站址CSV需含x_m,y_m表头和106行有限局部米坐标')
        else:
            self.gnb_xy = self.rng.uniform(0, c.area_side_m, (c.n_gnb, 2))
        # 条件于论文人数的PPP：固定人数时位置为独立均匀分布。
        self.ntn_xy = self.rng.uniform(0, c.area_side_m, (self.n_ntn, 2))
        self.is_t1 = np.zeros(self.n_ntn, dtype=bool)
        self.is_t1[:int(round(self.n_ntn * self.t1_fraction))] = True
        self.rng.shuffle(self.is_t1)
        self.terminal_peak_dbi = np.where(self.is_t1, c.t1_gain_dbi, c.t2_gain_dbi)
        # PPP候选UE最近站关联，每站保留12–20服务用户，每快照调度一个UL流。
        n_pool = max(c.n_gnb, int(self.rng.poisson(c.tn_ue_density_m2 * c.area_side_m ** 2)))
        pool = self.rng.uniform(0, c.area_side_m, (n_pool, 2))
        assoc = np.argmin(np.sum((pool[:, None, :] - self.gnb_xy[None, :, :]) ** 2, axis=-1), axis=1)
        self.ue_counts = self.rng.integers(c.ue_per_cell_min, c.ue_per_cell_max + 1, c.n_gnb)
        self.ue_xy = np.empty((c.n_snapshots, c.n_gnb, 2))
        self.empty_cell_fallbacks = 0
        self.cells_resampled_with_replacement = 0
        for i in range(c.n_gnb):
            candidates = pool[assoc == i]
            if len(candidates) == 0:
                # 极小Voronoi小区无候选时补一个零水平距离用户；记入metadata。
                candidates = self.gnb_xy[i:i+1].copy()
                self.empty_cell_fallbacks += 1
            if len(candidates) < self.ue_counts[i]:
                self.cells_resampled_with_replacement += 1
            chosen = candidates[self.rng.choice(len(candidates), int(self.ue_counts[i]), replace=len(candidates) < self.ue_counts[i])]
            self.ue_xy[:, i] = chosen[np.arange(c.n_snapshots) % len(chosen)]
        self.ue_pool_count = n_pool
        self._precompute()
        self.t = 0
        self._ascent_radius_m = 0.0
        self._previous_active = np.ones(c.n_gnb, dtype=float)
        self._last_info = self._evaluate(self.reference_action(), 0)
        self._ready = True
        return self.observe()

    def _precompute(self):
        c = self.cfg
        self.sweep_angles_deg = c.elevation_start_deg + np.arange(c.n_snapshots) * c.elevation_step_deg
        self.elevations_deg = np.minimum(self.sweep_angles_deg, 180 - self.sweep_angles_deg)
        distance = slant_distance_m(self.sweep_angles_deg, c.satellite_altitude_m, c.earth_radius_m)
        a = np.deg2rad(self.sweep_angles_deg)
        center = np.array([c.area_side_m / 2, c.area_side_m / 2, c.terminal_height_m])
        self.sat_xyz = center + np.stack([distance * np.cos(a), np.zeros_like(a), distance * np.sin(a)], axis=-1)
        self.gnb_xyz = np.column_stack([self.gnb_xy, np.full(c.n_gnb, c.gnb_height_m)])
        self.ntn_xyz = np.column_stack([self.ntn_xy, np.full(self.n_ntn, c.terminal_height_m)])
        self.ue_xyz = np.concatenate([self.ue_xy, np.full((c.n_snapshots, c.n_gnb, 1), c.terminal_height_m)], axis=-1)
        serving_vec = self.ue_xy[0] - self.gnb_xy
        self.boresight_azimuth = np.arctan2(serving_vec[:, 1], serving_vec[:, 0])
        vec_dl = self.ntn_xyz[None, :, :] - self.gnb_xyz[:, None, :]
        d2_dl = np.linalg.norm(vec_dl[:, :, :2], axis=-1)
        az_dl = np.arctan2(vec_dl[:, :, 1], vec_dl[:, :, 0])
        self.horizontal_deg = np.abs(np.rad2deg((az_dl - self.boresight_azimuth[:, None] + np.pi) % (2 * np.pi) - np.pi))
        self.downward_deg = np.rad2deg(np.arctan2(c.gnb_height_m - c.terminal_height_m, np.maximum(d2_dl, 1.0)))
        drawn_los_dl = self.rng.random(d2_dl.shape) < los_probability(d2_dl)
        los_dl = np.ones_like(drawn_los_dl) if c.propagation_model == 'icc_los' else drawn_los_dl
        pl_dl = uma_path_loss_db(d2_dl, c.frequency_hz, los_dl, c.gnb_height_m, c.terminal_height_m)
        shadow_dl = self.rng.normal(0, np.where(los_dl, 4.0, 6.0))
        if c.shadow_fading:
            pl_dl += shadow_dl
        sat_vec = self.sat_xyz[:, None, :] - self.ntn_xyz[None, :, :]
        rx_dl = self.gnb_xyz[None, :, None, :] - self.ntn_xyz[None, None, :, :]
        off_dl = _angle_deg(sat_vec[:, None, :, :], rx_dl)
        gain_dl = ut_gain_db(off_dl, self.terminal_peak_dbi[None, None, :], c)
        self.dl_coupling_mw_per_mw = db_to_linear(gain_dl - pl_dl[None, :, :]).astype(np.float32)
        vec_ul = self.ue_xyz[:, :, None, :] - self.ntn_xyz[None, None, :, :]
        d2_ul = np.linalg.norm(vec_ul[:, :, :, :2], axis=-1)
        # UE–NTN均近地，使用UMa大尺度锚定高度外推，并明确作为假设。
        drawn_los_ul = self.rng.random((c.n_gnb, self.n_ntn)) < los_probability(np.mean(d2_ul, axis=0))
        los_ul = np.ones_like(drawn_los_ul) if c.propagation_model == 'icc_los' else drawn_los_ul
        pl_ul = uma_path_loss_db(d2_ul, c.frequency_hz, los_ul[None, :, :], c.gnb_height_m, c.terminal_height_m)
        shadow_ul = self.rng.normal(0, np.where(los_ul, 4.0, 6.0))[None, :, :]
        if c.shadow_fading:
            pl_ul += shadow_ul
        off_ul = _angle_deg(sat_vec[:, None, :, :], vec_ul)
        gain_ul = ut_gain_db(off_ul, self.terminal_peak_dbi[None, None, :], c)
        self.ul_coupling_mw_per_mw = db_to_linear(gain_ul - pl_ul).astype(np.float32)
        d_sat = np.linalg.norm(sat_vec, axis=-1)
        # 卫星波束始终跟踪区域中心，用户在波束内的离轴增益同样使用Eq.(3)。
        sat_to_center = center[None, :] - self.sat_xyz
        sat_to_ntn = self.ntn_xyz[None, :, :] - self.sat_xyz[:, None, :]
        satellite_offaxis = _angle_deg(sat_to_center[:, None, :], sat_to_ntn)
        satellite_gain = bessel_gain_db(satellite_offaxis, c.satellite_gain_dbi, c.frequency_hz,
                                       c.aperture_efficiency, c.antenna_back_attenuation_db, c.antenna_rear_model)
        signal_dbm = c.satellite_power_dbm + satellite_gain + self.terminal_peak_dbi[None, :] - fspl_db(d_sat, c.frequency_hz) - c.ntn_extra_loss_db
        self.signal_mw = db_to_linear(signal_dbm)
        # Eq.(7)的kTB是总噪声功率；默认不额外加入论文未给出的7dB NF。
        self.noise_dbm = float(linear_to_db(1.38064852e-23 * c.noise_temperature_k * c.bandwidth_hz * 1000)) + c.noise_figure_db
        self.noise_mw = float(db_to_linear(self.noise_dbm))
        self.nearest_ntn_distance_m = np.min(d2_dl, axis=1)
        self.service_distance_m = np.linalg.norm(serving_vec, axis=1)
        self._default_gnb_gain = gnb_gain_db(self.horizontal_deg, self.downward_deg, c.default_tilt_deg, c)
        self._default_dl_weight = db_to_linear(c.gnb_max_power_dbm + self._default_gnb_gain) * c.downlink_duty
        self._default_ul_weight = db_to_linear(c.ue_max_power_dbm) * c.uplink_duty
        self.default_contributions_mw = (self.dl_coupling_mw_per_mw * self._default_dl_weight[None, :, :] + self.ul_coupling_mw_per_mw * self._default_ul_weight)

    def reference_action(self):
        c = self.cfg
        a = np.ones((c.n_gnb, 4), dtype=np.float32)
        a[:, 2] = 2 * (c.default_tilt_deg - c.tilt_min_deg) / (c.tilt_max_deg - c.tilt_min_deg) - 1
        return a

    def baseline_action(self, method='No-coordination'):
        a = self.reference_action()
        if method == 'No-coordination':
            return a
        if method not in ('ASCENT', 'ASCENT-style', 'ASCENT-derived'):
            raise ValueError(f'未知基线{method}')
        # 官方ASCENTController的实际gate是“每个MBS反馈INR超阈即静默”，
        # 并非约束所有存活站的聚合INR。这里不复制上游固定数组覆盖反馈的bug。
        # 多NTN扩展采用任一接收端超阈即静默；下一快照重新判断，允许恢复。
        c = self.cfg
        contributions = self.default_contributions_mw[min(self.t, c.n_snapshots - 1)]
        limit = self.noise_mw * db_to_linear(c.inr_threshold_db)
        a[:, 3] = np.max(contributions, axis=1) <= limit
        # 不实施公共排除圆；保留历史trace字段为0，避免把描述半径当成动作。
        self._ascent_radius_m = 0.0
        return a

    def _decode_action(self, action):
        if isinstance(action, dict):
            action = np.column_stack([action['continuous'], action['active']])
        action = np.asarray(action, dtype=float)
        if action.shape != self.action_shape or not np.all(np.isfinite(action)):
            raise ValueError(f'action需有限数组{self.action_shape}，收到{action.shape}')
        if np.any(np.abs(action[:, :3]) > 1.00001):
            raise ValueError('连续动作应预先映射到[-1,1]')
        if not np.all((action[:, 3] == 0) | (action[:, 3] == 1)):
            raise ValueError('active必须为二元0/1')
        c = self.cfg
        z = (np.clip(action[:, :3], -1, 1) + 1) / 2
        dl = c.gnb_min_power_dbm + z[:, 0] * (c.gnb_max_power_dbm - c.gnb_min_power_dbm)
        ul = c.ue_min_power_dbm + z[:, 1] * (c.ue_max_power_dbm - c.ue_min_power_dbm)
        tilt = c.tilt_min_deg + z[:, 2] * (c.tilt_max_deg - c.tilt_min_deg)
        return dl, ul, tilt, action[:, 3].astype(bool)

    def _evaluate(self, action, t):
        c = self.cfg
        dl, ul, tilt, active = self._decode_action(action)
        gain = gnb_gain_db(self.horizontal_deg, self.downward_deg, tilt[:, None], c)
        dl_contributions = self.dl_coupling_mw_per_mw[t] * db_to_linear(dl[:, None] + gain) * active[:, None] * c.downlink_duty
        ul_contributions = self.ul_coupling_mw_per_mw[t] * db_to_linear(ul[:, None]) * active[:, None] * c.uplink_duty
        dl_i = np.sum(dl_contributions, axis=0)
        ul_i = np.sum(ul_contributions, axis=0)
        per_bs_max_inr = linear_to_db(np.max(dl_contributions + ul_contributions, axis=1) / self.noise_mw)
        total = dl_i + ul_i
        inr = linear_to_db(total / self.noise_mw)
        sinr = self.signal_mw[t] / (total + self.noise_mw)
        throughput = float(c.bandwidth_hz * np.sum(np.log2(1 + sinr)))
        ideal = float(c.bandwidth_hz * np.sum(np.log2(1 + self.signal_mw[t] / self.noise_mw)))
        normalized = throughput / max(ideal, 1.0)
        violation = float(np.mean(inr > c.inr_threshold_db))
        active_ratio = float(np.mean(active))
        reward = c.reward_throughput_weight * normalized + c.reward_active_weight * active_ratio - c.reward_violation_weight * violation
        return {'inr_db': inr, 'sinr_db': linear_to_db(sinr), 'throughput_bps': throughput,
                'interference_free_throughput_bps': ideal, 'normalized_throughput': normalized,
                'active_ratio': active_ratio, 'violation_ratio': violation,
                'dl_interference_mw': dl_i, 'ul_interference_mw': ul_i,
                'dl_power_dbm': dl, 'ul_power_dbm': ul, 'tilt_deg': tilt,
                'active_mask': active, 'elevation_deg': float(self.elevations_deg[t]),
                'per_bs_max_inr_db': per_bs_max_inr,
                'sweep_angle_deg': float(self.sweep_angles_deg[t]), 'reward': float(reward),
                'ascent_radius_m': float(self._ascent_radius_m), 'n_ntn': self.n_ntn,
                't1_fraction': self.t1_fraction, 'density': self.density}

    def observe(self):
        c = self.cfg
        t = min(self.t, c.n_snapshots - 1)
        v = self._last_info
        glob = np.array([self.sweep_angles_deg[t] / 180, v['violation_ratio'], v['normalized_throughput'], v['active_ratio'], c.inr_threshold_db / 20])
        risk = linear_to_db(np.max(self.default_contributions_mw[t], axis=1) / self.noise_mw)
        local = np.column_stack([self.nearest_ntn_distance_m / c.area_side_m, np.clip((risk + 60) / 100, 0, 1),
                                 np.clip(self.service_distance_m / 5000, 0, 1), self._previous_active])
        return np.concatenate([glob, local.ravel()]).astype(np.float32)

    def step(self, action):
        if not self._ready or self.t >= self.cfg.n_snapshots:
            raise RuntimeError('先reset()，且不能在过境结束后继续step')
        info = self._evaluate(action, self.t)
        self._last_info = info
        self._previous_active = info['active_mask'].astype(float)
        self.t += 1
        return self.observe(), info['reward'], self.t == self.cfg.n_snapshots, info

    @property
    def metadata(self):
        return {'config': asdict(self.cfg), 'density': self.density, 't1_fraction': self.t1_fraction,
                'n_ntn': self.n_ntn, 'n_ue_candidates': self.ue_pool_count, 'n_served_ue': int(np.sum(self.ue_counts)),
                'empty_cell_fallbacks': self.empty_cell_fallbacks,
                'cells_resampled_with_replacement': self.cells_resampled_with_replacement,
                'layout': 'external coordinates' if self.cfg.bs_coordinates_csv else 'synthetic conditional-uniform BS layout; not Frankfurt original coordinates',
                'snapshots': self.cfg.n_snapshots, 'action_shape': list(self.action_shape), 'obs_dim': self.obs_dim}
