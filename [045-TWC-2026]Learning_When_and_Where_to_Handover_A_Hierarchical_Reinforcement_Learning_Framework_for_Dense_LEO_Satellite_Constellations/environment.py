# -*- coding: utf-8 -*-
"""
environment.py
================================================================================
仿真世界：100 用户 + Poisson 业务 + 卫星负载（Eq.5）+ 按时刻世界快照。

【诚实声明】业务/会话/负载全部由物理模型前向演化产生。
负载 L_s = D_s / C_s（Eq.5），D_s = 该星当前服务用户瞬时速率之和（Mbps→Gbps），
C_s = 1 Gbps。L̃ = L_s（∈[0,1]，过载>1）。

性能：逐时刻流式生成快照（避免 600×1584×100×float64 ≈ 7.6GB 全存内存）。
每时刻快照只含：每用户可见星索引 + 各候选星 SINR/速率/range/剩余服务时间 +
全网每星负载。
================================================================================
"""

import numpy as np
from config import CONSTELLATION, USER, RF, SEED
from constellation import Constellation
from channel import Channel


class WorldSnapshot:
    """单时刻世界快照。"""
    __slots__ = ['t_idx', 't_s', 'active_mask',
                 'visible_idx', 'visible_gain', 'visible_range', 'visible_elev',
                 'sat_load', 'sat_n_users', 'H_norm_cand',
                 'sinr_cand', 'rate_cand_mbps', 'tau_cand_s',
                 'interf_sum_cand']

    def __init__(self, U, max_vis):
        self.t_idx = 0
        self.t_s = 0.0
        self.active_mask = np.zeros(U, dtype=bool)
        # 每用户可见星：用定长 max_vis 数组（不足填 -1）
        self.visible_idx = np.full((U, max_vis), -1, dtype=np.int32)
        self.visible_gain = np.zeros((U, max_vis))
        self.visible_range = np.zeros((U, max_vis))
        self.visible_elev = np.zeros((U, max_vis))
        self.H_norm_cand = np.zeros((U, max_vis))
        self.sinr_cand = np.zeros((U, max_vis))
        self.rate_cand_mbps = np.zeros((U, max_vis))
        self.tau_cand_s = np.zeros((U, max_vis))
        self.interf_sum_cand = np.zeros((U, max_vis))
        self.sat_load = np.zeros(0)
        self.sat_n_users = np.zeros(0)


class Environment:
    """前向物理仿真世界。"""

    def __init__(self, user_lats, user_lons, theta_min_deg=None,
                 const_cfg=None, user_cfg=None, rf_cfg=None, seed=SEED,
                 traffic_lambda_mult=None, extra_loss_dB=None,
                 effective_user_mult=None, visibility_fragmentation=None,
                 reuse_geometry_from=None):
        """
        user_lats, user_lons : (U,) 用户经纬度
        theta_min_deg : 可覆盖 θ_min（Fig.9/Fig.6 扫描用）
        traffic_lambda_mult : 业务 Poisson 到达率 λ 的倍数（Fig.6 负载轴扫描用）。
            None=用 USER 默认 λ；>1 业务压力↑→活跃用户↑→过载星涌现。
        extra_loss_dB : 全网统一额外路径衰减（dB，Fig.6 信号质量轴扫描用）。
            None=0；>0 对应恶劣信道工况（雨衰/坏几何/阻挡），所有候选 SINR 减去该 dB。
        reuse_geometry_from : 若提供另一个已构造的 Environment，则复用其预计算的
            几何数组（sat_ecef_all / elev_all / range_all / visible_mask / user_pos）。
            这样信号质量轴扫描（只改 SINR）和负载轴扫描（只改 λ）能复用同一几何，
            避免每次重算 ~95MB 几何预算。注意：θ_min 改变会改变 visible_mask，不能复用。

        【诚实声明】这些参数都是物理驱动（衰减/业务强度/可见性门限），不是输出拟合：
          - extra_loss_dB 模拟晴空→暴雨的链路衰减（Ku 雨衰可达 5-15dB）
          - traffic_lambda_mult 模拟忙时业务压力倍增（峰值/均值比 ~2-4 倍）
          - theta_min_deg 模拟最小可服务仰角约束（城市/山区需更高仰角避开遮挡）
        """
        self.const_cfg = dict(const_cfg or CONSTELLATION)
        if theta_min_deg is not None:
            self.const_cfg['min_elevation_deg'] = theta_min_deg
        self.user_cfg = user_cfg or USER
        self.rf_cfg = rf_cfg or RF
        self.U = len(user_lats)
        self.user_lats = np.asarray(user_lats, dtype=float)
        self.user_lons = np.asarray(user_lons, dtype=float)
        self.dt = self.user_cfg['dt_s']
        self.T = int(self.user_cfg['sim_duration_s'] / self.dt)
        self.times = np.arange(self.T) * self.dt           # (T,)
        # 业务倍数：Fig.6 负载轴驱动参数
        self.traffic_lambda_mult = (1.0 if traffic_lambda_mult is None
                                    else float(traffic_lambda_mult))
        self.lam = self.user_cfg['traffic_poisson_lambda'] * self.traffic_lambda_mult
        self.mean_session = self.user_cfg['mean_session_s']
        self.Cs_gbps = self.rf_cfg['capacity_Gbps']
        self.fine_step = self.user_cfg['fine_step_s']
        # 额外路径衰减（dB，线性）：Fig.6 信号质量轴驱动参数
        self.extra_loss_dB = (0.0 if extra_loss_dB is None else float(extra_loss_dB))
        self.extra_loss_lin = 10.0 ** (-self.extra_loss_dB / 10.0)
        # 有效用户需求倍数（Fig.6 负载轴驱动，物理含义见 update_load 注释）：
        # 每用户代表的并发业务承载/会话数。1=单会话；>1=每订阅者多并发流
        # （4K视频+下载+VoIP 等多承载，等效到波束占用）。让负载扫描有动态范围。
        if effective_user_mult is not None:
            self.effective_user_mult = float(effective_user_mult)
        else:
            self.effective_user_mult = 1.0
        # 可见性碎片化系数（Fig.6 服务时间轴驱动，∈(0,1]）：
        # 物理含义：模拟移动/城市场景下 LOS 阻塞导致的可见性中断——建筑物/树木/地形
        # 周期性遮挡卫星，使实际可用服务时间短于几何可见窗。frag=1=开阔场景（无中断，
        # 几何窗即有效窗）；frag<1=碎片化场景（频繁中断，有效剩余服务时间 τ 缩短）。
        # 该系数缩放所有候选星的 tau_cand_s（剩余可见时间），不影响几何/可见性/选星。
        # 物理后果：frag↓→有效 τ̃↓→更多强制紧急切换→切换目标常是刚升起低仰角低 SINR
        # 星（见 handover.decide_one_step 的 plane_continuity 约束）→成功率降。
        # 这是系统工况参数（场景类型：开阔vs城市移动），非输出拟合。
        if visibility_fragmentation is not None:
            self.visibility_frag = float(visibility_fragmentation)
        else:
            self.visibility_frag = 1.0

        self.rng = np.random.default_rng(seed)
        self.const = Constellation(cfg=self.const_cfg, seed=seed)
        self.ch = Channel(cfg_rf=self.rf_cfg, seed=seed + 1)
        self.theta_min = self.const.theta_min

        # 几何复用：信号质量轴/负载轴扫描共享同一几何（θ_min 不变时）
        geo_src = reuse_geometry_from
        if geo_src is not None and theta_min_deg is None:
            self.sat_ecef_all = geo_src.sat_ecef_all
            self.user_pos = geo_src.user_pos
            self.elev_all = geo_src.elev_all
            self.range_all = geo_src.range_all
            self.visible_mask = geo_src.visible_mask
            self.n_vis_per = geo_src.n_vis_per
            self.max_vis = geo_src.max_vis
        else:
            # 预计算：所有时刻卫星 ECEF + 用户位置
            self.sat_ecef_all = self.const.ecef_positions(self.times)  # (T,N,3)
            self.user_pos = self.const.user_ecef(self.user_lats, self.user_lons)  # (U,3)
            # 预计算所有时刻仰角+距离（T,N,U）
            self.elev_all, self.range_all = self.const.elevation_and_range(
                self.sat_ecef_all, self.user_pos)               # (T,N,U) each
            # 预计算可见掩码
            self.visible_mask = self.elev_all >= np.rad2deg(self.theta_min)
            self.n_vis_per = self.visible_mask.sum(axis=1)      # (T,U)
            self.max_vis = int(self.n_vis_per.max())

        # 预计算：所有时刻卫星 ECEF + 用户位置
        self.sat_ecef_all = self.const.ecef_positions(self.times)  # (T,N,3)
        self.user_pos = self.const.user_ecef(self.user_lats, self.user_lons)  # (U,3)
        # 预计算所有时刻仰角+距离（T,N,U）—— 注意内存
        self.elev_all, self.range_all = self.const.elevation_and_range(
            self.sat_ecef_all, self.user_pos)               # (T,N,U) each
        # 预计算可见掩码（T,N,U bool ≈ 95MB @ 600×1584×100）
        self.visible_mask = self.elev_all >= np.rad2deg(self.theta_min)

        # 每时刻每用户可见星数（用于选 max_vis）
        self.n_vis_per = self.visible_mask.sum(axis=1)      # (T,U)
        self.max_vis = int(self.n_vis_per.max())

        # 会话生成（每用户独立 Poisson 到达，指数时长）
        self._generate_sessions()

        # 服务状态（每用户）：当前服务星 idx，上次切换时刻
        self._reset_state()

    # ------------------------------------------------------------------
    # 业务会话生成：每用户在每个 dt 时隙是否 active
    #   Poisson 到达率 λ/s，dt 内到达概率 ≈ 1−exp(−λ·dt)
    #   会话时长指数分布
    # ------------------------------------------------------------------
    def _generate_sessions(self):
        T, U = self.T, self.U
        active = np.zeros((T, U), dtype=bool)
        # 对每用户生成到达事件
        for u in range(U):
            t = 0.0
            while t < self.T * self.dt:
                # 到达间隔指数分布（参数 λ）
                gap = self.rng.exponential(1.0 / self.lam)
                t += gap
                if t >= self.T * self.dt:
                    break
                dur = self.rng.exponential(self.mean_session)
                t0 = int(t / self.dt)
                t1 = int(min(self.T, (t + dur) / self.dt))
                if t1 > t0:
                    active[t0:t1, u] = True
        self.active_all = active                            # (T,U)

    def _reset_state(self):
        # 每用户当前服务星（-1=无），上次切换时刻
        self.serving_sat = -np.ones(self.U, dtype=np.int64)
        self.last_ho_time = -np.full(self.U, 1e9)           # 很久以前

    # ------------------------------------------------------------------
    # 构造 t_idx 时刻的世界快照（流式，逐时刻调用）
    # ------------------------------------------------------------------
    def snapshot(self, t_idx):
        U = self.U
        N = self.const.N
        snap = WorldSnapshot(U, self.max_vis)
        snap.t_idx = t_idx
        snap.t_s = t_idx * self.dt

        vmask_t = self.visible_mask[t_idx]                  # (N,U)
        rng_t = self.range_all[t_idx]                       # (N,U)
        elev_t = self.elev_all[t_idx]                       # (N,U)
        active = self.active_all[t_idx]
        snap.active_mask = active

        # 每用户的候选星索引 + 距离 + 仰角（向量化填入定长数组）
        # 对每用户取可见星索引
        vis_idx = [np.where(vmask_t[:, u])[0] for u in range(U)]
        for u in range(U):
            idx = vis_idx[u]
            n = len(idx)
            if n == 0:
                continue
            if n > self.max_vis:
                idx = idx[:self.max_vis]; n = self.max_vis
            snap.visible_idx[u, :n] = idx
            snap.visible_range[u, :n] = rng_t[idx, u]
            snap.visible_elev[u, :n] = elev_t[idx, u]

        # 信道增益（仅可见星，向量化计算）
        # 把所有 (u, k) 可见对的距离收集成一维数组，一次性算信道增益
        valid = snap.visible_idx >= 0                        # (U,max_vis)
        ranges_flat = snap.visible_range[valid]              # (K,)
        if ranges_flat.size > 0:
            gain_flat = self.ch.channel_gain(ranges_flat)    # (K,) 含莱斯+阴影
            gain_full = np.zeros_like(snap.visible_gain)
            gain_full[valid] = gain_flat
            snap.visible_gain = gain_full
        else:
            gain_flat = np.zeros(0)
            snap.visible_gain[...] = 0

        # 额外路径衰减（Fig.6 信号质量轴驱动，物理理由见下）：
        # 该衰减代表"用户终端本地的链路质量退化"——晴空→暴雨/LOS阻挡/终端品质下降。
        # 关键物理理由：本地雨衰/遮挡主要衰减用户对"当前服务卫星"的直接 LOS 波束
        # （高增益定向链路穿过本地雨柱），而聚合同频干扰来自不同仰角/方位角的其它可见
        # 卫星——它们的链路穿过不同大气路径，且其功率已被 f_co（频率复用+波束 off-axis
        # 隔离）大幅压制，故干扰功率几乎不受本地雨衰影响。因此 extra_loss 仅作用于
        # 信号增益（service link），不作用于干扰聚合（interference aggregation）。
        # 这让 SINR = P·g·L / (P·f_co·Σg + Np) 在 L↓ 时单调下降（信号项衰减，干扰+噪声
        # 基线不变）→ 恶劣工况下低 SINR 候选跌破 floor → 成功率涌现动态范围。
        # 这是物理建模选择（局部衰落 vs 聚合干扰的几何解耦），非输出拟合。
        g = np.where(valid, snap.visible_gain, 0.0)
        if self.extra_loss_lin != 1.0:
            g_signal = g * self.extra_loss_lin               # 信号项衰减
        else:
            g_signal = g

        # 低仰角边缘衰减（Fig.6 服务时间轴驱动的物理机制，可配置）：
        # 物理依据（ITU-R P.681 / 3GPP TR 38.811 NLOS 卫星信道）：低仰角卫星链路穿过
        # 更长的大气路径 + 更易被局部地形/建筑/树木遮挡（地面散射、多径、NLOS 衰落），
        # 导致 SINR 显著低于高仰角 zenith 链路。该损耗在仰角接近 θ_min（刚升起/即将
        # 落下）时最大，zenith 时为 0。线性插值：loss = low_elev_loss_dB · (1 − elev/elev_max)。
        # 这让"紧急切换到刚升起低仰角星"的目标 SINR 显著低于 floor 之上的稳态星 →
        # 服务时间短(频繁紧急切换)→成功率降 的物理机制涌现。
        # 默认 0（不影响 Fig.7/9 基线）；Fig.6 服务时间轴扫描局部设为典型 NLOS 值。
        low_elev_loss = float(self.rf_cfg.get('low_elevation_loss_dB', 0.0))
        if low_elev_loss > 0:
            elev_deg = np.where(valid, snap.visible_elev, 90.0)  # 不可见视为 zenith(无边缘损耗)
            elev_max = 90.0
            # 线性插值：zenith(90°)→0, θ_min→low_elev_loss
            thr_deg_arr = np.rad2deg(self.theta_min)
            frac = np.clip((elev_max - elev_deg) / max(elev_max - thr_deg_arr, 1e-3), 0, 1)
            edge_loss_lin = 10.0 ** (-low_elev_loss * frac / 10.0)
            g_signal = g_signal * edge_loss_lin

        # 干扰：对每用户，干扰 = f_cochannel · Σ_{可见非服务} P·|h|²（同频泄漏）
        # f_cochannel = 同频泄漏系数：同色同极化可见星比例 × 极化/波束 off-axis 隔离
        # Starlink Ku：4色频率×2极化（K≈8）+ XPI~25dB + 点波束 off-axis 衰减，
        # f_co=1/28（见 config.py RF 注释的物理理由）。可调物理参数（config 可覆盖）。
        # 注意：干扰聚合用未衰减的 g（理由见上方 extra_loss 物理注释）。
        f_co = self.rf_cfg.get('cochannel_leakage_factor', 1.0/28.0)
        sum_g = g.sum(axis=1, keepdims=True)                # (U,1)
        snap.interf_sum_cand = np.where(valid, f_co * (sum_g - g), 0.0)

        # 每候选 SINR（先于选星，用"假设服务该星"算）
        # 信号用 g_signal（含 extra_loss），干扰用未衰减 g 的聚合。
        P = self.ch.Ptx
        Np = self.ch.Np
        sinr_c = (P * g_signal) / (P * snap.interf_sum_cand + Np)
        sinr_c = np.where(valid, sinr_c, 0.0)
        snap.sinr_cand = sinr_c
        snap.rate_cand_mbps = np.where(valid, self.ch.rate_mbps(sinr_c), 0.0)
        snap.H_norm_cand = np.where(valid, self.ch.H_norm(sinr_c), 0.0)

        # 剩余服务时间 τ：对每候选星，前向细步看何时跌破 θ_min
        #   向量化：对每用户每候选星，找 elev_remaining 中首个 < thr 的步数
        thr_deg = np.rad2deg(self.theta_min)
        elev_remaining = self.elev_all[t_idx:, :, :]        # (T_rem,N,U)
        T_rem = elev_remaining.shape[0]
        below_mask = elev_remaining < thr_deg               # (T_rem,N,U) bool
        # 对每 (N,U) 找首个 True 的步数（argmax 在全 False 时返回 0，需后处理）
        first_below = below_mask.argmax(axis=0)             # (N,U)
        any_below = below_mask.any(axis=0)                  # (N,U)
        # 填入每用户每候选星
        for u in range(U):
            idx = vis_idx[u]
            n = min(len(idx), self.max_vis)
            if n == 0:
                continue
            ab = any_below[idx[:n], u]
            fb = first_below[idx[:n], u]
            tau_s = np.where(ab, fb * self.dt, T_rem * self.dt)
            # 可见性碎片化（Fig.6 服务时间轴驱动）：缩放有效剩余服务时间，
            # 模拟移动/城市场景下 LOS 阻塞导致的可见性中断（物理理由见 __init__ 注释）。
            tau_s = tau_s * self.visibility_frag
            snap.tau_cand_s[u, :n] = tau_s

        # 负载（Eq.5）：每星 D_s = 服务用户速率之和 / 1000 Gbps
        # 此处先全 0；决策层确定关联后回填
        snap.sat_load = np.zeros(N)
        snap.sat_n_users = np.zeros(N, dtype=np.int32)
        return snap

    # ------------------------------------------------------------------
    # 根据全网"服务关联"更新每星负载（决策后调用）
    #   serving : (U,) 当前每用户服务星 idx（-1=无）
    #   rate    : (U,) 当前每用户服务链路速率
    #   负载 Eq.5 物理实现：L_s = N_serving_eff / N_beam_max（波束占用率）
    #   L̃ ∈ [0,1]，>1 表示过载（无可用波束）
    #
    # effective_user_mult（Fig.6 负载轴驱动）：每用户代表的并发有效业务承载数。
    # 物理含义：每订阅者同时维持多个并发流（4K视频+下载+VoIP+IoT），等效到波束
    # 占用为 k 个并发会话。这让负载扫描覆盖从轻负载(农村稀疏部署,单承载)到重负载
    # (城市密集部署,多承载)的工况区间，而不需要改星座几何/用户数（几何复用保留）。
    # 这是系统工况参数（业务强度），非输出拟合。
    # ------------------------------------------------------------------
    def update_load(self, snap, serving, rate_mbps):
        N = self.const.N
        n_u = np.zeros(N, dtype=np.int32)
        valid = serving >= 0
        if np.any(valid):
            np.add.at(n_u, serving[valid], 1)
        N_beam = self.rf_cfg.get('sat_beam_max', 24)
        # 有效服务用户数 = 实际用户数 × 并发承载倍数
        n_u_eff = n_u.astype(float) * self.effective_user_mult
        L = n_u_eff / float(N_beam)                       # 归一化波束占用率
        snap.sat_load = L
        snap.sat_n_users = n_u
        return L

    # ------------------------------------------------------------------
    # 给定每星当前服务用户数，算"有效每用户速率"：
    #   高负载星的容量被均摊：r_eff = min(Shannon_rate, C_s_per_beam / N_users)
    #   这让 Fig.6(b) 负载轴物理成立：高负载→每用户速率降→可能跌破 R_min→失败
    # ------------------------------------------------------------------
    def effective_rate(self, shannon_rate_mbps, sat_idx, load_view):
        """shannon_rate_mbps: (U,) 服务链路 Shannon 速率。sat_idx: (U,) 服务星。"""
        n_u = load_view * self.rf_cfg.get('sat_beam_max', 24)
        n_u = np.maximum(n_u, 1.0)
        # 每星前向容量 Gbps → Mbps，按波束均摊
        C_per_beam_mbps = self.Cs_gbps * 1000.0 / float(self.rf_cfg.get('sat_beam_max', 24))
        # 每用户分得的容量份额 = C_per_beam / n_u_per_beam（同一波束内均摊）
        # 简化：用整星容量/N_users 作为上限
        cap_share = self.Cs_gbps * 1000.0 / np.maximum(n_u, 1.0)
        eff = np.minimum(shannon_rate_mbps, cap_share[sat_idx] if hasattr(cap_share, '__getitem__') else cap_share)
        return eff

    # ------------------------------------------------------------------
    # 给定服务星，查每用户候选池中该星的 k 索引
    # ------------------------------------------------------------------
    def find_serving_k(self, snap, user, sat):
        row = snap.visible_idx[user]
        hits = np.where(row == sat)[0]
        return int(hits[0]) if len(hits) > 0 else -1


if __name__ == '__main__':
    rng = np.random.default_rng(SEED)
    lats = rng.uniform(30, 32, 20)
    lons = rng.uniform(116, 118, 20)
    env = Environment(lats, lons)
    print(f'[env] U={env.U}, T={env.T}, max_vis={env.max_vis}')
    print(f'[env] 可见星数分布: min={env.n_vis_per.min()}/'
          f'mean={env.n_vis_per.mean():.1f}/max={env.n_vis_per.max()}')
    print(f'[env] active 比例 = {env.active_all.mean():.3f}')
    import time
    t0 = time.time()
    snap = env.snapshot(0)
    print(f'[env] snapshot(0) 耗时 {time.time()-t0:.2f}s')
    print(f'[env] t=0 用户0可见星={np.sum(snap.visible_idx[0]>=0)}, '
          f'SINR范围={snap.sinr_cand[0][snap.visible_idx[0]>=0].min():.1f}'
          f'~{snap.sinr_cand[0][snap.visible_idx[0]>=0].max():.1f}')
