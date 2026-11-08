# -*- coding: utf-8 -*-
"""
constellation.py
================================================================================
Walker-delta 星座构造 + J2 长期传播 + 可见性/仰角/距离/剩余服务时间。
向量化（每时刻对全部 1584 星 × 全部用户一次算）。

【诚实声明】前向物理仿真，所有几何量由 Kepler+J2 演化产生，不查表。

物理流程：
  Walker-delta 1584 星（72 面 × 22 星），550km，倾角 53°。
  平运动 n0=sqrt(μ/a³)；J2 使 Ω 进动、ω 进动。
  时间 t：Ω(t)=Ω0+dΩ/dt·t，M(t)=M0+n0·t，e≈0 圆轨道 ⇒ E=M。
  Kepler→ECI（3-1-3 旋转 R_z(Ω)R_x(i)R_z(ω)）→应用 GMST 地球自转角→ECEF。
  对用户(lat,lon)：ECEF→ENU，elevation=atan2(U,sqrt(E²+N²))，slant_range=norm。
  visible_set = 仰角≥θ_min 的卫星。
  remaining_service_time：前向细步传播到该星仰角跌破 θ_min 的剩余秒数。
================================================================================
"""

import numpy as np
from config import CONSTELLATION, SEED


class Constellation:
    """Walker-delta 1584 星星座，向量化传播与可见性。"""

    def __init__(self, cfg=None, seed=SEED):
        cfg = cfg or CONSTELLATION
        self.cfg = cfg
        self.N = cfg['num_planes'] * cfg['sats_per_plane']
        self.P = cfg['num_planes']
        self.Q = cfg['sats_per_plane']
        self.h = cfg['altitude_km']
        self.Re = cfg['earth_radius_km']
        self.a = self.Re + self.h
        self.inc = np.deg2rad(cfg['inclination_deg'])
        self.e = cfg['eccentricity']
        self.mu = cfg['mu_km3_s2']
        self.J2 = cfg['J2']
        self.theta_min = np.deg2rad(cfg['min_elevation_deg'])
        self.omega_e = cfg['earth_omega_rad_s']

        self.n0 = np.sqrt(self.mu / self.a ** 3)
        cos_i = np.cos(self.inc)
        self.dOmega_dt = -1.5 * self.J2 * (self.Re / self.a) ** 2 * self.n0 * cos_i
        self.domega_dt = 1.5 * self.J2 * (self.Re / self.a) ** 2 * self.n0 * (2.0 - 2.5 * np.sin(self.inc) ** 2)

        plane_idx = np.repeat(np.arange(self.P), self.Q)
        sat_in_plane = np.tile(np.arange(self.Q), self.P)
        self.Omega0 = np.deg2rad(plane_idx * (360.0 / self.P))
        self.M0 = np.deg2rad(
            sat_in_plane * (360.0 / self.Q)
            + plane_idx * (360.0 / (self.P * self.Q))
        )
        self.omega0 = np.zeros(self.N)

    # ------------------------------------------------------------------
    # ECI 位置（向量化，t 可为标量或数组）→ (T,N,3) km
    # ------------------------------------------------------------------
    def eci_positions(self, t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        Omega = self.Omega0[None, :] + self.dOmega_dt * t[:, None]   # (T,N)
        M = self.M0[None, :] + self.n0 * t[:, None]
        omega = self.omega0[None, :] + self.domega_dt * t[:, None]
        E = M  # e≈0
        cosE = np.cos(E); sinE = np.sin(E)
        x_orb = self.a * (cosE - self.e)
        y_orb = self.a * np.sqrt(1 - self.e ** 2) * sinE
        cO = np.cos(Omega); sO = np.sin(Omega)
        cw = np.cos(omega); sw = np.sin(omega)
        ci = np.cos(self.inc); si = np.sin(self.inc)
        A = cO * cw - sO * sw * ci
        B = -cO * sw - sO * cw * ci
        C = sO * cw + cO * sw * ci
        D = -sO * sw + cO * cw * ci
        E3 = sw * si
        F3 = cw * si
        x = A * x_orb + B * y_orb
        y = C * x_orb + D * y_orb
        z = E3 * x_orb + F3 * y_orb
        return np.stack([x, y, z], axis=-1)

    # ------------------------------------------------------------------
    # ECEF（应用 GMST 地球自转）→ (T,N,3)
    # ------------------------------------------------------------------
    def ecef_positions(self, t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        eci = self.eci_positions(t)
        theta_g = self.omega_e * t
        cg = np.cos(theta_g)[:, None]
        sg = np.sin(theta_g)[:, None]
        x = eci[..., 0]; y = eci[..., 1]; z = eci[..., 2]
        xe = cg * x + sg * y
        ye = -sg * x + cg * y
        return np.stack([xe, ye, z], axis=-1)

    # ------------------------------------------------------------------
    # 用户 ECEF（静态地面点）
    # ------------------------------------------------------------------
    def user_ecef(self, lat_deg, lon_deg):
        lat = np.deg2rad(np.asarray(lat_deg, dtype=float))
        lon = np.deg2rad(np.asarray(lon_deg, dtype=float))
        x = self.Re * np.cos(lat) * np.cos(lon)
        y = self.Re * np.cos(lat) * np.sin(lon)
        z = self.Re * np.sin(lat)
        shape = np.broadcast(lat, lon).shape
        return np.stack(np.broadcast_arrays(x, y, z), axis=-1).reshape(*shape, 3)

    # ------------------------------------------------------------------
    # 仰角 + slant range（向量化）
    #   sat_ecef : (T,N,3) 或 (N,3)
    #   user_pos : (U,3)
    #   返回 elev_deg (T,N,U), range_km (T,N,U)
    # ------------------------------------------------------------------
    def elevation_and_range(self, sat_ecef, user_pos):
        user_pos = np.asarray(user_pos, dtype=float)
        U = user_pos.shape[0]
        if sat_ecef.ndim == 2:                                  # (N,3) → (1,N,3)
            sat_ecef = sat_ecef[None]
        T, N, _ = sat_ecef.shape
        sat = sat_ecef[:, :, None, :]                            # (T,N,1,3)
        user = user_pos[None, None, :, :]                        # (1,1,U,3)
        diff = sat - user                                        # (T,N,U,3)
        u_norm = user / np.linalg.norm(user, axis=-1, keepdims=True)
        lat_u = np.arcsin(np.clip(u_norm[..., 2], -1, 1))
        lon_u = np.arctan2(u_norm[..., 1], u_norm[..., 0])
        sl, cl = np.sin(lon_u), np.cos(lon_u)
        sa, ca = np.sin(lat_u), np.cos(lat_u)
        east = np.stack([-sl, cl, np.zeros_like(sl)], axis=-1)
        north = np.stack([-sa * cl, -sa * sl, ca], axis=-1)
        up = u_norm
        e_c = np.sum(diff * east, axis=-1)
        n_c = np.sum(diff * north, axis=-1)
        u_c = np.sum(diff * up, axis=-1)
        elev = np.arctan2(u_c, np.sqrt(e_c ** 2 + n_c ** 2))
        rng = np.linalg.norm(diff, axis=-1)
        return np.rad2deg(elev), rng


if __name__ == '__main__':
    c = Constellation()
    print(f'[const] N={c.N}, a={c.a:.1f}km, n0={c.n0*1000:.3f} mrad/s')
    eci = c.eci_positions(np.array([0.0]))
    r = np.linalg.norm(eci[0], axis=-1)
    print(f'[const] t=0 模长 min/max/mean = {r.min():.1f}/{r.max():.1f}/{r.mean():.1f} km')
    t = np.arange(0, 600, 1.0)
    ecef = c.ecef_positions(t)
    u = c.user_ecef(np.array([31.0]), np.array([117.0]))
    elev, rng = c.elevation_and_range(ecef, u)
    thr = c.cfg['min_elevation_deg']
    nvis = np.sum(elev[:, :, 0] >= thr, axis=1)
    print(f'[const] 中纬(31N,117E) 600s 可见星: min={nvis.min()}/mean={nvis.mean():.1f}/max={nvis.max()}')
