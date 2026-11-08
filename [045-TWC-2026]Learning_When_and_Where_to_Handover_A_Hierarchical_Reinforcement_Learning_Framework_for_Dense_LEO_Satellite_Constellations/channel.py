# -*- coding: utf-8 -*-
"""
channel.py
================================================================================
星地链路信道模型（论文 Eq.1-Eq.4 的物理实现）。
全部由距离/频率/功率/增益前向计算，不查表。

  Eq.1  FSPL：H_FSPL = (4π·d·fc/c)²
  Eq.2  信道增益 |h|² = Gt·Gr·(c/(4π·d·fc))² · F_shadow · F_rician
        莱斯 K=6dB：功率 = K/(K+1)·LOS 确定性 + 1/(K+1)·Rayleigh(randn)
        对数正态阴影 = 10^(σ·randn/10)
  Eq.3  星间干扰 I = Σ_{其他可见星} P·|h|²（同频可见星功率泄漏）
  Eq.4  SINR = P·|h_serving|²/(I + N0·B)；R = B·log2(1+SINR)

归一化 H̃ = clip(SINR_lin / SINR_ref_lin, 0, 1)。
================================================================================
"""

import numpy as np
from config import RF, DECISION, SEED


C_KM_S = 2.99792458e5   # 光速 km/s


class Channel:
    def __init__(self, cfg_rf=None, cfg_dec=None, seed=SEED):
        self.rf = cfg_rf or RF
        self.dec = cfg_dec or DECISION
        self.rng = np.random.default_rng(seed)
        # 基本量
        self.fc = self.rf['carrier_freq_GHz'] * 1e9          # Hz
        self.B = self.rf['bandwidth_Hz']                       # Hz
        self.Ptx = self.rf['tx_power_W']                       # W
        self.Gt = 10 ** (self.rf['tx_gain_dBi'] / 10.0)
        self.Gr = 10 ** (self.rf['rx_gain_dBi'] / 10.0)
        # 噪声功率 N0·B·F（W）：N0 dBm/Hz → W/Hz → ×B → ×接收机噪声系数 F
        # F（线性）= 10^(NF_dB/10)，NF_dB 默认 2.0（终端 LNB 典型）
        n0_dbm_hz = self.rf['noise_density_dbm_per_Hz']
        nf_db = self.rf.get('noise_figure_dB', 0.0)
        n0_W_Hz = 10 ** ((n0_dbm_hz - 30) / 10.0)
        F_lin = 10 ** (nf_db / 10.0)
        self.Np = n0_W_Hz * self.B * F_lin                      # W（含接收机噪声系数）
        # 莱斯 K 线性
        self.K_lin = 10 ** (self.rf['rice_K_dB'] / 10.0)
        self.sigma_shadow = self.rf['shadow_sigma_dB']
        # 归一化参考 SINR（线性）
        self.sinr_ref_lin = 10 ** (self.dec['sinr_ref_for_Hnorm_dB'] / 10.0)

    # ------------------------------------------------------------------
    # 信道增益 |h|²（线性，含 FSPL+莱斯+阴影）
    #   d : 距离 km，任意形状
    #   返回 与 d 同形状
    # ------------------------------------------------------------------
    def channel_gain(self, d_km):
        """
        信道增益（线性，功率比）。Gt·Gr·(c/(4π·d·fc))² · F_rician · F_shadow。
        d 为 km 标量或数组。
        """
        d = np.asarray(d_km, dtype=float)
        # FSPL 线性 = (c/(4π·d·fc))²，d 转 m
        lam = 2.99792458e8 / self.fc                          # 波长 m
        fspl_lin = (lam / (4 * np.pi * d * 1000.0)) ** 2      # d:km→m
        gain = self.Gt * self.Gr * fspl_lin
        # 莱斯：总快衰 = K/(K+1)·1（LOS确定性） + 1/(K+1)·|h_rayleigh|²
        # |h_rayleigh|² = |randn + j randn|² ~ Exp(1)
        shape = d.shape
        rayleigh_pwr = (self.rng.standard_normal(shape) ** 2
                        + self.rng.standard_normal(shape) ** 2)
        fast_fade = self.K_lin / (self.K_lin + 1.0) \
            + 1.0 / (self.K_lin + 1.0) * rayleigh_pwr
        # 对数正态阴影（dB 域高斯）
        shadow = 10 ** (self.sigma_shadow * self.rng.standard_normal(shape) / 10.0)
        return gain * fast_fade * shadow

    # ------------------------------------------------------------------
    # 每链路 SINR（向量化）
    #   gain_serving : (T,U) 服务星信道增益
    #   gain_interf  : (T,U,N_other) 其他可见星信道增益（已含 Ptx？否，这里只传增益）
    #   返回 (T,U) SINR 线性
    # ------------------------------------------------------------------
    def sinr(self, gain_serving_lin, gain_interf_sum_lin):
        """
        gain_serving_lin     : (T,U) 服务星 |h|²·（含 Gt Gr，已乘）
        gain_interf_sum_lin  : (T,U) 其它可见星 |h|² 之和（同频泄漏）
        SINR = P·g_s / (P·g_I + Np)
        """
        num = self.Ptx * gain_serving_lin
        interf = self.Ptx * gain_interf_sum_lin
        return num / (interf + self.Np)

    # ------------------------------------------------------------------
    # Shannon 速率（Mbps）
    # ------------------------------------------------------------------
    def rate_mbps(self, sinr_lin):
        """R = B·log2(1+SINR)（bps→Mbps）。"""
        return self.B * np.log2(1.0 + np.clip(sinr_lin, 0, None)) / 1e6

    # ------------------------------------------------------------------
    # 归一化 H̃
    # ------------------------------------------------------------------
    def H_norm(self, sinr_lin):
        return np.clip(sinr_lin / self.sinr_ref_lin, 0.0, 1.0)


if __name__ == '__main__':
    ch = Channel()
    # 自检：单链路 d=1000km
    g = ch.channel_gain(np.array([1000.0, 1200.0]))
    sinr = ch.sinr(g, np.zeros_like(g))
    print(f'[chan] d=1000km gain={g[0]:.2e}, SINR={10*np.log10(sinr[0]):.1f} dB,'
          f' R={ch.rate_mbps(sinr[0]):.1f} Mbps, H̃={ch.H_norm(sinr[0]):.3f}')
    sinr2 = ch.sinr(np.array([g[0]]), np.array([0.01 * g[0]]))  # 1% 干扰
    print(f'[chan] 1% 同频干扰 SINR={10*np.log10(sinr2[0]):.1f} dB')
    print(f'[chan] Np = {10*np.log10(ch.Np*1000):.1f} dBm over B={ch.B/1e6:.0f}MHz')
