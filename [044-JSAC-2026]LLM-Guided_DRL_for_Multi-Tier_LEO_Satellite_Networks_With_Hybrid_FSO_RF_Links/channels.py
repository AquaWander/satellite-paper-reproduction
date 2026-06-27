"""
channels.py
=========
严格按论文公式实现 FSO (Eq.4-7) 与 RF/OFDM (Eq.8-10) 信道模型。

==== FSO (卫星↔HAP) ====
Eq.4:  h_SH(t) = h_l · h_a(t)
Eq.5:  h_l = (1/2)(G_T + G_R − A_FS − A_ATM − L_loss − M_S)   [dB 代数和, 1/2 因子]
       h_a(t) ~ Gamma-Gamma(α, β)
Eq.6:  γ_H(t) = P_FSO · η_OE^2 · h_EGC^2(t) / (N_A · N_q)
       h_EGC(t) = Σ_{q=1}^{N_A} h_SH_q(t)   [N_A 孔径等增益合并, 标量和]
Eq.7:  R_FSO(t) = B_FSO · log2(1 + γ_H(t))

==== RF/OFDM (HAP↔地面簇 i) ====
Eq.8:  h_HC,i(t) = C_HC,i(t) · g_HC,i(t)
Eq.9:  C_HC,i(t) = G_HC + R_i + (1/2)(20·lg(λ_RF) − 10·η·lg(d_HC,i(t)) − 20·lg(4π))
       [1/2 系数, η 路径损耗指数, lg 以 10 为底]
       g_HC,i(t) ~ Nakagami-m
Eq.10: R_RF_i(t) = n_i(t)·(B_RF/N_S)·log2(1 + P_RF·|h_HC,i|^2 / ((B_RF/N_S)·σ_C^2))
"""
from __future__ import annotations
import numpy as np
from scipy.special import gamma as gamma_func

import config as C


def lg(x):
    """以 10 为底的对数 (论文公式中 lg)."""
    return np.log10(x)


# ============================================================================
# FSO 信道 (Eq.4-7)
# ============================================================================
def fso_path_loss_free_space(d_sh):
    """自由空间损耗 A_FS (dB): A_FS = 20·lg(4π·d/λ)."""
    return 20.0 * lg(4.0 * np.pi * d_sh / C.FSO_WAVELENGTH)


def fso_static_gain_db(d_sh, a_atm_db):
    """
    Eq.5: h_l (dB) = (1/2)(G_T + G_R − A_FS − A_ATM − L_loss − M_S)
    返回 dB 值(标量)；h_l(线性)后续用 10^(h_l_dB/10) 但保持 dB 处理更稳。
    """
    a_fs = fso_path_loss_free_space(d_sh)
    h_l_db = 0.5 * (C.G_T_FSO_DB + C.G_R_FSO_DB - a_fs - a_atm_db
                    - C.L_LOSS - C.M_S)
    return h_l_db


def gamma_gamma_sample(alpha, beta, rng, size=None):
    """
    Gamma-Gamma 分布采样 (Eq.4 中 h_a ~ ΓΓ(α,β)).
    用双 Gamma 乘积构造：I = X·Y, X~Γ(α,1), Y~Γ(β,1), 再归一化使 E[I]=1.
    """
    x = rng.gamma(shape=alpha, scale=1.0 / alpha, size=size)
    y = rng.gamma(shape=beta, scale=1.0 / beta, size=size)
    return x * y  # 均值约 1


def fso_channel_gain_linear(d_sh, a_atm_db, rng, n_aperture=None):
    """
    返回 N_A 个孔径各自的 h_SH (Eq.4) 线性值 (1D array, len=n_aperture).
    """
    if n_aperture is None:
        n_aperture = C.N_APERTURE
    h_l_db = fso_static_gain_db(d_sh, a_atm_db)
    h_l_lin = 10.0 ** (h_l_db / 10.0)
    h_a = gamma_gamma_sample(C.GAMMA_ALPHA, C.GAMMA_BETA, rng, size=n_aperture)
    return h_l_lin * h_a  # 每孔径 h_SH (Eq.4)


def fso_snr(h_sh_array, p_fso=None, eta_oe=None, n_q=None):
    """
    Eq.6: γ_H = P_FSO · η_OE^2 · h_EGC^2 / (N_A · N_q)
          h_EGC = Σ_q h_SH_q   (标量和, 等增益合并)
    """
    p_fso = C.P_FSO if p_fso is None else p_fso
    eta_oe = C.ETA_OE if eta_oe is None else eta_oe
    n_q = C.N_Q if n_q is None else n_q
    n_a = len(h_sh_array)
    h_egc = np.sum(h_sh_array)   # 标量和 (Eq.6)
    return p_fso * (eta_oe ** 2) * (h_egc ** 2) / (n_a * n_q)


def fso_rate(gamma_h, b_fso=None):
    """Eq.7: R_FSO = B_FSO · log2(1 + γ_H)."""
    b_fso = C.B_FSO if b_fso is None else b_fso
    return b_fso * np.log2(1.0 + gamma_h)


# ============================================================================
# RF/OFDM 信道 (Eq.8-10)
# ============================================================================
def rf_large_scale_db(d_hc_i, r_i_db, lam_rf=None, eta=None, g_hc=None):
    """
    Eq.9: C_HC,i(dB) = G_HC + R_i + (1/2)(20·lg(λ_RF) − 10·η·lg(d_HC,i) − 20·lg(4π))
    """
    lam_rf = C.RF_WAVELENGTH if lam_rf is None else lam_rf
    eta = C.ETA_PATH if eta is None else eta
    g_hc = C.G_HC_DB if g_hc is None else g_hc
    c_db = g_hc + r_i_db + 0.5 * (20.0 * lg(lam_rf) - 10.0 * eta * lg(d_hc_i)
                                  - 20.0 * lg(4.0 * np.pi))
    return c_db


def nakagami_sample(m, omega, rng, size=None):
    """
    Nakagami-m 采样 g_HC,i(t) (Eq.8).
    由 Nakagami ↔ Gamma 关系: |g|^2 ~ Gamma(shape=m, scale=omega/m).
    """
    g2 = rng.gamma(shape=m, scale=omega / m, size=size)
    return np.sqrt(g2)


def rf_channel_gain(d_hc_i, r_i_db, rng, m=None, omega=1.0):
    """
    Eq.8: h_HC,i = C_HC,i · g_HC,i  (复信道增益模的线性值)
    """
    m = C.NAKAGAMI_M if m is None else m
    c_db = rf_large_scale_db(d_hc_i, r_i_db)
    c_lin = 10.0 ** (c_db / 10.0)
    g = nakagami_sample(m, omega, rng)
    return c_lin * g


def rf_rate(n_i, h_hc_i, b_rf=None, n_sub=None, p_rf=None, sigma_c2=None):
    """
    Eq.10: R_RF_i = n_i·(B_RF/N_S)·log2(1 + P_RF·|h_HC,i|^2 / ((B_RF/N_S)·σ_C^2))
    """
    b_rf = C.B_RF if b_rf is None else b_rf
    n_sub = C.N_SUB if n_sub is None else n_sub
    p_rf = C.P_RF if p_rf is None else p_rf
    sigma_c2 = C.SIGMA_C2 if sigma_c2 is None else sigma_c2
    bw_sub = b_rf / n_sub
    snr_sub = p_rf * (h_hc_i ** 2) / (bw_sub * sigma_c2)
    return n_i * bw_sub * np.log2(1.0 + snr_sub)


# ============================================================================
# 物理层速率 → 归一化奖励  (物理校准, 非硬编码)
# ============================================================================
def rate_to_reward_gbps(rate_bps):
    """bps → Gbps."""
    return rate_bps / 1e9


def normalize_reward(rate_gbps):
    """
    物理归一化: r_norm = rate_gbps / R_REF, 使最好算法 episodic reward ~ [28,34].
    R_REF=0.5 (Gbps 量级, 由 config.R_REF 控制).
    """
    return rate_gbps / C.R_REF
