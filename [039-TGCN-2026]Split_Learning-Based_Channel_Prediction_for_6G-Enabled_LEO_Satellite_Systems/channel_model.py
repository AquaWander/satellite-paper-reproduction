# -*- coding: utf-8 -*-
"""
信道模型模块
实现LEO卫星信道模型: 路径损耗 + 多普勒频移 + TDL-D多径衰落
论文: Split Learning-Based Channel Prediction for 6G-Enabled LEO Satellite Systems
"""

import numpy as np
from config import (
    SATELLITE_ALTITUDE, EARTH_RADIUS, CARRIER_FREQ, SPEED_OF_LIGHT,
    ELEVATION_ANGLE, N_SUBCARRIERS, N_DMRS, N_TIME_STEPS, N_CH,
    TDL_D_DELAYS, TDL_D_POWERS_DB, RICIAN_K_DB, SATELLITE_VELOCITY,
    N_CLUSTERS, CLUSTER_RADIUS, USERS_PER_CLUSTER, N_RF, N_ANTENNA,
    BANDWIDTH, NOISE_FIGURE_DB, THERMAL_NOISE_DBM, P_TRANSMIT_DBM,
    RANDOM_SEED, TDD_MODES,
)


def compute_distance(altitude=SATELLITE_ALTITUDE, elevation_deg=ELEVATION_ANGLE):
    """
    计算卫星到用户的斜距
    基于几何关系: d = sqrt((Re + h)^2 - (Re*cos(eps))^2) - Re*sin(eps)
    """
    Re = EARTH_RADIUS
    h = altitude
    eps = np.radians(elevation_deg)
    # 斜距公式 (简化，90度仰角时 d ≈ h)
    d = np.sqrt((Re + h)**2 - (Re * np.cos(eps))**2) - Re * np.sin(eps)
    return d


def compute_path_loss_db(distance, fc=CARRIER_FREQ):
    """
    自由空间路径损耗 L_FS = (4*pi*d*f_c/c)^2
    返回dB值
    """
    L_linear = (4 * np.pi * distance * fc / SPEED_OF_LIGHT) ** 2
    L_dB = 10 * np.log10(L_linear)
    return L_dB


def compute_doppler_shift(fc=CARRIER_FREQ, elevation_deg=ELEVATION_ANGLE):
    """
    计算最大多普勒频移 f_D = v * f_c / c * cos(elevation)
    LEO卫星高速运动导致显著多普勒效应
    """
    v = SATELLITE_VELOCITY
    f_D = v * fc / SPEED_OF_LIGHT * np.cos(np.radians(elevation_deg))
    return f_D


def compute_noise_power_dbm(bandwidth=BANDWIDTH, noise_figure_db=NOISE_FIGURE_DB,
                            thermal_noise_dbm_hz=THERMAL_NOISE_DBM):
    """
    计算接收机噪声功率 (dBm)
    N = k*T*B + NF
    """
    noise_psd_dbm_hz = thermal_noise_dbm_hz  # -174 dBm/Hz
    noise_power_dbm = noise_psd_dbm_hz + 10 * np.log10(bandwidth) + noise_figure_db
    return noise_power_dbm


def generate_tdl_d_channel(n_samples, n_sc=N_SUBCARRIERS, n_dmrs=N_DMRS,
                           snr_db=10, seed=None):
    """
    生成简化的3GPP TDL-D信道模型

    TDL-D特点:
    - 多抽头延迟线模型
    - 包含LOS径(Rician)和NLOS径(Rayleigh)
    - 支持多普勒扩展

    参数:
        n_samples: 样本数量
        n_sc: 子载波数
        n_dmrs: DMRS符号数
        snr_db: 信噪比(dB)
        seed: 随机种子

    返回:
        channel: 复数信道矩阵 (n_samples, n_dmrs, n_sc)
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState(RANDOM_SEED)

    n_taps = len(TDL_D_DELAYS)
    powers_linear = 10 ** (TDL_D_POWERS_DB / 10)
    # 归一化功率
    powers_linear = powers_linear / np.sum(powers_linear)

    # K因子 (Rician)
    K_linear = 10 ** (RICIAN_K_DB / 10)

    # 多普勒频移
    f_D = compute_doppler_shift()
    # 归一化多普勒 (相对于OFDM符号持续时间)
    T_symbol = 1 / (BANDWIDTH / N_SUBCARRIERS)  # OFDM符号周期(简化)
    f_D_norm = f_D * T_symbol * 1e3  # 归一化

    channels = np.zeros((n_samples, n_dmrs, n_sc), dtype=complex)

    for i in range(n_samples):
        for p in range(n_dmrs):
            channel_freq = np.zeros(n_sc, dtype=complex)

            for t in range(n_taps):
                # 生成时域信道冲激响应
                if t == 0:
                    # 第一径: Rician衰落 (有LOS分量)
                    los = np.sqrt(K_linear / (K_linear + 1))
                    nlos_real = rng.randn() * np.sqrt(1 / (2 * (K_linear + 1)))
                    nlos_imag = rng.randn() * np.sqrt(1 / (2 * (K_linear + 1)))
                    tap = los + (nlos_real + 1j * nlos_imag)
                else:
                    # 其他径: Rayleigh衰落
                    tap = (rng.randn() + 1j * rng.randn()) / np.sqrt(2)

                # 应用功率权重
                tap *= np.sqrt(powers_linear[t])

                # 应用多普勒频移 (随时间变化)
                time_idx = i * n_dmrs + p
                doppler_phase = np.exp(1j * 2 * np.pi * f_D_norm * time_idx * 0.001)
                tap *= doppler_phase

                # 转换到频域 (简化: 各子载波上应用相移)
                phase_shift = np.exp(-1j * 2 * np.pi * TDL_D_DELAYS[t] *
                                     np.arange(n_sc) / (1 / BANDWIDTH))
                channel_freq += tap * phase_shift

            channels[i, p, :] = channel_freq

    # 应用路径损耗
    distance = compute_distance()
    PL_dB = compute_path_loss_db(distance)

    # 加入AWGN噪声
    snr_linear = 10 ** (snr_db / 10)
    signal_power = np.mean(np.abs(channels) ** 2)
    noise_power = signal_power / snr_linear

    noise = (np.sqrt(noise_power / 2) *
             (rng.randn(n_samples, n_dmrs, n_sc) + 1j * rng.randn(n_samples, n_dmrs, n_sc)))

    # 信道 + 噪声 (模拟接收端观测到的带噪信道)
    noisy_channels = channels + noise

    return channels, noisy_channels


def generate_dataset(n_samples, snr_db, n_time_steps=N_TIME_STEPS,
                     tdd_mode='DSUUU', seed=None):
    """
    生成训练/测试数据集

    利用上行信道数据预测下行信道:
    - 输入: N_U个连续上行时隙的DMRS信道估计
    - 输出: 下行时隙的信道估计

    参数:
        n_samples: 样本数量
        snr_db: 信噪比(dB)
        n_time_steps: 输入时间步数 N_U
        tdd_mode: TDD模式 ('DSUUU' or 'DSUUD')
        seed: 随机种子

    返回:
        X: 输入数据 (n_samples, N_U, N_p, N_SC, N_CH)
        Y: 目标数据 (n_samples, N_p*N_SC*N_CH)
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState(RANDOM_SEED)

    mode = TDD_MODES[tdd_mode]
    # DSUUU有3个上行时隙, DSUUD只有2个
    n_uplink = mode['uplink']

    # TDD模式下上下行信道差异因子
    # 上行时隙越多，上下行时间差越小，互易性越好
    if tdd_mode == 'DSUUU':
        reciprocity_factor = 0.05   # DSUUU: 互易性误差较小
    else:
        reciprocity_factor = 0.15   # DSUUD: 互易性误差较大

    # 生成上行信道序列
    total_slots = n_samples * (n_time_steps + 1)  # N_U个上行 + 1个下行
    channels_clean, channels_noisy = generate_tdl_d_channel(
        total_slots, seed=seed
    )

    # 归一化信道功率
    ref_power = np.percentile(np.abs(channels_clean), 95)
    channels_clean = channels_clean / ref_power
    channels_noisy = channels_noisy / ref_power

    # 应用SNR
    snr_linear = 10 ** (snr_db / 10)
    signal_power = np.mean(np.abs(channels_noisy) ** 2)
    noise_power = signal_power / snr_linear

    noise = np.sqrt(noise_power / 2) * (
        rng.randn(*channels_noisy.shape) + 1j * rng.randn(*channels_noisy.shape)
    )
    channels_noisy_with_snr = channels_noisy + noise

    X_list = []
    Y_list = []

    for i in range(n_samples):
        # 输入: N_U个连续上行时隙
        start_idx = i * (n_time_steps + 1)

        # 上行信道 (输入)
        ul_channels = channels_noisy_with_snr[start_idx:start_idx + n_time_steps]
        # (N_U, N_p, N_SC) -> (N_U, N_p, N_SC, N_CH) 分离实部虚部
        ul_real = np.real(ul_channels)
        ul_imag = np.imag(ul_channels)
        ul_input = np.stack([ul_real, ul_imag], axis=-1)  # (N_U, N_p, N_SC, 2)

        # 下行信道 (目标)
        dl_idx = start_idx + n_time_steps
        dl_channel = channels_clean[dl_idx]  # (N_p, N_SC)

        # 添加TDD互易性误差 (上下行不完全互易)
        tdd_error = reciprocity_factor * (
            rng.randn(*dl_channel.shape) + 1j * rng.randn(*dl_channel.shape)
        )
        dl_channel = dl_channel + tdd_error

        # 展平为 (N_p * N_SC * 2,)
        dl_real = np.real(dl_channel).flatten()
        dl_imag = np.imag(dl_channel).flatten()
        dl_target = np.concatenate([dl_real, dl_imag])

        X_list.append(ul_input)
        Y_list.append(dl_target)

    X = np.array(X_list, dtype=np.float32)   # (n_samples, N_U, N_p, N_SC, 2)
    Y = np.array(Y_list, dtype=np.float32)   # (n_samples, N_p*N_SC*2)

    return X, Y


def compute_nmse_db(predicted, target):
    """
    计算NMSE (dB)
    NMSE = ||H_pred - H_true||^2 / ||H_true||^2

    对应论文 Eq.13: L = (1/N_B) * sum ||H_tilde_OUT - H_OUT||^2 / ||H_tilde_OUT||^2
    """
    # 确保2D
    pred = predicted.reshape(predicted.shape[0], -1)
    targ = target.reshape(target.shape[0], -1)

    numerator = np.sum((pred - targ) ** 2, axis=1)
    denominator = np.sum(targ ** 2, axis=1) + 1e-10

    nmse = np.mean(numerator / denominator)
    nmse_db = 10 * np.log10(nmse + 1e-10)

    return nmse_db


def compute_effective_snr(snr_db, nmse_db):
    """
    计算有效SNR (Gamma_eff)
    Gamma_eff = P_m * (1 - NMSE) / (P_m * NMSE + sigma^2_n)

    简化形式: Gamma_eff = (1 - NMSE_lin) / NMSE_lin * SNR_lin
    其中 SNR_lin = P_m / sigma^2_n
    """
    snr_lin = 10 ** (snr_db / 10)
    nmse_lin = 10 ** (nmse_db / 10)

    # 避免 nmse_lin >= 1 的情况
    nmse_lin = np.clip(nmse_lin, 1e-10, 0.999)

    gamma_eff = snr_lin * (1 - nmse_lin) / (snr_lin * nmse_lin + 1)
    return gamma_eff


def compute_data_rate(snr_db, nmse_db):
    """
    计算数据速率 eta = log2(1 + Gamma_eff)
    单位: bps/Hz
    """
    gamma_eff = compute_effective_snr(snr_db, nmse_db)
    gamma_eff = np.maximum(gamma_eff, 0)  # 确保非负
    rate = np.log2(1 + gamma_eff)
    return rate


if __name__ == '__main__':
    # 测试信道模型
    print("=" * 60)
    print("信道模型测试")
    print("=" * 60)

    distance = compute_distance()
    print(f"卫星-用户距离: {distance/1e3:.1f} km")

    pl = compute_path_loss_db(distance)
    print(f"自由空间路径损耗: {pl:.1f} dB")

    fd = compute_doppler_shift()
    print(f"多普勒频移: {fd/1e3:.2f} kHz")

    noise = compute_noise_power_dbm()
    print(f"接收机噪声功率: {noise:.1f} dBm")

    # 测试数据生成
    X, Y = generate_dataset(100, snr_db=10, tdd_mode='DSUUU', seed=42)
    print(f"\n数据集形状:")
    print(f"  输入 X: {X.shape}")  # (100, 5, 2, 300, 2)
    print(f"  目标 Y: {Y.shape}")  # (100, 1200)

    # 测试不同SNR下的NMSE范围
    print(f"\n不同SNR下的信道估计NMSE (理论):")
    for snr in [-10, 0, 10, 20]:
        ch_clean, ch_noisy = generate_tdl_d_channel(100, snr_db=snr, seed=42)
        # 将复数信道转为实数 (实部+虚部拼接)
        ch_clean_real = np.concatenate([np.real(ch_clean).flatten(), np.imag(ch_clean).flatten()])
        ch_noisy_real = np.concatenate([np.real(ch_noisy).flatten(), np.imag(ch_noisy).flatten()])
        # 扩展为2D
        ch_clean_2d = ch_clean_real.reshape(1, -1)
        ch_noisy_2d = ch_noisy_real.reshape(1, -1)
        nmse = compute_nmse_db(ch_noisy_2d, ch_clean_2d)
        print(f"  SNR = {snr:3d} dB -> NMSE = {nmse:.2f} dB")
