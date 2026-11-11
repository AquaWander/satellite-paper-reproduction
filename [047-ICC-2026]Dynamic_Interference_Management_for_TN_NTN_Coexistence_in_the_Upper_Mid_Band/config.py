"""论文参数与可复查的复现假设；不以论文结果作为算法输入。"""
from dataclasses import dataclass


@dataclass(frozen=True)
class SimConfig:
    # Table I / Section V：原文提供的参数。
    seed: int = 20260823
    area_side_m: float = 20_000.0
    n_gnb: int = 106
    frequency_hz: float = 12e9
    bandwidth_hz: float = 200e6
    satellite_power_dbm: float = 48.0
    satellite_gain_dbi: float = 38.3
    t1_gain_dbi: float = 33.0
    t2_gain_dbi: float = 17.0
    gnb_max_power_dbm: float = 33.0
    ue_max_power_dbm: float = 23.0
    density_values: tuple = (6e-8, 1e-7, 3e-7)
    mix_values: tuple = (1.0, 0.5, 0.0)
    tn_ue_density_m2: float = 1.3e-5
    ue_per_cell_min: int = 12
    ue_per_cell_max: int = 20
    satellite_altitude_m: float = 600_000.0
    earth_radius_m: float = 6_371_000.0
    n_snapshots: int = 160
    elevation_start_deg: float = 10.0
    elevation_step_deg: float = 1.0
    # 以下数值未由ICC论文完全提供，属于显式复现假设。
    bs_coordinates_csv: str = ''
    gnb_height_m: float = 25.0
    terminal_height_m: float = 1.5
    gnb_peak_gain_dbi: float = 15.0
    gnb_antenna_model: str = 'parabolic'
    gnb_horizontal_hpbw_deg: float = 65.0
    gnb_vertical_hpbw_deg: float = 10.0
    gnb_front_back_db: float = 30.0
    gnb_sidelobe_db: float = 30.0
    aperture_efficiency: float = 0.65
    # Eq.(3)明确描述卫星Tx；UT接收图未公开，以下为独立重建假设。
    ut_antenna_model: str = 'bessel'
    antenna_back_attenuation_db: float = 55.0
    # 原文只定义前半球；另提供sin对称延拓以量化未知背瓣影响。
    antenna_rear_model: str = 'relative_floor'
    # 主profile遵循原文明确公开的Eq.(4) LOS式；混合NLOS只作敏感性。
    propagation_model: str = 'icc_los'
    shadow_fading: bool = False
    noise_temperature_k: float = 290.0
    noise_figure_db: float = 0.0
    thermal_noise_dbm_hz: float = -174.0
    ntn_extra_loss_db: float = 3.0
    gnb_min_power_dbm: float = 10.0
    ue_min_power_dbm: float = -10.0
    tilt_min_deg: float = 0.0
    tilt_max_deg: float = 20.0
    default_tilt_deg: float = 6.0
    inr_threshold_db: float = -6.0
    downlink_duty: float = 0.75
    uplink_duty: float = 0.25
    reward_throughput_weight: float = 1.0
    reward_active_weight: float = 0.5
    reward_violation_weight: float = 1.0
    ascent_radius_step_m: float = 500.0
    ascent_hysteresis_db: float = 3.0
    methods: tuple = ("No-coordination", "ASCENT-derived", "PPO")


CFG = SimConfig()
