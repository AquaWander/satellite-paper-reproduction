"""冻结物理模型的同输入链路预算与敏感性诊断；不修改模型或拟合原图。

运行：python diagnostics.py。所有干预均为单因素诊断，不是被采纳的参数。
"""
from __future__ import annotations

import csv
import json
import ast
import cmath
import math
import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np

from config import CFG
from tn_ntn_model import (CoexistenceEnv, _angle_deg, bessel_gain_db,
    db_to_linear, linear_to_db, uma_path_loss_db)


def official_array_gain(theta, phi, tilt, scan, rows=16, columns=16):
    """独立向量化重写标准均匀平面阵列求和，与官方同角度函数交叉验证。

    theta是从天顶向下量的zenith；官方weighting中的正tilt表示朝上，
    因此物理正下倾角必须以负号传入；不复制官方坐标换算错误。
    """
    theta, phi, tilt, scan = np.broadcast_arrays(theta, phi, tilt, scan)
    th, ph, ti, sc = [np.deg2rad(x) for x in (theta, phi, tilt, scan)]
    qv = 0.5 * (np.cos(th) - np.sin(ti))
    qh = 0.5 * (np.sin(th) * np.sin(ph) - np.cos(ti) * np.sin(sc))
    def normalized_sum(n, q):
        den = np.sin(np.pi * q)
        value = np.ones_like(q)
        np.divide(np.sin(n * np.pi * q), n * den, out=value, where=np.abs(den) > 1e-12)
        return value
    af = rows * columns * normalized_sum(rows, qv) ** 2 * normalized_sum(columns, qh) ** 2
    element = 8.0 - np.minimum(np.minimum(12 * (phi / 80) ** 2, 30) + np.minimum(12 * ((theta - 90) / 65) ** 2, 30), 30)
    return element + linear_to_db(af)


def official_wbes_gain(phi):
    """官方WBES_B前/侧瓣定义，只用于ASCENT型接收器敏感性，非T1/T2替换结论。"""
    phi = np.abs(np.asarray(phi))
    return np.where(phi < 6, 20.0, np.where(phi < 48, 40 - 25 * np.log10(np.maximum(phi, 1e-12)), -2.0))


def validate_official_functions(root):
    """不导入服务器/缓存，仅运行官方文件中的纯数学函数。"""
    source = root / 'external_sources' / 'ASCENT' / 'Simulator.py'
    if not source.exists():
        return {'available': False}
    wanted = {'gain_antenna_element_horizontal', 'gain_antenna_element_vertical',
              'gain_antenna_element', 'superposition', 'weighting', 'beam_pattern_5g', 'gain_5g', 'gain_fss_wbes_b'}
    parsed = ast.parse(source.read_text(encoding='utf-8-sig'))
    module = ast.Module(body=[n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name in wanted], type_ignores=[])
    namespace = {'math': math, 'cmath': cmath}
    exec(compile(ast.fix_missing_locations(module), str(source), 'exec'), namespace)
    examples = []
    rng = np.random.default_rng(20261109)
    for i in range(40):
        theta, phi = rng.uniform(0, 180), rng.uniform(-180, 180)
        tilt, scan = rng.uniform(-45, 45), rng.uniform(-90, 90)
        official = namespace['gain_5g'](theta, phi, tilt, scan)
        ours = float(official_array_gain(theta, phi, tilt, scan))
        examples.append({'theta_deg': theta, 'phi_deg': phi, 'tilt_deg': tilt, 'scan_deg': scan,
                         'official_gain_dbi': official, 'independent_gain_dbi': ours, 'error_db': ours - official})
    rx_error = max(abs(float(official_wbes_gain(x)) - namespace['gain_fss_wbes_b'](x)) for x in np.linspace(0, 180, 361))
    return {'available': True, 'array_max_abs_error_db': max(abs(x['error_db']) for x in examples),
            'rx_max_abs_error_db': rx_error, 'examples': examples,
            'interpretation': '相同角度、相同权重的纯函数一致性，不包括官方错误坐标换算或整案例一致性。'}


def compare_control_profiles(seed, density):
    """相同物理输入比较aggregate/EZ与官方per-MBS gate，阈值不调参。"""
    env = CoexistenceEnv(CFG, seed, density=density, t1_fraction=1.0)
    env.reset(seed=seed)
    c = env.cfg
    sat = env.sat_xyz[:, None, :] - env.ntn_xyz[None, :, :]
    vv = env.gnb_xyz[None, :, None, :] - env.ntn_xyz[None, None, :, :]
    uv = env.ue_xyz[:, :, None, :] - env.ntn_xyz[None, None, :, :]
    angle_dl, angle_ul = _angle_deg(sat[:, None, :, :], vv), _angle_deg(sat[:, None, :, :], uv)
    pl_dl = uma_path_loss_db(np.linalg.norm(vv[..., :2], axis=-1), c.frequency_hz, True, c.gnb_height_m, c.terminal_height_m)
    pl_ul = uma_path_loss_db(np.linalg.norm(uv[..., :2], axis=-1), c.frequency_hz, True, c.gnb_height_m, c.terminal_height_m)
    gr_dl = bessel_gain_db(angle_dl, env.terminal_peak_dbi[None, None, :], c.frequency_hz, c.aperture_efficiency, c.antenna_back_attenuation_db)
    gr_ul = bessel_gain_db(angle_ul, env.terminal_peak_dbi[None, None, :], c.frequency_hz, c.aperture_efficiency, c.antenna_back_attenuation_db)
    ga = official_array_gain(90 + env.downward_deg, env.horizontal_deg, -c.default_tilt_deg, 0.0)[None, :, :]
    profiles = {
        'ICC_literal_LOS_kTB290_Bessel': (
            db_to_linear(c.gnb_max_power_dbm + env._default_gnb_gain[None, :, :] + gr_dl - pl_dl) * c.downlink_duty
            + db_to_linear(c.ue_max_power_dbm + gr_ul - pl_ul) * c.uplink_duty,
            1.38064852e-23 * 290 * c.bandwidth_hz * 1000),
        'ASCENT_compatible_LOS_kTB200_WBES_actualArray': (
            db_to_linear(c.gnb_max_power_dbm + ga + official_wbes_gain(angle_dl) - pl_dl) * c.downlink_duty
            + db_to_linear(c.ue_max_power_dbm + official_wbes_gain(angle_ul) - pl_ul) * c.uplink_duty,
            1.38064852e-23 * 200 * c.bandwidth_hz * 1000),
    }
    records = []
    for name, (contribution, noise) in profiles.items():
        for control in ('No-coordination', 'per_MBS_instantaneous_gate', 'per_MBS_latching_gate', 'aggregate_EZ_hysteresis'):
            samples, activities, harms = [], [], []
            active_previous = np.ones(c.n_gnb, dtype=bool)
            radius = 0.0
            limit = noise * db_to_linear(c.inr_threshold_db)
            for t in range(c.n_snapshots):
                if control == 'No-coordination':
                    keep = np.ones(c.n_gnb, dtype=bool)
                elif control.startswith('per_MBS'):
                    keep = np.max(contribution[t], axis=1) <= limit
                    if control == 'per_MBS_latching_gate':
                        keep &= active_previous
                    active_previous = keep
                else:
                    def worst(r):
                        allowed = env.nearest_ntn_distance_m >= r
                        return float(np.max(np.sum(contribution[t, allowed], axis=0))) if np.any(allowed) else 0.0
                    while radius > 0 and worst(max(0, radius-c.ascent_radius_step_m)) <= limit * db_to_linear(-c.ascent_hysteresis_db):
                        radius = max(0, radius-c.ascent_radius_step_m)
                    while worst(radius) > limit and radius <= np.max(env.nearest_ntn_distance_m):
                        radius += c.ascent_radius_step_m
                    keep = env.nearest_ntn_distance_m >= radius
                inr = linear_to_db(np.sum(contribution[t, keep], axis=0) / noise)
                samples.extend(inr)
                activities.append(float(np.mean(keep)))
                harms.append(float(np.mean(inr > c.inr_threshold_db)))
            q = np.percentile(samples, [5, 25, 50, 75, 95])
            records.append({'seed': seed, 'density': density, 'profile': name, 'control': control,
                'threshold_db': c.inr_threshold_db, 'median_inr_db': q[2], 'q05_db': q[0], 'q25_db': q[1],
                'q75_db': q[3], 'q95_db': q[4], 'active_ratio': float(np.mean(activities)),
                'harmful_fraction': float(np.mean(harms)), 'active_min': min(activities), 'active_max': max(activities)})
    return records


def frozen_budget(seed=81001, density=1e-7, fraction=1.0):
    # 显式重建先前待诊断profile，避免主配置修正后悄悄改变“旧模型”定义。
    legacy_temperature = 10 ** ((-174.0 - 30.0) / 10) / 1.38064852e-23
    frozen_cfg = replace(CFG, propagation_model='uma_mixed', shadow_fading=True,
                         noise_temperature_k=legacy_temperature, noise_figure_db=7.0,
                         antenna_rear_model='relative_floor', ut_antenna_model='bessel',
                         gnb_antenna_model='parabolic')
    env = CoexistenceEnv(frozen_cfg, seed, density=density, t1_fraction=fraction)
    class CapturingRNG:
        """记录模型自身产生的阴影随机量，不改抽样次序或分布。"""
        def __init__(self, original):
            self.original, self.normal_draws = original, []
        def __getattr__(self, name):
            return getattr(self.original, name)
        def normal(self, *args, **kwargs):
            value = self.original.normal(*args, **kwargs)
            self.normal_draws.append(np.asarray(value).copy())
            return value
    recorder = CapturingRNG(np.random.default_rng(seed))
    env.rng = recorder
    env.reset()
    c = env.cfg
    sat_vec = env.sat_xyz[:, None, :] - env.ntn_xyz[None, :, :]
    dl_vec = env.gnb_xyz[None, :, None, :] - env.ntn_xyz[None, None, :, :]
    ul_vec = env.ue_xyz[:, :, None, :] - env.ntn_xyz[None, None, :, :]
    dl_angle = _angle_deg(sat_vec[:, None, :, :], dl_vec)
    ul_angle = _angle_deg(sat_vec[:, None, :, :], ul_vec)
    peaks = env.terminal_peak_dbi[None, None, :]
    rx_dl = bessel_gain_db(dl_angle, peaks, c.frequency_hz, c.aperture_efficiency, c.antenna_back_attenuation_db)
    rx_ul = bessel_gain_db(ul_angle, peaks, c.frequency_hz, c.aperture_efficiency, c.antenna_back_attenuation_db)
    pl_dl = rx_dl - linear_to_db(env.dl_coupling_mw_per_mw)
    pl_ul = rx_ul - linear_to_db(env.ul_coupling_mw_per_mw)
    tx_dl = np.broadcast_to(env._default_gnb_gain[None, :, :], pl_dl.shape)
    dl = env.dl_coupling_mw_per_mw * env._default_dl_weight[None, :, :]
    ul = env.ul_coupling_mw_per_mw * env._default_ul_weight
    distance_dl = np.linalg.norm(dl_vec[..., :2], axis=-1)
    distance_ul = np.linalg.norm(ul_vec[..., :2], axis=-1)
    los_dl = uma_path_loss_db(distance_dl, c.frequency_hz, True, c.gnb_height_m, c.terminal_height_m)
    los_ul = uma_path_loss_db(distance_ul, c.frequency_hz, True, c.gnb_height_m, c.terminal_height_m)
    shadow_dl = recorder.normal_draws[0][None, :, :]
    shadow_ul = recorder.normal_draws[1][None, :, :]

    def inr(dl_power, ul_power, noise=env.noise_mw):
        return linear_to_db((np.sum(dl_power, axis=1) + np.sum(ul_power, axis=1)) / noise)

    base = inr(dl, ul)
    # 每项只替换所述因素，保留同样的站址、用户、卫星轨迹与其他参数。
    candidates = {
        'frozen_current': (dl, ul, env.noise_mw,
            '冻结当前模型；不是原作者实现'),
        'all_LOS_no_shadow': (dl * db_to_linear(pl_dl - los_dl), ul * db_to_linear(pl_ul - los_ul), env.noise_mw,
            '仅用原文Eq4 LOS距离式，删除NLOS与随机阴影；用于传播敏感性，不自动采纳'),
        'current_LOS_NLOS_no_shadow': (dl * db_to_linear(shadow_dl), ul * db_to_linear(shadow_ul), env.noise_mw,
            '保留原LOS/NLOS选择，只去掉记录到的原阴影项，不推断LOS标签'),
        'all_LOS_keep_original_shadow': (dl * db_to_linear(pl_dl - shadow_dl - los_dl), ul * db_to_linear(pl_ul - shadow_ul - los_ul), env.noise_mw,
            '仅把NLOS基式替换为LOS，保留实际抽到的阴影量，隔离路径损耗分支'),
        'RX_boresight_bound': (dl * db_to_linear(peaks - rx_dl), ul * db_to_linear(peaks - rx_ul), env.noise_mw,
            '干扰从所有接收主轴入射的上界；与真实几何不符，不是候选修正'),
        'BS_boresight_bound': (dl * db_to_linear(c.gnb_peak_gain_dbi - tx_dl), ul, env.noise_mw,
            '所有gNB均向每个NTN接收端主轴照射的上界；不是物理可同时实现方案'),
        'DL_64_element_gain_bound': (dl * 64.0, ul, env.noise_mw,
            '仅检查10log10(64)=18.06dB阵列上界对DL影响，不代表官方真实离轴阵列增益'),
        'UL_16_equal_streams_bound': (dl, ul * 16.0, env.noise_mw,
            '每小区16个等功率共频UL流的上界；不能等同16个正交调度用户'),
        'noise_50MHz_only': (dl, ul, env.noise_mw / 4,
            '只将噪声带宽改为50MHz而保持干扰不变，隔离6.02dB噪声项；频谱重叠尚未同步'),
        'noise_no_receiver_NF': (dl, ul, env.noise_mw / db_to_linear(c.noise_figure_db),
            '仅删除当前7dB接收机噪声系数，隔离与论文kTB定义的差别'),
        'full_DL_and_UL_duty_bound': (dl / c.downlink_duty, ul / c.uplink_duty, env.noise_mw,
            '两方向都视作持续共频发射的干扰上界，删除当前0.75/0.25占空比'),
    }
    los_dd = dl * db_to_linear(pl_dl - los_dl)
    los_uu = ul * db_to_linear(pl_ul - los_ul)
    noise_kTB_290 = env.noise_mw / db_to_linear(c.noise_figure_db)
    noise_kTB_200 = 1.38064852e-23 * 200.0 * c.bandwidth_hz * 1000
    # 独立向量化阵列使用真实方位/zenith和固定服务波束；绝不用峰值上界代替。
    actual_array = official_array_gain(90 + env.downward_deg, env.horizontal_deg,
                                      -c.default_tilt_deg, 0.0)[None, :, :]
    array_dd = los_dd * db_to_linear(actual_array - tx_dl)
    wb_dl = db_to_linear(official_wbes_gain(dl_angle) - rx_dl)
    wb_ul = db_to_linear(official_wbes_gain(ul_angle) - rx_ul)
    candidates.update({
        'LOS_kTB290_Bessel_currentTx': (los_dd, los_uu, noise_kTB_290,
            '原文Eq4 LOS、无阴影、kTB不加NF；保留当前Bessel孔径假设与Tx方向图'),
        'LOS_kTB200_Bessel_currentTx': (los_dd, los_uu, noise_kTB_200,
            '按官方200K温度和ICC200MHz总带宽，LOS与当前Bessel/Tx；不是50MHz噪声拟合'),
        'LOS_kTB200_WBES_currentTx': (los_dd * wb_dl, los_uu * wb_ul, noise_kTB_200,
            '将接收器换为官方WBES_B用于敏感性，不能声称等价ICC的Bessel T1/T2'),
        'LOS_kTB200_Bessel_officialArray': (array_dd, los_uu, noise_kTB_200,
            '采用官方16x16标准阵列函数但真实几何及固定服务波束；总Pt仍ICC33dBm'),
        'LOS_kTB200_WBES_officialArray': (array_dd * wb_dl, los_uu * wb_ul, noise_kTB_200,
            '官方WBES_B+16x16阵列的移植敏感性：真实几何，ICC33dBm总功率，不复制原坐标错误'),
    })
    rows = []
    for name, (dd, uu, noise, comment) in candidates.items():
        x = inr(dd, uu, noise)
        q = np.percentile(x, [5, 25, 50, 75, 95])
        rows.append({'seed': seed, 'density': density, 't1_fraction': fraction,
            'experiment': name, 'q05_db': q[0], 'q25_db': q[1], 'median_db': q[2],
            'q75_db': q[3], 'q95_db': q[4], 'paired_median_shift_db': float(np.median(x - base)),
            'median_change_db': float(np.median(x) - np.median(base)), 'interpretation': comment})

    # 每个接收端每快照仅记录其最大单站DL贡献，避免用海量无关远距链路统计解释聚合INR。
    index = np.argmax(dl, axis=1)
    ti = np.arange(c.n_snapshots)[:, None]
    ui = np.arange(env.n_ntn)[None, :]
    dominant = []
    for t in range(c.n_snapshots):
        for u in range(env.n_ntn):
            i = int(index[t, u])
            dominant.append({'seed': seed, 'snapshot': t, 'user': u, 'gnb': i,
                'distance_m': float(distance_dl[0, i, u]),
                'offaxis_rx_deg': float(dl_angle[t, i, u]),
                'tx_power_dbm': c.gnb_max_power_dbm, 'tx_gain_dbi': float(tx_dl[t, i, u]),
                'rx_gain_dbi': float(rx_dl[t, i, u]), 'path_loss_db': float(pl_dl[t, i, u]),
                'los_only_path_loss_db': float(los_dl[0, i, u]),
                'dl_duty_db': float(linear_to_db(c.downlink_duty)),
                'dominant_dl_received_dbm': float(linear_to_db(dl[t, i, u])),
                'aggregate_dl_received_dbm': float(linear_to_db(np.sum(dl[t, :, u]))),
                'aggregate_ul_received_dbm': float(linear_to_db(np.sum(ul[t, :, u]))),
                'noise_dbm': env.noise_dbm, 'aggregate_inr_db': float(base[t, u])})
    medians = {key: float(np.median([r[key] for r in dominant])) for key in dominant[0]
               if key not in ('seed', 'snapshot', 'user', 'gnb')}
    return rows, dominant, {'seed': seed, 'density': density, 'fraction': fraction,
        'dominant_link_component_medians': medians, 'metadata': env.metadata,
        'note': '中位分量相加不等于聚合INR中位数；请使用逐样本表逐项核算。'}


def main():
    root = Path(__file__).resolve().parent
    output = root / 'output' / 'diagnostics'
    output.mkdir(parents=True, exist_ok=True)
    rows, links, summaries = [], [], []
    for seed in (81001, 81002, 81003):
        r, d, s = frozen_budget(seed)
        rows.extend(r)
        links.extend(d)
        summaries.append(s)
    for filename, data in (('frozen_ablation.csv', rows), ('dominant_link_budget.csv', links)):
        with (output / filename).open('w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    (output / 'frozen_budget_summary.json').write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding='utf-8')
    function_comparison = validate_official_functions(root)
    (output / 'official_function_comparison.json').write_text(json.dumps(function_comparison, indent=2), encoding='utf-8')
    print('official function comparison',function_comparison.get('array_max_abs_error_db'),function_comparison.get('rx_max_abs_error_db'))
    provenance = {'core_sha256': {n: hashlib.sha256((root / n).read_bytes()).hexdigest()
        for n in ('config.py', 'tn_ntn_model.py')},
        'diagnostics_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'note': '只做冻结模型诊断；不代表已采纳任何诊断干预。'}
    (output / 'diagnostic_provenance.json').write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding='utf-8')
    controls = []
    for seed in (81001, 81002, 81003):
        for density in CFG.density_values:
            controls.extend(compare_control_profiles(seed, density))
    with (output / 'profile_controller_comparison.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(controls[0]))
        writer.writeheader()
        writer.writerows(controls)
    sensitivity_rows = []
    sensitivities = {
        'main_bessel_relative_rear': CFG,
        'bessel_symmetric_rear': replace(CFG, antenna_rear_model='symmetric'),
        'mixed_NLOS_no_shadow': replace(CFG, propagation_model='uma_mixed'),
        'mixed_NLOS_with_shadow': replace(CFG, propagation_model='uma_mixed', shadow_fading=True),
        'official_envelope_actual_array': replace(CFG, ut_antenna_model='wbes_envelope',
            gnb_antenna_model='ascent_array', noise_temperature_k=200.0),
    }
    for profile, cfg in sensitivities.items():
        for seed in (81001, 81002, 81003):
            for method in ('No-coordination', 'ASCENT-derived'):
                env = CoexistenceEnv(cfg, seed, density=1e-7, t1_fraction=1.0)
                env.reset(seed=seed)
                values, activities, violations = [], [], []
                for t in range(cfg.n_snapshots):
                    _, _, _, info = env.step(env.baseline_action(method))
                    values.extend(info['inr_db'])
                    activities.append(info['active_ratio'])
                    violations.append(info['violation_ratio'])
                sensitivity_rows.append({'profile': profile, 'seed': seed, 'method': method,
                    'density': 1e-7, 't1_fraction': 1.0, 'median_inr_db': float(np.median(values)),
                    'q05_db': float(np.percentile(values, 5)), 'q95_db': float(np.percentile(values, 95)),
                    'active_ratio': float(np.mean(activities)), 'harmful_fraction': float(np.mean(violations))})
    with (output / 'profile_sensitivity.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(sensitivity_rows[0]))
        writer.writeheader()
        writer.writerows(sensitivity_rows)
    for profile in sorted(set(r['profile'] for r in controls)):
        for density in CFG.density_values:
            for control in ('No-coordination', 'per_MBS_instantaneous_gate', 'per_MBS_latching_gate', 'aggregate_EZ_hysteresis'):
                group = [r for r in controls if r['profile']==profile and r['density']==density and r['control']==control]
                print(profile,density,control,'median',round(float(np.mean([r['median_inr_db'] for r in group])),3),
                      'active',round(float(np.mean([r['active_ratio'] for r in group])),4),flush=True)
    for row in rows:
        print(row['seed'], row['experiment'], round(row['median_db'], 3), round(row['median_change_db'], 3), flush=True)


if __name__ == '__main__':
    main()
