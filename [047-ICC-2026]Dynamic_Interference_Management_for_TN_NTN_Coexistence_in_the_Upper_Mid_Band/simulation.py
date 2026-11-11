"""固定独立评估种子；所有方法使用相同布局、用户与卫星过境。"""
from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from config import CFG
from tn_ntn_model import CoexistenceEnv
from baselines import ascent_action, no_coordination_action

METHODS = ("No-coordination", "ASCENT-derived", "PPO")
EVALUATION_SEEDS = (91001, 91002, 91003)


def scenarios(cfg=CFG):
    """Fig.4 为全 T1；Fig.5 固定中等密度，按用户类型分配。"""
    for density in cfg.density_values:
        yield "density", f"density_{density:.0e}", density, 1.0
    for fraction in cfg.mix_values:
        yield "mix", f"T1_{int(100*fraction)}_T2_{int(100*(1-fraction))}", 1e-7, fraction


def evaluate(policies, out: Path, cfg=CFG, seeds=EVALUATION_SEEDS, deterministic=False):
    """每个独立场景种子配一个训练种子；方差含两类随机性。"""
    if not policies or not seeds:
        raise ValueError("At least one policy and one evaluation seed are required")
    out.mkdir(parents=True, exist_ok=True)
    arrays = {}
    episode_rows = []
    control_rows = []
    layouts = []
    with gzip.open(out / "inr_samples.csv.gz", "wt", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["group", "scenario", "method", "evaluation_seed", "snapshot", "user", "inr_db"])
        for group, name, density, fraction in scenarios(cfg):
            arrays.setdefault(group, {})[name] = {}
            for method in METHODS:
                collected, active, violation, rates, per_seed = [], [], [], [], []
                for run_index, seed in enumerate(seeds):
                    policy = policies[run_index % len(policies)]
                    if method == "PPO":
                        torch.manual_seed(700000 + seed)
                    env = CoexistenceEnv(cfg, seed=seed, density=density, t1_fraction=fraction)
                    obs = env.reset(seed=seed, density=density, t1_fraction=fraction)
                    if method == METHODS[0]:
                        layouts.append({"group": group, "scenario": name, "seed": seed,
                            "environment": env.metadata, "gnb_xy_m": env.gnb_xy.tolist(),
                            "ntn_xy_m": env.ntn_xy.tolist(), "is_t1": env.is_t1.tolist()})
                    episode_inr, episode_active, episode_eta, episode_rate = [], [], [], []
                    done = False
                    snapshot = 0
                    while not done:
                        active_probability = None
                        if method == "PPO":
                            active_probability = float(np.mean(policy.action_statistics(obs)["active_probability"]))
                            action = policy.act(obs, deterministic=deterministic)
                        else:
                            action = ascent_action(env) if method == "ASCENT-derived" else no_coordination_action(env)
                        obs, reward, done, info = env.step(action)
                        inr = np.asarray(info["inr_db"], dtype=float)
                        ratio = float(info["active_ratio"])
                        eta = float(info["violation_ratio"])
                        rate = float(info["throughput_bps"])
                        if not (np.isfinite(inr).all() and np.isfinite(rate) and 0 <= ratio <= 1 and 0 <= eta <= 1):
                            raise ValueError(f"Invalid numerical result: {group}/{name}/{method}")
                        per_bs_max = float(np.max(info["per_bs_max_inr_db"]))
                        if method == "ASCENT-derived" and per_bs_max > cfg.inr_threshold_db + 1e-5:
                            raise ValueError("ASCENT-derived per-MBS gate violated")
                        episode_inr.extend(inr)
                        episode_active.append(ratio)
                        episode_eta.append(eta)
                        episode_rate.append(rate)
                        control_rows.append({"group": group, "scenario": name, "method": method,
                            "evaluation_seed": seed, "snapshot": snapshot,
                            "elevation_deg": info["elevation_deg"],
                            "dl_power_mean_dbm": float(np.mean(info["dl_power_dbm"])),
                            "ul_power_mean_dbm": float(np.mean(info["ul_power_dbm"])),
                            "tilt_mean_deg": float(np.mean(info["tilt_deg"])),
                            "active_count": int(np.count_nonzero(info["active_mask"])),
                            "ascent_radius_m": info["ascent_radius_m"], "active_ratio": ratio,
                            "policy_active_probability": active_probability,
                            "max_per_bs_inr_db": per_bs_max,
                            "violation_ratio": eta, "throughput_bps": rate, "reward": float(reward)})
                        writer.writerows([group, name, method, seed, snapshot, user, f"{value:.9f}"] for user, value in enumerate(inr))
                        snapshot += 1
                    episode_median = float(np.median(episode_inr))
                    per_seed.append(episode_median)
                    collected.extend(episode_inr)
                    active.append(float(np.mean(episode_active)))
                    violation.append(float(np.mean(episode_eta)))
                    rates.append(float(np.mean(episode_rate)))
                    episode_rows.append({"group": group, "scenario": name, "method": method,
                        "evaluation_seed": seed, "policy_index": run_index % len(policies) if method == "PPO" else None,
                        "snapshots": snapshot, "median_inr_db": episode_median,
                        "active_ratio": active[-1], "violation_ratio": violation[-1],
                        "throughput_bps": rates[-1]})
                arrays[group][name][method] = {"inr_db": np.asarray(collected),
                    "active_per_seed": np.asarray(active), "violation_per_seed": np.asarray(violation),
                    "throughput_per_seed": np.asarray(rates), "median_per_seed": np.asarray(per_seed)}
            print(f"Evaluated {group}/{name}", flush=True)
    with (out / "episode_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(episode_rows[0]))
        writer.writeheader()
        writer.writerows(episode_rows)
    with (out / "control_trace.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(control_rows[0]))
        writer.writeheader()
        writer.writerows(control_rows)
    (out / "scenario_metadata.json").write_text(json.dumps(layouts, indent=2), encoding="utf-8")
    summary = summarize(arrays, cfg, seeds)
    summary["metadata"]["ppo_inference"] = "deterministic" if deterministic else "sampled with fixed independent action seed"
    summary["metadata"]["action_seeds"] = [700000 + seed for seed in seeds]
    (out / "results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return arrays, summary


def summarize(arrays, cfg=CFG, seeds=EVALUATION_SEEDS):
    summary = {"metadata": {"evaluation_seeds": list(seeds), "methods": list(METHODS),
        "uncertainty": "sample SD across independent pass/policy seed pairs; not per-user confidence intervals",
        "baseline": "per-MBS threshold gate derived from checked official ASCENT; aggregate protection not guaranteed",
        "geometry": "synthetic BS positions unless a coordinate CSV is configured",
        "threshold_db": cfg.inr_threshold_db}, "density": {}, "mix": {}}
    for group, data in arrays.items():
        for scenario, methods in data.items():
            summary[group][scenario] = {}
            for method, data in methods.items():
                values = data["inr_db"]
                row = {"sample_count": len(values), "percentiles": [5, 10, 25, 50, 75, 90, 95],
                    "percentiles_db": np.percentile(values, [5, 10, 25, 50, 75, 90, 95]).tolist(),
                    "median_inr_db": float(np.median(values))}
                for label, key in (("active_ratio", "active_per_seed"), ("violation_ratio", "violation_per_seed"),
                                   ("throughput_bps", "throughput_per_seed"), ("seed_median_inr_db", "median_per_seed")):
                    x = data[key]
                    row[label] = float(np.mean(x))
                    row[label + "_sd"] = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
                    row[label + "_per_seed"] = x.tolist()
                summary[group][scenario][method] = row
    return summary
