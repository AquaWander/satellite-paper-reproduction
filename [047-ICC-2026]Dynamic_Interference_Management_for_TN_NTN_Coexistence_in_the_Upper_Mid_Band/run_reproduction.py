"""官方核验后的双模型复现：同场景公平评估、学习归因、原图误差及文档。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

from config import CFG
from paper_assets import prepare_assets, compare_reference
from plotting import save_all
from ppo import load_policy, train
from simulation import evaluate, EVALUATION_SEEDS, METHODS, scenarios
from verify import run_checks

TRAINING_SEEDS = (11, 22, 33)


def experiment_profiles(base_output):
    return {
        "bessel": (CFG, base_output),
        "ascent-envelope": (replace(CFG, ut_antenna_model="wbes_envelope",
            gnb_antenna_model="ascent_array", noise_temperature_k=200.),
            base_output / "profiles" / "ascent_envelope"),
    }


def scientific_checks(arrays):
    same = all(np.array_equal(arrays["density"]["density_1e-07"][m]["inr_db"],
                              arrays["mix"]["T1_100_T2_0"][m]["inr_db"]) for m in METHODS)
    type_delta = {m: float(np.median(arrays["mix"]["T1_100_T2_0"][m]["inr_db"]) -
                           np.median(arrays["mix"]["T1_0_T2_100"][m]["inr_db"])) for m in METHODS}
    return {"identical_scenarios_identical_samples": same,
        "t1_minus_t2_median_db": type_delta,
        "type_order_within_paper_reading_resolution": {m: delta <= 0.2 for m, delta in type_delta.items()},
        "ppo_median_lower_than_no_coordination_all_scenarios": all(
            np.median(row["PPO"]["inr_db"]) < np.median(row["No-coordination"]["inr_db"])
            for group in arrays.values() for row in group.values()),
        "ascent_gate": "per-MBS checked each step, not an aggregate guarantee",
        "fig4_curve_count": sum(len(row) for row in arrays["density"].values()),
        "fig5_curve_count": sum(len(row) for row in arrays["mix"].values())}


def policy_pass(policy, cfg, seed, density, fraction, deterministic):
    from tn_ntn_model import CoexistenceEnv
    torch.manual_seed(700000 + seed)
    env = CoexistenceEnv(cfg, seed, density, fraction)
    obs = env.reset(seed=seed)
    rewards, active, expected_active, inr, eta, values = [], [], [], [], [], []
    for _ in range(cfg.n_snapshots):
        stats = policy.action_statistics(obs)
        values.append(stats["value"])
        expected_active.append(float(np.mean(stats["active_probability"])))
        obs, reward, done, info = env.step(policy.act(obs, deterministic=deterministic))
        rewards.append(reward)
        active.append(info["active_ratio"])
        eta.append(info["violation_ratio"])
        inr.extend(info["inr_db"])
    gamma = policy.metadata["ppo_settings"]["gamma"]
    returns = np.empty(len(rewards))
    total = 0.0
    for i in reversed(range(len(rewards))):
        total = rewards[i] + gamma * total
        returns[i] = total
    ev = 1 - np.var(returns - np.asarray(values)) / max(np.var(returns), 1e-12)
    return {"evaluation_seed": seed, "inference": "deterministic" if deterministic else "sampled",
        "density": density, "t1_fraction": fraction, "mean_reward": float(np.mean(rewards)),
        "active_ratio": float(np.mean(active)), "expected_active_probability": float(np.mean(expected_active)),
        "median_inr_db": float(np.median(inr)), "violation_ratio": float(np.mean(eta)),
        "mc_value_explained_variance": float(ev)}


def attribution(policies, initial_policies, cfg):
    rows = []
    for index, seed in enumerate(EVALUATION_SEEDS):
        for deterministic in (False, True):
            for label, policy in (("untrained", initial_policies[index]), ("trained", policies[index])):
                row = policy_pass(policy, cfg, seed, 1e-7, 1.0, deterministic)
                row.update(training_seed=TRAINING_SEEDS[index], policy=label)
                rows.append(row)
    return rows


def load_training_artifacts(root, out, cfg, episodes, retrain):
    expected_sources = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                        for name in ("config.py", "tn_ntn_model.py", "ppo.py")}
    policies, initial, logs = [], [], []
    for seed in TRAINING_SEEDS:
        checkpoint = out / "models" / f"ppo_seed_{seed}.pt"
        if retrain or not checkpoint.exists():
            train(cfg, seed, episodes, out)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload.get("source_sha256") != expected_sources:
            raise RuntimeError(f"Seed {seed} was trained on different source; run --train")
        if json.dumps(payload["simulation_config"], sort_keys=True) != json.dumps(asdict(cfg), sort_keys=True):
            raise RuntimeError("Model/profile configuration differs; run --train")
        initial_path = out / "models" / f"ppo_untrained_seed_{seed}.pt"
        initial_payload = torch.load(initial_path, map_location="cpu", weights_only=True)
        for key in ("source_sha256", "simulation_config", "seed", "n_gnb", "ppo_settings"):
            if initial_payload.get(key) != payload.get(key):
                raise RuntimeError(f"Initialization/trained checkpoint {key} differs")
        if payload.get("seed") != seed or initial_payload.get("episodes") != 0:
            raise RuntimeError("Unexpected checkpoint seed/initialization metadata")
        summary = json.loads((out / "training" / f"summary_seed_{seed}.json").read_text(encoding="utf-8"))
        for key in ("source_sha256", "seed", "episodes", "env_steps", "ppo_settings"):
            if summary.get(key) != payload.get(key):
                raise RuntimeError(f"Training summary/checkpoint {key} differs")
        log_path = out / "training" / f"episodes_seed_{seed}.csv"
        with log_path.open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != payload["episodes"] or any(int(row["seed"]) != seed for row in rows):
            raise RuntimeError("Episode CSV/model seed or budget differs")
        policies.append(load_policy(checkpoint))
        initial.append(load_policy(initial_path))
        logs.append(log_path)
    return policies, initial, logs, expected_sources


def execute_profile(root, out, cfg, name, args, checks):
    out.mkdir(parents=True, exist_ok=True)
    reference = prepare_assets(root, out)
    policies, initial, logs, sources = load_training_artifacts(root, out, cfg, args.episodes, args.train)
    arrays, summary = evaluate(policies, out, cfg, deterministic=args.inference == "deterministic")
    summary["metadata"]["profile"] = name
    summary["metadata"]["simulation_config"] = asdict(cfg)
    (out / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    comparison = compare_reference(summary, arrays, reference, out)
    validation = scientific_checks(arrays)
    if not validation["identical_scenarios_identical_samples"]:
        raise RuntimeError("Identical evaluation scenarios do not match")
    if validation["fig4_curve_count"] != 9 or validation["fig5_curve_count"] != 9:
        raise RuntimeError("Unexpected number of comparison curves")
    errors = np.array([abs(row["median_error_db"]) for row in comparison])
    validation.update(behavior_checks=checks, paper_curve_median_mae_db=float(np.mean(errors)),
        paper_curves_within_3db=int(np.count_nonzero(errors <= 3)),
        paper_curves_within_5db=int(np.count_nonzero(errors <= 5)),
        paper_median_target_3db_met=bool(np.mean(errors) <= 3),
        paper_numerical_match=bool(np.max(errors) <= 3),
        attribution=attribution(policies, initial, cfg))
    validation["value_function_validation"] = [
        row for row in validation["attribution"] if row["policy"] == "trained" and row["inference"] == "sampled"]
    (out / "verification.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    mode_rows = []
    for _, scenario, density, fraction in scenarios(cfg):
        for index, seed in enumerate(EVALUATION_SEEDS):
            row = policy_pass(policies[index], cfg, seed, density, fraction, True)
            row.update(scenario=scenario, training_seed=TRAINING_SEEDS[index])
            mode_rows.append(row)
    (out / "deterministic_policy_evaluation.json").write_text(json.dumps(mode_rows, indent=2), encoding="utf-8")
    save_all(arrays, out, logs)
    import scipy
    import matplotlib
    manifest = {"created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "python": platform.python_version(), "profile": name,
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__, "matplotlib": matplotlib.__version__},
        "simulation_config": asdict(cfg), "training_seeds": TRAINING_SEEDS,
        "evaluation_seeds": EVALUATION_SEEDS, "source_sha256": sources,
        "pipeline_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob("*.py")},
        "inference": args.inference, "action_seed_offset": 700000}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"{name}: plotted median MAE={np.mean(errors):.3f} dB", flush=True)


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--output", type=Path, default=root / "output")
    parser.add_argument("--profile", choices=("bessel", "ascent-envelope", "all"), default="all")
    parser.add_argument("--inference", choices=("sampled", "deterministic"), default="sampled")
    args = parser.parse_args()
    torch.set_num_threads(1)
    started = time.perf_counter()
    checks = run_checks()
    output = args.output.resolve()
    completed = 0
    for name, (cfg, destination) in experiment_profiles(output).items():
        if args.profile not in ("all", name):
            continue
        execute_profile(root, destination, cfg, name, args, checks)
        completed += 1
    print(f"Completed {completed} profile(s), {time.perf_counter()-started:.1f} s", flush=True)


if __name__ == "__main__":
    main()
