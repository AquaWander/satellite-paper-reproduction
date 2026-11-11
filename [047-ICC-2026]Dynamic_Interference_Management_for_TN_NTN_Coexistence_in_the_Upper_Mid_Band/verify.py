"""独立行为检查；不以论文活跃率或曲线位置作为预设答案。"""
from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np
import torch

from config import CFG
from tn_ntn_model import (CoexistenceEnv, bessel_gain_db, db_to_linear,
                          fspl_db, slant_distance_m, uma_path_loss_db, ut_gain_db)
from ppo import ActorCritic


class PhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = CoexistenceEnv(CFG, 123, 1e-7, 1.0)
        cls.obs = cls.env.reset(seed=123)

    def test_units_and_geometry(self):
        self.assertAlmostEqual(float(fspl_db(1000, 12e9)), 114.0314, places=3)
        self.assertAlmostEqual(self.env.noise_dbm, -90.964888747874, places=6)
        self.assertAlmostEqual(float(slant_distance_m(90)), 600000, places=6)
        self.assertAlmostEqual(float(slant_distance_m(10)), float(slant_distance_m(170)), places=5)
        self.assertGreater(float(slant_distance_m(10)), float(slant_distance_m(90)))
        np.testing.assert_allclose(bessel_gain_db(0, np.array([33.0, 17.0])), [33, 17], atol=1e-10)
        np.testing.assert_allclose(bessel_gain_db([-120, -45], 33), bessel_gain_db([120, 45], 33))
        with self.assertRaises(ValueError):
            bessel_gain_db(181, 33)
        distances = np.array([10, 100, 1000, 5000, 10000])
        los = uma_path_loss_db(distances, CFG.frequency_hz, True)
        nlos = uma_path_loss_db(distances, CFG.frequency_hz, False)
        self.assertTrue(np.all(np.diff(los) > 0))
        self.assertTrue(np.all(nlos >= los))

    def test_linear_power_aggregation_and_silence(self):
        action = self.env.reference_action()
        result = self.env._evaluate(action, 0)
        total = result["dl_interference_mw"] + result["ul_interference_mw"]
        np.testing.assert_allclose(db_to_linear(result["inr_db"]) * self.env.noise_mw, total, rtol=1e-12)
        self.assertTrue(np.all(result["ul_interference_mw"] > 0))
        action[:, 3] = 0
        result = self.env._evaluate(action, 0)
        np.testing.assert_array_equal(result["dl_interference_mw"], 0)
        np.testing.assert_array_equal(result["ul_interference_mw"], 0)
        self.assertEqual(result["active_ratio"], 0)
        self.assertAlmostEqual(result["normalized_throughput"], 1)

    def test_seed_identity_and_counterfactual_types(self):
        e1 = CoexistenceEnv(CFG, 123, 1e-7, 1.0)
        e2 = CoexistenceEnv(CFG, 123, 1e-7, 0.0)
        a, b = e1.reset(seed=123), e2.reset(seed=123)
        np.testing.assert_array_equal(e1.gnb_xy, e2.gnb_xy)
        np.testing.assert_array_equal(e1.ntn_xy, e2.ntn_xy)
        np.testing.assert_array_equal(e1.ue_xy, e2.ue_xy)
        t1, t2 = [], []
        for _ in range(CFG.n_snapshots):
            _, _, _, v1 = e1.step(e1.reference_action())
            _, _, _, v2 = e2.step(e2.reference_action())
            t1.extend(v1["inr_db"])
            t2.extend(v2["inr_db"])
        self.assertLess(np.median(t1), np.median(t2))
        fresh = CoexistenceEnv(CFG, 123, 1e-7, 1.0)
        np.testing.assert_array_equal(a, fresh.reset(seed=123))

    def test_constraints_pass_and_ascent(self):
        env = CoexistenceEnv(CFG, 23, 3e-7, 1.0)
        obs = env.reset(seed=23)
        self.assertEqual(obs.shape, (429,))
        ratios = []
        for t in range(CFG.n_snapshots):
            action = env.baseline_action("ASCENT")
            obs, reward, done, info = env.step(action)
            self.assertTrue(np.isfinite(obs).all() and np.isfinite(reward))
            self.assertLessEqual(np.max(info["per_bs_max_inr_db"]), CFG.inr_threshold_db + 1e-5)
            self.assertTrue(np.all(info["dl_power_dbm"] <= CFG.gnb_max_power_dbm))
            self.assertTrue(np.all(info["ul_power_dbm"] <= CFG.ue_max_power_dbm))
            self.assertTrue(np.all((info["tilt_deg"] >= CFG.tilt_min_deg) & (info["tilt_deg"] <= CFG.tilt_max_deg)))
            ratios.append(info["active_ratio"])
            self.assertEqual(done, t == CFG.n_snapshots - 1)
        with self.assertRaises(RuntimeError):
            env.step(action)
        self.assertGreater(np.ptp(ratios), 0)
        with self.assertRaises(ValueError):
            env.reset(seed=23, density=-1)

    def test_invalid_actions(self):
        action = self.env.reference_action()
        for value in (np.nan, 1.2):
            bad = action.copy()
            bad[0, 0] = value
            with self.assertRaises(ValueError):
                self.env._decode_action(bad)
        bad = action.copy()
        bad[0, 3] = 0.5
        with self.assertRaises(ValueError):
            self.env._decode_action(bad)

    def test_gate_is_per_station_not_aggregate(self):
        env = CoexistenceEnv(CFG, 124, 6e-8, 1)
        env.reset(seed=124)
        limit = env.noise_mw * db_to_linear(CFG.inr_threshold_db)
        env.default_contributions_mw[:] = 0.9 * limit
        action = env.baseline_action("ASCENT-derived")
        self.assertTrue(np.all(action[:, 3] == 1))
        self.assertGreater(10*np.log10(CFG.n_gnb*0.9*limit/env.noise_mw), CFG.inr_threshold_db)

    def test_secondary_profile_units_and_peak(self):
        cfg = replace(CFG, ut_antenna_model="wbes_envelope", gnb_antenna_model="ascent_array", noise_temperature_k=200.)
        env = CoexistenceEnv(cfg, 125, 1e-7, 0.5)
        obs = env.reset(seed=125)
        np.testing.assert_allclose(ut_gain_db(0, np.array([33., 17.]), cfg), [33., 17.])
        np.testing.assert_allclose(ut_gain_db(90, np.array([33., 17.]), cfg), [-2., -2.])
        for _ in range(cfg.n_snapshots):
            obs, reward, done, info = env.step(env.baseline_action("ASCENT-derived"))
            self.assertTrue(np.isfinite(obs).all() and np.isfinite(reward))
            self.assertLessEqual(np.max(info["per_bs_max_inr_db"]), cfg.inr_threshold_db + 1e-5)


class PPOTests(unittest.TestCase):
    def test_joint_log_probability_and_gradient(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        model = ActorCritic()
        obs = torch.randn(4, 429)
        action, latent, active, old_logp, value = model.sample(obs)
        new_logp, new_value, entropy = model.evaluate(obs, latent, active)
        torch.testing.assert_close(new_logp, old_logp)
        torch.testing.assert_close((new_logp - old_logp).exp(), torch.ones(4))
        self.assertEqual(tuple(action.shape), (4, 106, 4))
        self.assertTrue(torch.all(action[:, :, :3].abs() <= 1))
        self.assertTrue(torch.all((active == 0) | (active == 1)))
        loss = -new_logp.mean() + new_value.square().mean() - 0.01 * entropy.mean()
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_shared_policy_permutation(self):
        torch.manual_seed(8)
        model = ActorCritic()
        obs = torch.randn(2, 429)
        order = torch.randperm(106)
        permuted = torch.cat((obs[:, :5], obs[:, 5:].reshape(2, 106, 4)[:, order].flatten(1)), dim=1)
        c1, b1, v1 = model.distributions(obs)
        c2, b2, v2 = model.distributions(permuted)
        torch.testing.assert_close(c1.mean[:, order], c2.mean)
        torch.testing.assert_close(b1.probs[:, order], b2.probs)
        torch.testing.assert_close(v1, v2)


def run_checks():
    suite = unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError("Independent behavior checks failed")
    return {"tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors)}


if __name__ == "__main__":
    run_checks()
