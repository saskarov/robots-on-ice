"""Tests for evaluate(): consistent metrics, reproducible test set, trajectories CSV round trip."""
import numpy as np
import pandas as pd
import pytest

from src.config import EnvConfig
from src.evaluate import evaluate, trajectories_frame, trajectories_from_frame
from src.policy import LinearGaussianPolicy


def spring_damper(kp=3.0, kd=3.0):
    """A hand-set linear policy a = -kp (x - g)/30 - kd v/10: push to the target, damp the speed."""
    pol = LinearGaussianPolicy()
    pol.K = np.array([[-kp, 0.0, -kd, 0.0], [0.0, -kp, 0.0, -kd]])
    return pol


def test_evaluate_is_consistent_and_reproducible():
    cfg = EnvConfig(m=75, C=0.5)
    res = evaluate(spring_damper(), cfg, n_episodes=20)
    assert res == evaluate(spring_damper(), cfg, n_episodes=20)   # same test episodes
    assert res["return_mean"] == pytest.approx(-(res["dist_cost"] + res["energy_cost"]))


def test_pushing_to_the_target_beats_doing_nothing():
    cfg = EnvConfig(m=75, C=0.5)
    zero = evaluate(LinearGaussianPolicy(), cfg, n_episodes=20)    # K = 0, b = 0: no force
    assert zero["energy_cost"] == 0.0 and zero["frac_reached"] < 0.1
    assert evaluate(spring_damper(), cfg, n_episodes=20)["return_mean"] > zero["return_mean"]


def test_trajectories_survive_the_csv_round_trip(tmp_path):
    traj = evaluate(spring_damper(), EnvConfig(), n_episodes=10, n_trajectories=3)["trajectories"]
    trajectories_frame(traj).to_csv(tmp_path / "t.csv", index=False)
    pos, targets = trajectories_from_frame(pd.read_csv(tmp_path / "t.csv"))
    np.testing.assert_allclose(pos, traj["pos"])
    np.testing.assert_allclose(targets, traj["targets"])
