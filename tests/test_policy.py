"""Tests for LinearGaussianPolicy. The key one: the score function matches finite differences."""
import numpy as np
from scipy.stats import norm

from src.policy import LinearGaussianPolicy


def random_policy(d, seed=0):
    """A policy with non-trivial parameters (the zero init would hide bugs in dK)."""
    rng = np.random.default_rng(seed)
    pol = LinearGaussianPolicy(obs_dim=d)
    pol.set_flat(np.concatenate([rng.normal(size=2 * d + 2), np.log([0.3, 1.2])]))
    return pol


def test_score_matches_finite_differences(d=4):
    """d log pi(a|s) / d theta_k  ~=  [log pi_{theta + eps e_k} - log pi_{theta - eps e_k}] / (2 eps),
    checked for every sample and every parameter."""
    pol = random_policy(d)
    rng = np.random.default_rng(1)
    obs = rng.normal(size=(6, d))
    a = pol.sample(obs, rng)
    theta, eps = pol.get_flat(), 1e-6

    numeric = np.zeros((6, pol.n_params))
    for k in range(pol.n_params):
        e = np.zeros_like(theta)
        e[k] = eps
        pol.set_flat(theta + e)
        lp_plus = pol.log_prob(obs, a)
        pol.set_flat(theta - e)
        lp_minus = pol.log_prob(obs, a)
        numeric[:, k] = (lp_plus - lp_minus) / (2 * eps)
    pol.set_flat(theta)

    np.testing.assert_allclose(pol.score(obs, a), numeric, rtol=1e-5, atol=1e-6)


def test_log_prob_matches_scipy():
    pol = random_policy(4)
    rng = np.random.default_rng(2)
    obs = rng.normal(size=(5, 4))
    a = rng.normal(size=(5, 2))
    expected = norm.logpdf(a, loc=pol.mean(obs), scale=pol.std).sum(axis=1)
    np.testing.assert_allclose(pol.log_prob(obs, a), expected)


def test_score_has_zero_mean():
    """E_{a ~ pi}[grad log pi(a|s)] = 0: why subtracting a baseline adds no bias."""
    pol = random_policy(4)
    rng = np.random.default_rng(3)
    obs = np.tile(rng.normal(size=(1, 4)), (200_000, 1))   # one state, many actions
    a = pol.sample(obs, rng)
    np.testing.assert_allclose(pol.score(obs, a).mean(axis=0), 0.0, atol=0.03)


def test_init_and_sampling_statistics():
    pol = LinearGaussianPolicy(obs_dim=4, init_std=0.5)
    obs = np.random.default_rng(4).normal(size=(100_000, 4))
    np.testing.assert_array_equal(pol.act(obs), 0.0)            # K = 0, b = 0: mean action 0
    a = pol.act(obs, deterministic=False, rng=np.random.default_rng(5))
    np.testing.assert_allclose(a.mean(axis=0), 0.0, atol=0.01)
    np.testing.assert_allclose(a.std(axis=0), 0.5, rtol=0.01)


def test_flat_roundtrip_and_std_clamp():
    pol = random_policy(4)
    theta = pol.get_flat()
    pol.set_flat(theta)
    np.testing.assert_array_equal(pol.get_flat(), theta)
    theta[-2:] = [-10.0, 10.0]                                   # sigma far outside [0.05, 2]
    pol.set_flat(theta)
    np.testing.assert_allclose(pol.std, [0.05, 2.0])


def test_save_load_roundtrip(tmp_path):
    pol = random_policy(4)
    pol.save(tmp_path / "policy.npz")
    loaded = LinearGaussianPolicy.load(tmp_path / "policy.npz")
    np.testing.assert_array_equal(loaded.get_flat(), pol.get_flat())
