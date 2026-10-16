"""Tests for returns-to-go, the GPOMDP weights and gradient, Adam, and a short training run."""
import numpy as np
import pytest

from src.config import EnvConfig, TrainConfig
from src.gpomdp import advantages, policy_gradient, returns_to_go, train
from src.optim import Adam


def test_returns_to_go_hand_computed():
    """r = (1, 2, 3):  G_2 = 3,  G_1 = 2 + 3 = 5,  G_0 = 1 + 5 = 6."""
    R = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 1.0]])         # two episodes as columns
    np.testing.assert_allclose(returns_to_go(R), [[6.0, 1.0], [5.0, 1.0], [3.0, 1.0]])


def test_baseline_gives_zero_mean_weights_per_step():
    """b_t = mean over episodes of G_t, so the weights average to 0 at every time step."""
    R = np.random.default_rng(1).normal(size=(10, 8))
    np.testing.assert_allclose(advantages(R).mean(axis=1), 0.0, atol=1e-12)


def test_baseline_cancels_identical_episodes():
    """If every episode got the same rewards, nothing tells actions apart: the weights are 0."""
    R = np.tile(np.random.default_rng(2).normal(size=(10, 1)), (1, 5))
    np.testing.assert_allclose(advantages(R), 0.0, atol=1e-6)


def test_gradient_equals_the_course_second_form():
    """pg1 writes GPOMDP with a reward baseline as
        (1/N) sum_n sum_t (r_t^n - mean_n r_t^n) * sum_{h<=t} score_h^n.
    Swapping the sums gives our form (G_t - mean_n G_t) * score_t, so both are equal
    (up to the common std rescaling)."""
    rng = np.random.default_rng(3)
    T, N, P = 7, 5, 3
    R, scores = rng.normal(size=(T, N)), rng.normal(size=(T, N, P))
    centred = R - R.mean(axis=1, keepdims=True)
    second_form = np.einsum("tn,tnp->p", centred, np.cumsum(scores, axis=0)) / N
    G = returns_to_go(R)
    scale = (G - G.mean(axis=1, keepdims=True)).std()
    np.testing.assert_allclose(policy_gradient(R, scores) * scale, second_form, atol=1e-6)


def test_adam_climbs_a_concave_quadratic():
    """Maximize f(x) = -||x - x*||^2: gradient 2 (x* - x)."""
    target = np.array([1.0, -2.0, 0.5])
    adam, x = Adam(3, lr=0.05), np.zeros(3)
    for _ in range(1000):
        x = adam.step(x, 2 * (target - x))
    np.testing.assert_allclose(x, target, atol=1e-3)


@pytest.fixture(scope="module")
def short_runs():
    cfg = TrainConfig(n_iters=40, batch=32, eval_every=40, eval_episodes=20, seed=0)
    return [train(EnvConfig(m=75, C=0.5), cfg)[1] for _ in range(2)]


def test_short_training_run_improves_return(short_runs):
    h = short_runs[0]
    assert h[-1]["eval_return"] > 1.5 * h[0]["eval_return"]    # cost at least a third lower
    first = np.mean([row["mean_return"] for row in h[:5]])
    last = np.mean([row["mean_return"] for row in h[-6:-1]])
    assert last > first


def test_training_is_reproducible(short_runs):
    a, b = short_runs
    np.testing.assert_array_equal([r.get("mean_return", np.nan) for r in a],
                                  [r.get("mean_return", np.nan) for r in b])
