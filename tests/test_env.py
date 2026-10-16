"""Tests for IceRinkEnv: physics against closed forms, force limit, reward, seeding, target switch."""
import numpy as np

from src.config import EnvConfig
from src.env import IceRinkEnv


def make_env(n_envs=8, seed=0, **cfg_kwargs):
    env = IceRinkEnv(EnvConfig(**cfg_kwargs), n_envs=n_envs)
    env.reset(seed)
    return env


def test_zero_force_from_rest_stays_put():
    env = make_env()
    x0 = env.pos.copy()
    for _ in range(50):
        env.step(np.zeros((env.n, 2)))
    np.testing.assert_allclose(env.pos, x0)
    np.testing.assert_allclose(env.vel, 0.0)


def test_zero_force_moves_in_straight_line():
    """No force, v0 != 0: x_n = x_0 + n * dt * v0 (no friction on ice)."""
    env = make_env()
    env.vel = np.random.default_rng(1).normal(size=(env.n, 2))
    x0, v0 = env.pos.copy(), env.vel.copy()
    n = 30
    for _ in range(n):
        env.step(np.zeros((env.n, 2)))
    np.testing.assert_allclose(env.pos, x0 + n * env.cfg.dt * v0)
    np.testing.assert_allclose(env.vel, v0)


def test_constant_force_from_rest_matches_closed_form():
    """v_n = n*dt*F/m and x_n = x_0 + dt^2 * (F/m) * n(n-1)/2."""
    env = make_env(m=60.0)
    a = np.tile([0.5, -0.3], (env.n, 1))           # ||a|| < 1, so F = 100 a = (50, -30) N
    F = 100.0 * a
    x0 = env.pos.copy()
    dt, m, n = env.cfg.dt, env.cfg.m, 40
    for _ in range(n):
        env.step(a)
    np.testing.assert_allclose(env.vel, n * dt * F / m)
    np.testing.assert_allclose(env.pos, x0 + dt**2 * (F / m) * n * (n - 1) / 2)


def test_force_never_exceeds_limit():
    env = make_env(n_envs=1000)
    a = np.random.default_rng(2).normal(scale=50.0, size=(env.n, 2))   # huge actions
    _, _, _, info = env.step(a)
    assert np.all(np.linalg.norm(info["F"], axis=1) <= 100.0 + 1e-9)


def test_projection_keeps_small_actions_and_direction():
    env = make_env()
    small = np.array([[0.3, 0.4], [0.0, -1.0]])    # ||a|| <= 1: unchanged, F = 100 a
    np.testing.assert_allclose(env.project(small), 100.0 * small)
    big = np.array([[3.0, 4.0]])                   # ||a|| = 5: rescaled onto ||F|| = 100
    np.testing.assert_allclose(env.project(big), [[60.0, 80.0]])


def test_reward_is_zero_at_target_with_no_force():
    env = make_env()
    env.pos = env.current_target().copy()
    _, r, _, _ = env.step(np.zeros((env.n, 2)))
    np.testing.assert_allclose(r, 0.0)


def test_reward_uses_state_before_update():
    """r = -(||x(t) - g||^2 + C/100 * ||F(t)||^2) with x(t), not x(t+1)."""
    env = make_env(C=0.5)
    env.vel = np.ones((env.n, 2))                  # the robot moves during the step
    a = np.tile([0.6, 0.8], (env.n, 1))            # F = (60, 80), ||F||^2 = 10000
    expected = -(((env.pos - env.current_target()) ** 2).sum(1) + 0.5 / 100.0 * 10000.0)
    _, r, _, _ = env.step(a)
    np.testing.assert_allclose(r, expected)


def test_same_seed_gives_same_rollout():
    def rollout(seed):
        env = IceRinkEnv(EnvConfig(), n_envs=16)
        obs_list, r_list = [env.reset(seed)], []
        act_rng = np.random.default_rng(99)        # same action sequence in both runs
        done = False
        while not done:
            obs, r, done, _ = env.step(act_rng.normal(size=(env.n, 2)))
            obs_list.append(obs)
            r_list.append(r)
        return np.array(obs_list), np.array(r_list)

    obs_a, r_a = rollout(7)
    obs_b, r_b = rollout(7)
    np.testing.assert_array_equal(obs_a, obs_b)
    np.testing.assert_array_equal(r_a, r_b)
    obs_c, _ = rollout(8)
    assert not np.array_equal(obs_a[0], obs_c[0])  # a different seed gives different episodes


def test_starts_and_targets_inside_rink_with_margin():
    env = make_env(n_envs=2000, n_targets=5)
    for pts in (env.pos, env.targets.reshape(-1, 2)):
        assert np.all(np.abs(pts[:, 0]) <= 30.0 - 2.0)
        assert np.all(np.abs(pts[:, 1]) <= 15.0 - 2.0)


def test_observation_is_normalized_error_and_velocity():
    env = make_env()
    env.vel = np.full((env.n, 2), 5.0)
    obs = env._obs()
    np.testing.assert_allclose(obs[:, :2], (env.pos - env.targets[:, 0]) / 30.0)
    np.testing.assert_allclose(obs[:, 2:], 0.5)


def test_target_switches_at_step_200_and_episode_ends():
    env = make_env(n_targets=2)
    a = np.zeros((env.n, 2))
    for _ in range(199):
        _, _, done, _ = env.step(a)
        assert not done
    np.testing.assert_array_equal(env.current_target(), env.targets[:, 0])   # t = 199
    obs, _, done, _ = env.step(a)                                            # t = 200
    np.testing.assert_array_equal(env.current_target(), env.targets[:, 1])
    np.testing.assert_allclose(obs[:, :2], (env.pos - env.targets[:, 1]) / 30.0)
    assert not done
    for _ in range(200):
        _, _, done, _ = env.step(a)
    assert done and env.t == 400

