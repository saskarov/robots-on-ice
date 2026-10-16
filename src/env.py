"""Vectorized ice-rink environment: N independent robots simulated in parallel.

MDP (one robot), finite horizon T = steps_per_target * n_targets:
    state   s = (x - g, v) in R^4      position error w.r.t. the current target g, velocity
    action  a in R^2, in units of 100 N; the env projects it onto the force disc:
                F = f_max * a / max(1, ||a||)          so ||F|| <= f_max always
    reward  r(t) = -( ||x(t) - g||^2 + C/100 * ||F(t)||^2 )   (uses x(t), F(t) before the update)
    dynamics (explicit Euler, dt = 0.1 s, frictionless ice):
                x(t+1) = x(t) + dt * v(t)
                v(t+1) = v(t) + dt * F(t) / m

The target changes every `steps_per_target` steps. The dynamics are deterministic: the only
randomness is the start position and the targets, drawn in `reset` (the initial-state
distribution p0). The next state depends only on (s, a): the process is Markov.
"""
import numpy as np

from src.config import EnvConfig

POS_SCALE = 30.0   # obs = (x - g) / POS_SCALE   [m]
VEL_SCALE = 10.0   # obs = v / VEL_SCALE         [m/s]
OBS_DIM = 4


class IceRinkEnv:
    def __init__(self, cfg: EnvConfig, n_envs: int = 64):
        self.cfg = cfg
        self.n = n_envs
        self.horizon = cfg.steps_per_target * cfg.n_targets

    def reset(self, seed=None) -> np.ndarray:
        """Draw start positions and targets uniformly inside the rink (minus the margin).

        `seed` may be an int or an np.random.Generator. Returns obs of shape (N, 4).
        """
        rng = np.random.default_rng(seed)
        self.pos = self._random_points(rng, (self.n,))                       # (N, 2)
        self.targets = self._random_points(rng, (self.n, self.cfg.n_targets))  # (N, K, 2)
        self.vel = np.zeros((self.n, 2))
        self.t = 0
        return self._obs()

    def _random_points(self, rng, shape) -> np.ndarray:
        hw = self.cfg.half_w - self.cfg.margin
        hh = self.cfg.half_h - self.cfg.margin
        return rng.uniform(low=[-hw, -hh], high=[hw, hh], size=(*shape, 2))

    def current_target(self) -> np.ndarray:
        """g = targets[:, t // steps_per_target], shape (N, 2).

        Clamped to the last target so the final observation (t = horizon) is defined.
        """
        k = min(self.t // self.cfg.steps_per_target, self.cfg.n_targets - 1)
        return self.targets[:, k]

    def project(self, a: np.ndarray) -> np.ndarray:
        """F = f_max * a / max(1, ||a||), row by row.

        Inside the unit disc the action is used as is (F = 100 a); outside, it is
        rescaled onto the circle ||F|| = f_max, keeping its direction.
        """
        norm = np.linalg.norm(a, axis=1, keepdims=True)
        return self.cfg.f_max * a / np.maximum(1.0, norm)

    def _obs(self) -> np.ndarray:
        """[(x - g)/30, v/10]: every feature roughly in [-1, 1], so a single learning
        rate suits all the policy gains."""
        return np.concatenate([(self.pos - self.current_target()) / POS_SCALE,
                               self.vel / VEL_SCALE], axis=1)

    def step(self, a: np.ndarray):
        """Apply actions a (N, 2). Returns obs (N, 4), reward (N,), done (bool), info."""
        F = self.project(a)
        dist = ((self.pos - self.current_target()) ** 2).sum(axis=1)  # ||x(t) - g||^2
        energy = (F ** 2).sum(axis=1)                                 # ||F(t)||^2
        r = -(dist + self.cfg.C / 100.0 * energy)

        dt = self.cfg.dt
        self.pos = self.pos + dt * self.vel                           # uses v(t), not v(t+1)
        self.vel = self.vel + dt * F / self.cfg.m
        self.t += 1

        done = self.t >= self.horizon
        return self._obs(), r, done, {"dist": dist, "energy": energy, "F": F}
