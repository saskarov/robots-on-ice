"""Linear Gaussian policy  a ~ N(mu(s), diag(sigma^2)),  mu(s) = K s + b.

Parameters theta = (K (2, 4), b (2,), log_sigma (2,)): 12 numbers.
Why this policy (the Gaussian policy of the course, pg1):
  - the mean is linear in the state, like a spring-damper controller: with K_ij < 0 it pushes
    towards the target (position error) and against the velocity (damping). Few parameters,
    each with a physical meaning;
  - the Gaussian makes the policy stochastic, so it explores, and its score function
    grad log pi(a|s) has a closed form (written below);
  - sigma is learned too (the course's "learnable covariance" generalization), through
    log sigma, which keeps sigma > 0 without constraints. It is clamped to [0.05, 2] so
    exploration never vanishes or explodes.
"""
import numpy as np

from src.env import OBS_DIM

LOG_STD_MIN, LOG_STD_MAX = np.log(0.05), np.log(2.0)


def pack(K, b, log_std):
    """Flatten (K, b, log_std) into one vector [K row by row, b, log_std].

    Works with leading batch dims too: K (..., 2, d), b (..., 2), log_std (..., 2) -> (..., 2d + 4).
    """
    return np.concatenate([K.reshape(*K.shape[:-2], -1), b, log_std], axis=-1)


class LinearGaussianPolicy:
    def __init__(self, obs_dim=OBS_DIM, init_std=2.0):
        self.d = obs_dim
        self.K = np.zeros((2, obs_dim))          # start with "no feedback": mean action 0
        self.b = np.zeros(2)
        self.log_std = np.full(2, np.log(init_std))

    @property
    def std(self):
        return np.exp(self.log_std)

    @property
    def n_params(self):
        return 2 * self.d + 4

    # ---- acting -------------------------------------------------------------
    def mean(self, obs):
        """mu = K s + b, row by row: obs (N, d) -> (N, 2)."""
        return obs @ self.K.T + self.b

    def sample(self, obs, rng):
        """a = mu + sigma * eps,  eps ~ N(0, I)."""
        return self.mean(obs) + self.std * rng.standard_normal((obs.shape[0], 2))

    def act(self, obs, deterministic=True, rng=None):
        """Mean action (used in evaluation), or a sample when deterministic=False (needs rng)."""
        return self.mean(obs) if deterministic else self.sample(obs, rng)

    # ---- log-likelihood and score ------------------------------------------
    def log_prob(self, obs, a):
        """log pi(a|s) = sum_i [ -z_i^2 / 2 - log sigma_i - log(2 pi) / 2 ],  z = (a - mu) / sigma.  -> (N,)"""
        z = (a - self.mean(obs)) / self.std
        return (-0.5 * z**2 - self.log_std - 0.5 * np.log(2 * np.pi)).sum(axis=1)

    def grad_log_prob(self, obs, a):
        """Score function, per sample. With z = (a - mu) / sigma:

            d log pi / d mu_i        = (a_i - mu_i) / sigma_i^2 = z_i / sigma_i   (course: (a - mu)/sigma^2 * grad mu)
            d log pi / d K_ij        = (z_i / sigma_i) * s_j      (chain rule, mu_i = sum_j K_ij s_j + b_i)
            d log pi / d b_i         =  z_i / sigma_i
            d log pi / d log sigma_i =  z_i^2 - 1
        Returns dK (N, 2, d), db (N, 2), dlog_std (N, 2).
        """
        std = self.std
        z = (a - self.mean(obs)) / std
        g_mu = z / std
        dK = g_mu[:, :, None] * obs[:, None, :]
        return dK, g_mu, z**2 - 1.0

    def score(self, obs, a):
        """grad_theta log pi(a|s) as flat vectors, in the order of get_flat(): (N, n_params)."""
        return pack(*self.grad_log_prob(obs, a))

    # ---- parameters as one vector (what the optimizer sees) -----------------
    def get_flat(self):
        return pack(self.K, self.b, self.log_std)

    def set_flat(self, theta):
        """Inverse of get_flat; log sigma is clamped to [log 0.05, log 2]."""
        n_K = 2 * self.d
        self.K = theta[:n_K].reshape(2, self.d).copy()
        self.b = theta[n_K:n_K + 2].copy()
        self.log_std = np.clip(theta[n_K + 2:], LOG_STD_MIN, LOG_STD_MAX)

    def save(self, path):
        np.savez(path, K=self.K, b=self.b, log_std=self.log_std)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        policy = cls(obs_dim=data["K"].shape[1])
        policy.K, policy.b, policy.log_std = data["K"], data["b"], data["log_std"]
        return policy
