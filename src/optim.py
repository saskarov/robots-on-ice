"""Adam for gradient ASCENT on a flat parameter vector."""
import numpy as np


class Adam:
    """theta <- theta + lr * m_hat / (sqrt(v_hat) + eps), with
        m = b1 m + (1 - b1) g        (running mean of the gradient: momentum)
        v = b2 v + (1 - b2) g^2      (running mean of its square: per-parameter scale)
    and bias corrections m_hat = m / (1 - b1^t), v_hat = v / (1 - b2^t).
    Each parameter moves by about lr per step whatever its gradient scale, which
    suits K, b and log sigma having very different gradient magnitudes.
    (The course mentions Adam as a common practical choice for the update in pg1.)
    """

    def __init__(self, n_params, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = np.zeros(n_params)
        self.v = np.zeros(n_params)
        self.t = 0

    def step(self, theta, g):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g**2
        m_hat = self.m / (1 - self.b1**self.t)
        v_hat = self.v / (1 - self.b2**self.t)
        return theta + self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
