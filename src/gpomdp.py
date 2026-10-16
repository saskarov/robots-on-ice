"""GPOMDP with a time-dependent baseline (finite horizon, no discount), and train().

Objective (finite-horizon MDP, horizon T = 200 steps = one target):
    J(theta) = E[ sum_{t<T} r_t ]
Policy gradient theorem (likelihood-ratio trick):
    grad J = E[ sum_t grad log pi(a_t|s_t) * R(tau) ]                  (REINFORCE)
GPOMDP keeps, for the action at step t, only the rewards from t on (an action cannot change
the rewards before it; those terms have zero mean because E[grad log pi(a|s) | s] = 0):
    grad J ~= (1/N) sum_n sum_t grad log pi(a_t^n|s_t^n) * (G_t^n - b_t),   G_t = sum_{k>=t} r_k
with the time-dependent average baseline b_t = (1/N) sum_n G_t^n. A baseline that does not
depend on the action also has zero mean, so the estimate stays unbiased with lower variance.
This is the course's "GPOMDP with time-dependent baseline" (pg1): its reward baseline
mean_n r_k, summed over k >= t, equals our mean_n G_t.
"""
import numpy as np

from src.config import EnvConfig, TrainConfig
from src.env import IceRinkEnv
from src.evaluate import VAL_SEED, evaluate
from src.optim import Adam
from src.policy import LinearGaussianPolicy


def rollout(env, policy, rng):
    """Run one batch of N episodes with the stochastic policy.

    Returns rewards (T, N) and scores (T, N, P) = grad log pi(a_t|s_t).
    """
    obs = env.reset(rng)
    rewards, scores = [], []
    done = False
    while not done:
        a = policy.sample(obs, rng)
        scores.append(policy.score(obs, a))
        obs, r, done, _ = env.step(a)
        rewards.append(r)
    return np.array(rewards), np.array(scores)


def returns_to_go(R):
    """G_t = r_t + r_{t+1} + ... + r_{T-1} (backward sum). R (T, N) -> G (T, N)."""
    return np.cumsum(R[::-1], axis=0)[::-1]


def advantages(R):
    """Weights of grad log pi(a_t|s_t): (G_t - b_t) / std, shape (T, N).

    b_t = mean over the N episodes of G_t (one baseline per time step). Dividing by one
    positive number (the std of all the weights) does not change the gradient's direction,
    only its size: the costs differ a lot between masses and C values and fall during
    training, and this keeps the step size comparable, so one learning rate works everywhere.
    """
    G = returns_to_go(R)
    A = G - G.mean(axis=1, keepdims=True)
    return A / (A.std() + 1e-8)


def policy_gradient(R, scores):
    """grad J ~= (1/N) sum_n sum_t A_t^n * grad log pi(a_t^n | s_t^n).  -> (P,)"""
    A = advantages(R)
    return np.einsum("tn,tnp->p", A, scores) / A.shape[1]


def train(env_cfg: EnvConfig, cfg: TrainConfig, log_every=0):
    """Train a linear Gaussian policy. Returns (policy, history: list of dicts, one per iteration).

    Each iteration: N episodes -> GPOMDP gradient -> Adam ascent step (log sigma is clamped).
    Every cfg.eval_every iterations it also logs a deterministic (mean-action) evaluation
    on the fixed validation episodes (VAL_SEED, not the test set).
    """
    rng = np.random.default_rng(cfg.seed)
    env = IceRinkEnv(env_cfg, n_envs=cfg.batch)
    policy = LinearGaussianPolicy(init_std=cfg.init_std)
    adam = Adam(policy.n_params, lr=cfg.lr)
    history = []

    for it in range(cfg.n_iters + 1):
        row = {"iter": it}
        if it % cfg.eval_every == 0 or it == cfg.n_iters:
            ev = evaluate(policy, env_cfg, n_episodes=cfg.eval_episodes,
                          n_targets=env_cfg.n_targets, seed=VAL_SEED)
            row["eval_return"] = ev["return_mean"]
        if it == cfg.n_iters:                    # last row: evaluation of the final policy only
            history.append(row)
            break

        R, scores = rollout(env, policy, rng)
        policy.set_flat(adam.step(policy.get_flat(), policy_gradient(R, scores)))

        row.update(mean_return=R.sum(axis=0).mean(), std_x=policy.std[0], std_y=policy.std[1])
        history.append(row)
        if log_every and it % log_every == 0:
            print(f"iter {it:4d}  return {row['mean_return']:12.0f}  sigma {policy.std.round(3)}")
    return policy, history
