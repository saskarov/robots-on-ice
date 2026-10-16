"""Evaluation on a fixed test set: every policy faces the same episodes (same seed)."""
from dataclasses import replace

import numpy as np
import pandas as pd

from src.config import EnvConfig
from src.env import IceRinkEnv

TEST_SEED = 12345       # final evaluation of every policy
VAL_SEED = 777          # monitoring during training (never the test episodes)
REACH_RADIUS = 1.0      # a target counts as reached within 1 m


def evaluate(policy, env_cfg: EnvConfig, n_episodes=100, n_targets=5, seed=TEST_SEED,
             deterministic=True, n_trajectories=0):
    """Run n_episodes in parallel and return metrics (undiscounted, averaged over episodes).

    - return_mean / return_std:  G = sum_t r_t over the whole episode (no discount)
    - dist_cost:    sum_t ||x_t - g||^2        (so return = -(dist_cost + energy_cost))
    - energy_cost:  sum_t C/100 ||F_t||^2
    - time_to_reach_s: first time within 1 m of each target, 20 s if never (mean over targets)
    - frac_reached: share of targets ever reached
    With n_trajectories > 0 it also returns "trajectories": positions (T+1, n, 2) and
    targets (n, K, 2) of the first n episodes (not JSON-serializable: drop before saving).
    """
    cfg = replace(env_cfg, n_targets=n_targets)
    env = IceRinkEnv(cfg, n_envs=n_episodes)
    obs = env.reset(seed)

    spt, N = cfg.steps_per_target, n_episodes
    G, dist_cost, energy_cost = np.zeros(N), np.zeros(N), np.zeros(N)
    reach_step = np.full((N, n_targets), spt)    # spt steps = 20 s = "never reached"
    path = [env.pos[:n_trajectories].copy()]

    for t in range(env.horizon):
        obs, r, _, info = env.step(policy.act(obs, deterministic=deterministic))
        k, j = divmod(t, spt)                    # target index, step within its segment
        G += r
        dist_cost += info["dist"]
        energy_cost += cfg.C / 100.0 * info["energy"]
        first = (info["dist"] < REACH_RADIUS**2) & (reach_step[:, k] == spt)
        reach_step[first, k] = j
        path.append(env.pos[:n_trajectories].copy())

    out = {
        "return_mean": float(G.mean()),
        "return_std": float(G.std()),
        "dist_cost": float(dist_cost.mean()),
        "energy_cost": float(energy_cost.mean()),
        "time_to_reach_s": float(reach_step.mean() * cfg.dt),
        "frac_reached": float((reach_step < spt).mean()),
    }
    if n_trajectories:
        out["trajectories"] = {"pos": np.array(path), "targets": env.targets[:n_trajectories].copy()}
    return out


def trajectories_frame(traj) -> pd.DataFrame:
    """Long-format table of evaluate()'s trajectories, for saving as CSV:
    kind = "pos" (step = time step) or "target" (step = target index), episode, step, x, y."""
    pos, targets = traj["pos"], traj["targets"]
    T, n, _ = pos.shape
    t, ep = np.meshgrid(np.arange(T), np.arange(n), indexing="ij")
    k, ep_g = np.meshgrid(np.arange(targets.shape[1]), np.arange(n), indexing="ij")
    path = pd.DataFrame({"kind": "pos", "episode": ep.ravel(), "step": t.ravel(),
                         "x": pos[..., 0].ravel(), "y": pos[..., 1].ravel()})
    goal = pd.DataFrame({"kind": "target", "episode": ep_g.ravel(), "step": k.ravel(),
                         "x": targets.transpose(1, 0, 2)[..., 0].ravel(),
                         "y": targets.transpose(1, 0, 2)[..., 1].ravel()})
    return pd.concat([path, goal], ignore_index=True)


def trajectories_from_frame(df: pd.DataFrame):
    """Inverse of trajectories_frame: pos (T+1, n, 2), targets (n, K, 2)."""
    def grid(kind, axis_order):
        part = df[df["kind"] == kind]
        arr = part.pivot(index="step", columns="episode", values=["x", "y"])
        xy = np.stack([arr["x"].to_numpy(), arr["y"].to_numpy()], axis=-1)   # (steps, n, 2)
        return xy.transpose(axis_order)
    return grid("pos", (0, 1, 2)), grid("target", (1, 0, 2))
