"""Train one policy and save config.json, history.csv, policy.npz, eval.json.

    python experiments/train_single.py --m 75 --C 0.5 --seed 0
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import EnvConfig, TrainConfig, save_config
from src.evaluate import evaluate, trajectories_frame
from src.gpomdp import train


def run_name(env_cfg, train_cfg):
    return f"m{env_cfg.m:g}_C{env_cfg.C:g}_s{train_cfg.seed}"


OUTPUT_FILES = ["config.json", "history.csv", "policy.npz", "trajectories.csv", "eval.json"]


def is_done(run_dir):
    return all((Path(run_dir) / f).exists() for f in OUTPUT_FILES)


def run_and_save(env_cfg, train_cfg, run_dir, log_every=0):
    """Train, then evaluate the mean action on the fixed test set (100 episodes x 5 targets).

    eval.json is written last, so a run interrupted halfway is redone by run_grid.py.
    trajectories.csv holds the paths of the first 3 test episodes (for the figures).
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(run_dir / "config.json", env_cfg, train_cfg)
    policy, history = train(env_cfg, train_cfg, log_every=log_every)
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    policy.save(run_dir / "policy.npz")
    result = evaluate(policy, env_cfg, n_trajectories=3)
    trajectories_frame(result.pop("trajectories")).to_csv(run_dir / "trajectories.csv", index=False)
    (run_dir / "eval.json").write_text(json.dumps(result, indent=2))
    return policy, history, result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=float, default=75.0)
    p.add_argument("--C", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-iters", type=int, default=TrainConfig.n_iters)
    p.add_argument("--batch", type=int, default=TrainConfig.batch)
    p.add_argument("--lr", type=float, default=TrainConfig.lr)
    p.add_argument("--init-std", type=float, default=TrainConfig.init_std)
    p.add_argument("--out", default=None, help="run folder (default results/runs/m.._C.._s..)")
    args = p.parse_args()

    env_cfg = EnvConfig(m=args.m, C=args.C)
    train_cfg = TrainConfig(n_iters=args.n_iters, batch=args.batch, lr=args.lr,
                            init_std=args.init_std, seed=args.seed)
    run_dir = Path(args.out) if args.out else ROOT / "results" / "runs" / run_name(env_cfg, train_cfg)
    _, _, result = run_and_save(env_cfg, train_cfg, run_dir, log_every=25)
    print(f"\ntest return {result['return_mean']:,.0f}   time to reach a target "
          f"{result['time_to_reach_s']:.1f} s   saved to {run_dir}")


if __name__ == "__main__":
    main()
