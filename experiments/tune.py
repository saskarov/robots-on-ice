"""Hyperparameter tuning at m = 75, C = 0.5, scored on the VALIDATION episodes (never the test set).

    python experiments/tune.py --stage lr
    python experiments/tune.py --stage other --lr <best lr from stage lr>

Stage "lr":    lr in {3e-3, 1e-2, 3e-2, 1e-1}, everything else at the design's starting values (BASE).
Stage "other": at the chosen lr, change one thing at a time: batch in {32, 128}, init_std in {0.25, 1.0, 2.0}.
Each config x 3 seeds, 1000 iterations. Score: mean return of the mean action on
100 validation episodes (1 target). Writes results/tuning/<config>_s<seed>.csv
(learning curve) and results/tuning_<stage>.csv (one row per run).
"""
import argparse
import sys
from dataclasses import asdict, replace
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import EnvConfig, TrainConfig
from src.evaluate import VAL_SEED, evaluate
from src.gpomdp import train

ENV = EnvConfig(m=75.0, C=0.5)
# The pre-tuning starting point from the design (lr 1e-2, batch 64, init_std 0.5). Written out
# explicitly because the TrainConfig defaults now hold the frozen (post-tuning) values.
BASE = TrainConfig(n_iters=1000, lr=1e-2, batch=64, init_std=0.5)
SEEDS = [0, 1, 2]


def config_name(cfg):
    return f"lr{cfg.lr:g}_b{cfg.batch}_std{cfg.init_std:g}"


def stage_configs(stage, lr):
    if stage == "lr":
        return [replace(BASE, lr=x) for x in (3e-3, 1e-2, 3e-2, 1e-1)]
    base = replace(BASE, lr=lr)
    return [base] + [replace(base, batch=x) for x in (32, 128)] + \
           [replace(base, init_std=x) for x in (0.25, 1.0, 2.0)]


def run_one(cfg):
    policy, history = train(ENV, cfg)
    pd.DataFrame(history).to_csv(ROOT / "results" / "tuning" / f"{config_name(cfg)}_s{cfg.seed}.csv",
                                 index=False)
    val = evaluate(policy, ENV, n_episodes=100, n_targets=1, seed=VAL_SEED)
    return {"config": config_name(cfg), **asdict(cfg), "val_return": val["return_mean"],
            "final_std": float(policy.std.mean())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["lr", "other"], required=True)
    p.add_argument("--lr", type=float, default=BASE.lr)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    (ROOT / "results" / "tuning").mkdir(parents=True, exist_ok=True)
    jobs = [replace(c, seed=s) for c in stage_configs(args.stage, args.lr) for s in SEEDS]
    with Pool(args.workers) as pool:
        rows = pool.map(run_one, jobs)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / f"tuning_{args.stage}.csv", index=False)

    summary = df.groupby("config", sort=False).agg(
        val_mean=("val_return", "mean"), val_std=("val_return", "std"),
        final_std=("final_std", "mean"))
    print(summary.round(2).to_string())


if __name__ == "__main__":
    main()
