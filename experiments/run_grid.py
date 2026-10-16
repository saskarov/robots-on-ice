"""The main experiment: m in {50, 75, 100} x C in {0, 0.1, 0.5, 1} x 5 seeds = 60 runs,
all with the frozen TrainConfig defaults.

    python experiments/run_grid.py --workers 4

Resumable: runs whose folder already has every output file are skipped (unless --force). At the end it
collects every eval.json into results/grid.csv (one row per run).
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import EnvConfig, TrainConfig
from train_single import is_done, run_and_save, run_name

MASSES = [50.0, 75.0, 100.0]
CS = [0.0, 0.1, 0.5, 1.0]
SEEDS = [0, 1, 2, 3, 4]
RUNS = ROOT / "results" / "runs"


def grid_jobs():
    return [(EnvConfig(m=m, C=C), TrainConfig(seed=s)) for m in MASSES for C in CS for s in SEEDS]


def run_job(job):
    env_cfg, train_cfg = job
    run_and_save(env_cfg, train_cfg, RUNS / run_name(env_cfg, train_cfg))
    print(f"done {run_name(env_cfg, train_cfg)}", flush=True)


def collect():
    rows = []
    for env_cfg, train_cfg in grid_jobs():
        result = json.loads((RUNS / run_name(env_cfg, train_cfg) / "eval.json").read_text())
        rows.append({"m": env_cfg.m, "C": env_cfg.C, "seed": train_cfg.seed, **result})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--force", action="store_true", help="retrain every run, even finished ones")
    args = p.parse_args()

    todo = [j for j in grid_jobs() if args.force or not is_done(RUNS / run_name(*j))]
    print(f"{len(grid_jobs()) - len(todo)} runs already done, {len(todo)} to go")
    with Pool(args.workers) as pool:
        pool.map(run_job, todo, chunksize=1)

    df = collect()
    df.to_csv(ROOT / "results" / "grid.csv", index=False)
    print(df.groupby(["m", "C"])["return_mean"].agg(["mean", "std"]).round(0).to_string())


if __name__ == "__main__":
    main()
