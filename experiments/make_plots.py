"""Every figure, built ONLY from files saved in results/ (this script never trains or simulates).

    python experiments/make_plots.py

Also writes results/summary.csv: the grid averaged over seeds, one row per (m, C).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from src.evaluate import trajectories_from_frame
from src.plotting import (INK, ORDINAL, plot_heatmap, plot_learning_curves, plot_trajectories,
                          save)

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
RUNS = RESULTS / "runs"
THOUSANDS = FuncFormatter(lambda v, _: f"{v / 1000:.0f}k")


def run_dir(m, C, seed):
    return RUNS / f"m{m:g}_C{C:g}_s{seed}"


def summary(grid):
    """Mean over seeds per (m, C); return_std_seeds = std of the final return across seeds."""
    return grid.groupby(["m", "C"]).agg(
        return_mean=("return_mean", "mean"), return_std_seeds=("return_mean", "std"),
        dist_cost=("dist_cost", "mean"), energy_cost=("energy_cost", "mean"),
        time_to_reach_s=("time_to_reach_s", "mean"),
        frac_reached=("frac_reached", "mean")).reset_index()


def cost_curves(grid, m, C):
    """(iterations, (n_seeds, n_points) validation cost = -return) from each seed's history.csv."""
    hists = [pd.read_csv(run_dir(m, C, s) / "history.csv").dropna(subset=["eval_return"])
             for s in sorted(grid["seed"].unique())]
    return hists[0]["iter"].to_numpy(), np.array([-h["eval_return"].to_numpy() for h in hists])


def fig_learning_curves(grid, panel_by, curve_by, filename):
    """One panel per value of `panel_by`, one curve per value of `curve_by`."""
    panels, lines = sorted(grid[panel_by].unique()), sorted(grid[curve_by].unique())
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.6), sharey=True)
    for ax, p in zip(axes, panels):
        curves = {}
        for v in lines:
            m, C = (p, v) if panel_by == "m" else (v, p)
            curves["_x"], curves[f"{curve_by} = {v:g}"] = cost_curves(grid, m, C)
        plot_learning_curves(ax, curves, colors=ORDINAL[len(lines)],
                             ylabel="validation cost (−return, log scale)" if ax is axes[0] else "")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(THOUSANDS)
        ax.yaxis.set_minor_formatter(THOUSANDS)
        ax.set_xlabel("iteration")
        ax.set_title(f"{panel_by} = {p:g}" + (" kg" if panel_by == "m" else ""), fontsize=10)
        if ax is not axes[0]:
            ax.get_legend().remove()
    fig.suptitle("Learning curves, mean ± std over 5 seeds (lower is better)", fontsize=11, color=INK)
    save(fig, FIGURES / filename)


def fig_heatmaps(s):
    def table(col, scale=1.0):
        return s.pivot(index="m", columns="C", values=col) * scale

    total, dist = table("return_mean", -1e-3), table("dist_cost", 1e-3)
    shared = {"vmin": min(total.min().min(), dist.min().min()),
              "vmax": max(total.max().max(), dist.max().max())}
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13, 3.4))
    plot_heatmap(a1, total, fmt="{:.0f}k", title="Total cost (−return)", **shared)
    plot_heatmap(a2, dist, fmt="{:.0f}k", title="Distance cost only  Σ‖x−g‖²", **shared)
    plot_heatmap(a3, table("time_to_reach_s"), fmt="{:.1f} s",
                 title="Time to get within 1 m of a target")
    fig.suptitle("Learned policy, test set (100 episodes × 5 targets), mean over 5 seeds",
                 fontsize=11, color=INK)
    save(fig, FIGURES / "heatmaps.png")


def fig_trajectories(grid):
    settings = [(50.0, 0.0), (75.0, 0.5), (100.0, 1.0)]
    fig, axes = plt.subplots(len(settings), 1, figsize=(6.5, 3.5 * len(settings)))
    for ax, (m, C) in zip(axes, settings):
        pos, targets = trajectories_from_frame(pd.read_csv(run_dir(m, C, 0) / "trajectories.csv"))
        ret = grid[(grid["m"] == m) & (grid["C"] == C) & (grid["seed"] == 0)]["return_mean"].item()
        plot_trajectories(ax, pos, targets, title=f"m = {m:g} kg, C = {C:g} (seed 0)   return {ret:,.0f}")
    save(fig, FIGURES / "trajectories.png")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    grid = pd.read_csv(RESULTS / "grid.csv")
    s = summary(grid)
    s.to_csv(RESULTS / "summary.csv", index=False)
    fig_learning_curves(grid, panel_by="m", curve_by="C", filename="learning_curves_by_C.png")
    fig_learning_curves(grid, panel_by="C", curve_by="m", filename="learning_curves_by_m.png")
    fig_heatmaps(s)
    fig_trajectories(grid)
    print(f"figures written to {FIGURES.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
