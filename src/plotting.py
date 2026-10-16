"""Figures. Every function draws on a given matplotlib Axes, so callers lay out the grid."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

# Ordered values (m, C) get light-to-dark steps of one hue instead of unrelated colours.
ORDINAL = {3: ["#86b6ef", "#2a78d6", "#0d366b"],
           4: ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]}
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

# Categorical slots in fixed order (blue, orange, aqua, yellow); the first three stay
# distinguishable for colour-blind readers even in scatter-like plots.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, RINK, GRID = "#0b0b0b", "#898781", "#c3c2b7", "#e1e0d9"


def style_axes(ax):
    """Recessive axes: no top/right spines, light grid, muted ticks."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RINK)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(which="both", colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def plot_learning_curves(ax, curves, ylabel="", colors=SERIES):
    """curves: {label: array (n_seeds, n_points) of y} with a shared x in curves["_x"].

    Draws the mean over seeds with a +-1 std band.
    """
    x = curves["_x"]
    for i, (label, ys) in enumerate((k, v) for k, v in curves.items() if k != "_x"):
        mean, std = ys.mean(axis=0), ys.std(axis=0)
        ax.plot(x, mean, color=colors[i], linewidth=2, label=label)
        ax.fill_between(x, mean - std, mean + std, color=colors[i], alpha=0.15, linewidth=0)
    ax.set_ylabel(ylabel)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    return ax


def plot_trajectories(ax, pos, targets, half_w=30.0, half_h=15.0, title=""):
    """x-y paths of up to 3 episodes on the rink.

    pos (T+1, n, 2), targets (n, K, 2). Start = open circle, targets = ✕ numbered in order.
    """
    ax.add_patch(Rectangle((-half_w, -half_h), 2 * half_w, 2 * half_h,
                           fill=False, edgecolor=RINK, linewidth=1.5))
    for i in range(min(pos.shape[1], len(SERIES))):
        c = SERIES[i]
        ax.plot(pos[:, i, 0], pos[:, i, 1], color=c, linewidth=1.5)
        ax.plot(*pos[0, i], "o", mfc="white", mec=c, mew=2, ms=8)
        for k, g in enumerate(targets[i]):
            ax.plot(*g, "x", color=c, mew=2.5, ms=10)
            ax.annotate(str(k + 1), g, xytext=(5, 5), textcoords="offset points",
                        fontsize=8, color=INK)
    pad = 2.0
    ax.set_xlim(-half_w - pad, half_w + pad)
    ax.set_ylim(-half_h - pad, half_h + pad)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10, color=INK)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


def plot_heatmap(ax, table, fmt="{:.0f}", title="", xlabel="C", ylabel="m [kg]", vmin=None, vmax=None):
    """table: DataFrame (rows = m, columns = C). Darker = larger; every cell is labelled.

    Pass the same vmin/vmax to panels that are meant to be compared side by side.
    """
    values = table.to_numpy(dtype=float)
    lo = np.nanmin(values) if vmin is None else vmin
    hi = np.nanmax(values) if vmax is None else vmax
    ax.imshow(values, cmap=BLUES, aspect="auto", vmin=lo, vmax=hi)
    for (i, j), v in np.ndenumerate(values):
        dark = (v - lo) / (hi - lo + 1e-12) > 0.55
        ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=9,
                color="white" if dark else INK)
    ax.set_xticks(range(len(table.columns)), [f"{c:g}" for c in table.columns])
    ax.set_yticks(range(len(table.index)), [f"{r:g}" for r in table.index])
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.set_title(title, fontsize=10, color=INK)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


def save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
