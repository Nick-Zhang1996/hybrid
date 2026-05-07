"""Create IEEE-style IPM benchmark plots."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

CARS = np.array([2, 3, 4, 5, 6, 7, 8], dtype=float)
OUTPUT_DIR = Path(__file__).resolve().parent

CONFIGS = [
    {
        "label": "RD3G",
        "color": "#222222",
        "marker": "o",
        "linestyle": "--",
        "runtime_mean": np.array(
            [18.03941011428833, 36.18350028991699, 119.83912467956543,
             196.03961944580078, 347.52816677093506, 504.63954925537115,
             772.9371523857117]
        ),
        "runtime_var": np.array(
            [0.24843177743588854, 0.8780258517435869, 1.6947821358804505,
             3.967572886913467, 13.18215077847364, 23.356799934903023,
             53.30265708290528]
        ),
        "conv_mean": np.array([0.85, 0.85, 0.53, 0.59, 0.25, 0.26, 0.27]),
        "optimal_mean": np.array([0.84, 0.85, 0.50, 0.58, 0.23, 0.22, 0.25]),
    },
    {
        "label": "IPM",
        "color": "#0055AA",
        "marker": "s",
        "linestyle": "-",
        "runtime_mean": np.array(
            [7.761335372924805, 14.16980266571045, 40.58115243911743,
             84.0429162979126, 71.83857679367065, 95.17242431640625,
             135.7657790184021]
        ),
        "runtime_var": np.array(
            [0.003562006717174882, 0.01849313556069774, 0.2779077248882174,
             1.113932508520088, 0.6698600898017221, 1.1692867704444778,
             1.9990461818433403]
        ),
        "conv_mean": np.array([0.88, 0.89, 0.61, 0.52, 0.53, 0.50, 0.30]),
        "optimal_mean": np.array([0.76, 0.89, 0.61, 0.52, 0.53, 0.50, 0.30]),
    },
    {
        "label": "IPM + VNE",
        "color": "#D55E00",
        "marker": "^",
        "linestyle": "-.",
        "runtime_mean": np.array(
            [7.288565635681152, 13.858742713928223, 30.7962965965271,
             57.02540636062622, 72.94968605041504, 100.255868434906,
             178.30322742462158]
        ),
        "runtime_var": np.array(
            [0.004494280009816976, 0.015744796825833865, 0.14020147304728994,
             0.48128142525899303, 0.7938164180151489, 1.2777843095501509,
             3.0549603715449165]
        ),
        "conv_mean": np.array([0.88, 0.89, 0.61, 0.52, 0.52, 0.51, 0.30]),
        "optimal_mean": np.array([0.76, 0.89, 0.61, 0.52, 0.52, 0.51, 0.30]),
    },
    {
        "label": "IPM + VNE + Precond.",
        "color": "#009E73",
        "marker": "D",
        "linestyle": ":",
        "runtime_mean": np.array(
            [3.7877464294433594, 5.803897380828857, 12.848892211914062,
             19.959585666656494, 27.584717273712158, 37.57180690765381,
             62.17010498046875]
        ),
        "runtime_var": np.array(
            [0.001305362991297443, 0.002761745883725553, 0.025913257188358324,
             0.053253872326030204, 0.11183415707784548, 0.18337279271597706,
             0.4708664846496959]
        ),
        "conv_mean": np.array([0.88, 0.89, 0.61, 0.52, 0.52, 0.51, 0.30]),
        "optimal_mean": np.array([0.76, 0.89, 0.61, 0.52, 0.52, 0.51, 0.30]),
    },
    {
        "label": "Robust IPM",
        "color": "#CC79A7",
        "marker": "v",
        "linestyle": "--",
        "runtime_mean": np.array(
            [7.753942012786865, 11.276869773864746, 23.011724948883057,
             30.463788509368896, 49.09109592437744, 64.56922769546509,
             80.32926082611084]
        ),
        "runtime_var": np.array(
            [0.021188990485705977, 0.02945152343929749, 0.05760140417584694,
             0.10225845669526164, 0.15530631785393328, 0.21311291747276187,
             0.33595816528556954]
        ),
        "conv_mean": np.array([0.83, 0.81, 0.53, 0.56, 0.33, 0.29, 0.37]),
        "optimal_mean": np.array([0.83, 0.81, 0.53, 0.56, 0.33, 0.29, 0.37]),
    },
]


def set_ieee_style() -> None:
    """Apply a compact IEEE-friendly Matplotlib style."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "figure.figsize": (3.5, 2.6),
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "lines.linewidth": 1.5,
            "lines.markersize": 5,
            "axes.linewidth": 0.8,
            "text.usetex": False,
        }
    )


def style_axes(ax: plt.Axes, ylabel: str, percent_axis: bool = False) -> None:
    ax.set_xlabel("Number of Cars ($N$)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(CARS)
    ax.set_xlim(CARS.min() - 0.15, CARS.max() + 0.15)
    ax.grid(True, which="major", linestyle=":", linewidth=0.7, alpha=0.7)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.25)
    if percent_axis:
        ax.set_ylim(0.0, 1.0)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.legend(loc="best", frameon=True, fancybox=False, edgecolor="black", framealpha=1.0)


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTPUT_DIR / f"{stem}.png", bbox_inches="tight")


def show_figure(fig: plt.Figure) -> None:
    plt.show()
    plt.close(fig)


def plot_runtime() -> None:
    fig, ax = plt.subplots(constrained_layout=True)
    for config in CONFIGS:
        runtime_std = np.sqrt(np.clip(config["runtime_var"], a_min=0.0, a_max=None))
        ax.errorbar(
            CARS,
            config["runtime_mean"],
            yerr=runtime_std,
            label=config["label"],
            color=config["color"],
            linestyle=config["linestyle"],
            marker=config["marker"],
            markerfacecolor="white" if config["label"] == "RD3G" else config["color"],
            markeredgecolor=config["color"],
            markeredgewidth=1.0,
            capsize=2.5,
            elinewidth=0.9,
        )

    style_axes(ax, "Runtime (ms)")
    save_figure(fig, "ipm_benchmark_runtime")
    show_figure(fig)


def plot_rate(metric_key: str, ylabel: str, stem: str) -> None:
    fig, ax = plt.subplots(constrained_layout=True)
    for config in CONFIGS:
        ax.plot(
            CARS,
            config[metric_key],
            label=config["label"],
            color=config["color"],
            linestyle=config["linestyle"],
            marker=config["marker"],
            markerfacecolor="white" if config["label"] == "RD3G" else config["color"],
            markeredgecolor=config["color"],
            markeredgewidth=1.0,
        )

    style_axes(ax, ylabel, percent_axis=True)
    save_figure(fig, stem)
    show_figure(fig)


def main() -> None:
    set_ieee_style()
    plot_runtime()
    plot_rate("conv_mean", "Convergence Rate", "ipm_benchmark_convergence")
    plot_rate("optimal_mean", "Optimal Rate", "ipm_benchmark_optimal")


if __name__ == "__main__":
    main()
