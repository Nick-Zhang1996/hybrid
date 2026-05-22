"""Create IEEE-style IPM benchmark plots from a saved pickle file."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = ROOT_DIR / "outputs" / "merge_ipm_benchmark.pkl"
OUTPUT_DIR = ROOT_DIR / "outputs" / "benchmarks"

STYLE_BY_LABEL = {
    "RD3G CasADi": {
        "color": "#222222",
        "marker": "o",
        "linestyle": "--",
    },
    "IPM (no abs split)": {
        "color": "#CA676A",
        "marker": "s",
        "linestyle": "-.",
    },
    "IPM (abs split)": {
        "color": "#8565C5",
        "marker": "D",
        "linestyle": "--",
    },
}

FALLBACK_STYLES = [
    {"color": "#222222", "marker": "o", "linestyle": "--"},
    {"color": "#CA676A", "marker": "s", "linestyle": "-."},
    {"color": "#8565C5", "marker": "D", "linestyle": "--"},
]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Pickle file created by scripts/run_merge_ipm_benchmarks.py.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Prefix for the generated PNG files. Defaults to the pickle stem.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figures interactively after saving them.",
    )
    return parser.parse_args()


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


def style_axes(ax: plt.Axes, cars: np.ndarray, ylabel: str, percent_axis: bool = False) -> None:
    """Apply the same axis styling as plot_ipm_benchmark.py."""
    ax.set_xlabel("Number of Cars ($N$)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(cars)
    ax.set_xlim(cars.min() - 0.15, cars.max() + 0.15)
    ax.grid(True, which="major", linestyle=":", linewidth=0.7, alpha=0.7)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.25)
    if percent_axis:
        ax.set_ylim(0.0, 1.0)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.legend(loc="best", frameon=True, fancybox=False, edgecolor="black", framealpha=1.0)


def save_figure(fig: plt.Figure, stem: str) -> None:
    """Save one figure to the benchmark output directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{stem}.png", bbox_inches="tight")


def finish_figure(fig: plt.Figure, show: bool) -> None:
    """Show or close a figure after saving it."""
    if show:
        plt.show()
    plt.close(fig)


def load_benchmark_data(input_path: Path) -> tuple[np.ndarray, list[dict]]:
    """Load the saved benchmark arrays and add plot styling."""
    with input_path.open("rb") as file:
        payload = pickle.load(file)

    cars = np.asarray(payload["cars"], dtype=float)
    configs = []
    for index, raw_config in enumerate(payload["configs"]):
        config = dict(raw_config)
        for key in ("runtime_mean", "runtime_var", "conv_mean", "optimal_mean"):
            config[key] = np.asarray(config[key], dtype=float)
        config["optimality_ratio"] = np.divide(
            config["optimal_mean"],
            config["conv_mean"],
            out=np.zeros_like(config["optimal_mean"], dtype=float),
            where=config["conv_mean"] > 0.0,
        )

        style = STYLE_BY_LABEL.get(
            config["label"],
            FALLBACK_STYLES[index % len(FALLBACK_STYLES)],
        )
        config.update(style)
        configs.append(config)

    return cars, configs


def plot_runtime(cars: np.ndarray, configs: list[dict], stem: str, show: bool) -> None:
    """Plot mean runtime with error bars."""
    fig, ax = plt.subplots(constrained_layout=True)
    for config in configs:
        runtime_std = np.sqrt(np.clip(config["runtime_var"], a_min=0.0, a_max=None))
        markerfacecolor = "white" if config["label"].startswith("RD3G") else config["color"]
        ax.errorbar(
            cars,
            config["runtime_mean"],
            yerr=runtime_std,
            label=config["label"],
            color=config["color"],
            linestyle=config["linestyle"],
            marker=config["marker"],
            markerfacecolor=markerfacecolor,
            markeredgecolor=config["color"],
            markeredgewidth=1.0,
            capsize=2.5,
            elinewidth=0.9,
        )

    style_axes(ax, cars, "Runtime (ms)")
    save_figure(fig, f"{stem}_runtime")
    finish_figure(fig, show)


def plot_rate(
    cars: np.ndarray,
    configs: list[dict],
    metric_key: str,
    ylabel: str,
    stem: str,
    show: bool,
) -> None:
    """Plot a rate curve such as convergence or optimality."""
    fig, ax = plt.subplots(constrained_layout=True)
    for config in configs:
        markerfacecolor = "white" if config["label"].startswith("RD3G") else config["color"]
        ax.plot(
            cars,
            config[metric_key],
            label=config["label"],
            color=config["color"],
            linestyle=config["linestyle"],
            marker=config["marker"],
            markerfacecolor=markerfacecolor,
            markeredgecolor=config["color"],
            markeredgewidth=1.0,
        )

    style_axes(ax, cars, ylabel, percent_axis=True)
    save_figure(fig, stem)
    finish_figure(fig, show)


def main() -> None:
    """Load benchmark results from pickle and generate the plots."""
    args = parse_args()
    input_path = args.input.resolve()
    output_prefix = args.output_prefix or input_path.stem

    set_ieee_style()
    cars, configs = load_benchmark_data(input_path)
    plot_runtime(cars, configs, output_prefix, args.show)
    plot_rate(
        cars,
        configs,
        "conv_mean",
        "Convergence Rate",
        f"{output_prefix}_convergence",
        args.show,
    )
    plot_rate(
        cars,
        configs,
        "optimality_ratio",
        "Optimality Ratio",
        f"{output_prefix}_optimal",
        args.show,
    )


if __name__ == "__main__":
    main()
