import numpy as np
import matplotlib.pyplot as plt

CARS = np.array([2, 3, 4, 5, 6, 7, 8], dtype=float)

SCENARIOS = {
    # "non_vne": {
    #     "label": "Non-VNE",
    #     "color": "#1f3c88",
    #     "marker": "o",
    #     "linestyle": "-",
    # },
    "vne_no_precondition": {
        "label": "VNE (No Preconditioner)",
        "color": "#8a5a00",
        "marker": "s",
        "linestyle": "--",
    },
    "vne_with_precondition": {
        "label": "VNE (With Preconditioner)",
        "color": "#0f766e",
        "marker": "^",
        "linestyle": "-",
    },
    "ilqgame": {
        "label": "iLQGame",
        "color": "#6d28d9",
        "marker": "D",
        "linestyle": ":",
    },
}

GAMES = {
    "car_merge": {
        "title": "Car Merge",
        "series": {
            # "non_vne": {
            #     "converged_rate": np.array([0.86, 0.88, 0.66, 0.50, 0.56, 0.52, 0.30]),
            #     "optimal_rate": np.array([0.74, 0.88, 0.66, 0.50, 0.56, 0.52, 0.30]),
            #     "overall_mean_ms": np.array([7.3, 13.5, 38.0, 82.3, 67.5, 91.4, 139.9]),
            #     "overall_var_ms": np.array([0.0, 0.0, 0.2, 1.2, 0.6, 1.1, 1.8]),
            #     "converged_mean_ms": np.array([7.0, 12.9, 28.9, 53.3, 48.2, 65.9, 96.8]),
            #     "converged_var_ms": np.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.8]),
            # },
            "vne_no_precondition": {
                "converged_rate": np.array([0.86, 0.88, 0.66, 0.50, 0.56, 0.50, 0.30]),
                "optimal_rate": np.array([0.74, 0.88, 0.66, 0.50, 0.56, 0.50, 0.30]),
                "overall_mean_ms": np.array([7.1, 13.6, 29.9, 53.2, 70.9, 97.0, 174.5]),
                "overall_var_ms": np.array([0.0, 0.0, 0.1, 0.4, 0.8, 1.3, 2.6]),
                "converged_mean_ms": np.array([6.7, 13.0, 23.2, 35.7, 51.0, 69.2, 117.2]),
                "converged_var_ms": np.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.9]),
            },
            "vne_with_precondition": {
                "converged_rate": np.array([0.86, 0.88, 0.66, 0.50, 0.56, 0.50, 0.30]),
                "optimal_rate": np.array([0.74, 0.88, 0.66, 0.50, 0.56, 0.50, 0.30]),
                "overall_mean_ms": np.array([3.4, 5.6, 12.0, 18.4, 25.9, 35.0, 54.5]),
                "overall_var_ms": np.array([0.0, 0.0, 0.0, 0.1, 0.1, 0.2, 0.3]),
                "converged_mean_ms": np.array([3.2, 5.3, 9.2, 12.3, 18.5, 25.0, 35.9]),
                "converged_var_ms": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]),
            },
            "ilqgame": {
                "converged_rate": np.array([0.86, 0.8, 0.53, 0.59, 0.35, 0.35, 0.3]),
                "optimal_rate": np.array([0.86, 0.8, 0.53, 0.59, 0.35, 0.35, 0.3]),
                "overall_mean_ms": np.array(
                    [
                        66.98693513870239,
                        86.18404626846313,
                        166.93608045578003,
                        296.275839805603,
                        588.9146089553833,
                        901.2544202804565,
                        1366.5526676177979,
                    ]
                ),
                "overall_var_ms": np.array(
                    [
                        0.5432048071534779,
                        0.7233699463030177,
                        5.033985963476715,
                        7.366273484815135,
                        16.239235671409347,
                        33.9074968033219,
                        65.63233244570698,
                    ]
                ),
                "converged_mean_ms": np.array([59.9, 74.0, 112.8, 227.9, 423.1, 663.6, 990.0]),
                "converged_var_ms": np.array([0.2, 0.1, 0.3, 0.4, 1.5, 6.9, 10.4]),
            },
        },
    },
    "car_racing": {
        "title": "Car Racing",
        "series": {
            # "non_vne": {
            #     "converged_rate": np.array([0.68, 0.36, 0.32, 0.26, 0.16, 0.10, 0.04]),
            #     "optimal_rate": np.array([0.68, 0.36, 0.32, 0.26, 0.16, 0.10, 0.04]),
            #     "overall_mean_ms": np.array([13.6, 48.6, 199.7, 597.4, 155.8, 205.3, 279.0]),
            #     "overall_var_ms": np.array([0.0, 0.5, 10.2, 81.0, 5.4, 8.2, 13.5]),
            #     "converged_mean_ms": np.array([13.0, 44.6, 227.3, 655.0, 153.9, 225.2, 302.0]),
            #     "converged_var_ms": np.array([0.0, 0.2, 6.7, 57.1, 1.5, 0.2, 0.3]),
            # },
            "vne_no_precondition": {
                "converged_rate": np.array([0.70, 0.36, 0.32, 0.26, 0.18, 0.10, np.nan]),
                "optimal_rate": np.array([0.70, 0.36, 0.32, 0.26, 0.18, 0.10, np.nan]),
                "overall_mean_ms": np.array([13.3, 38.8, 84.6, 156.6, 269.2, 433.2, np.nan]),
                "overall_var_ms": np.array([0.0, 0.3, 1.7, 5.4, 16.8, 39.2, np.nan]),
                "converged_mean_ms": np.array([13.2, 35.2, 93.3, 166.1, 286.8, 483.9, np.nan]),
                "converged_var_ms": np.array([0.0, 0.1, 1.1, 3.3, 9.6, 0.4, np.nan]),
            },
            "vne_with_precondition": {
                "converged_rate": np.array([0.68, 0.38, 0.26, 0.26, 0.20, 0.10, 0.04]),
                "optimal_rate": np.array([0.68, 0.38, 0.26, 0.26, 0.20, 0.10, 0.04]),
                "overall_mean_ms": np.array([8.7, 33.4, 77.2, 154.8, 275.2, 477.2, 767.2]),
                "overall_var_ms": np.array([0.0, 0.2, 0.5, 1.8, 3.8, 8.0, 23.3]),
                "converged_mean_ms": np.array([6.2, 18.5, 40.9, 99.4, 181.4, 288.6, 457.7]),
                "converged_var_ms": np.array([0.0, 0.0, 0.1, 2.4, 3.4, 0.9, 1.7]),
            },
            "ilqgame": {
                "converged_rate": np.array([0.62, 0.33, 0.27, 0.1, 0.07, 0.01, 0.0]),
                "optimal_rate": np.array([0.62, 0.33, 0.27, 0.1, 0.07, 0.01, 0.0]),
                "overall_mean_ms": np.array(
                    [
                        45.940704345703125,
                        91.2267518043518,
                        171.39769792556763,
                        380.6813359260559,
                        618.7534976005554,
                        1001.6530418395997,
                        1512.8580856323242,
                    ]
                ),
                "overall_var_ms": np.array(
                    [
                        0.25818565158497214,
                        0.7999751282930617,
                        1.8214060676045565,
                        2.7686233750095823,
                        6.180023067747533,
                        4.564591651106322,
                        2.858841629938616,
                    ]
                ),
                "converged_mean_ms": np.array(
                    [
                        34.305983974087624,
                        54.13887717507102,
                        107.14636025605378,
                        230.8809518814087,
                        339.9608816419329,
                        552.6065826416016,
                        np.nan,
                    ]
                ),
                "converged_var_ms": np.array(
                    [
                        0.03769501961262479,
                        0.12919423973900004,
                        0.5582652037230221,
                        0.8348763115742487,
                        0.8533594901095689,
                        0.0,
                        np.nan,
                    ]
                ),
            },
        },
    },
}


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.figsize": (7.0, 4.2),
        "axes.linewidth": 0.9,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def _valid_xy(x, y, yerr=None):
    mask = np.isfinite(y)
    if yerr is not None:
        mask &= np.isfinite(yerr)
        return x[mask], y[mask], yerr[mask]
    return x[mask], y[mask]


def _style_axis(ax, ylabel, ymax=None):
    ax.set_xlabel("Number of Cars")
    ax.set_ylabel(ylabel)
    ax.set_xticks(CARS)
    if ymax is not None:
        ax.set_ylim(0.0, ymax)
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_runtime(game_key, runtime_key, variance_key, title_suffix):
    game = GAMES[game_key]
    fig, ax = plt.subplots()

    for scenario_key, scenario_style in SCENARIOS.items():
        series = game["series"][scenario_key]
        mean = series[runtime_key]
        std = np.sqrt(series[variance_key])
        x, y, yerr = _valid_xy(CARS, mean, std)
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            label=scenario_style["label"],
            color=scenario_style["color"],
            linestyle=scenario_style["linestyle"],
            marker=scenario_style["marker"],
            capsize=3,
            markerfacecolor="white",
            markeredgewidth=1.4,
        )

    ax.set_title(f"{game['title']}: {title_suffix}")
    _style_axis(ax, "Runtime (ms)")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    plt.show()


def plot_rates(game_key):
    game = GAMES[game_key]
    fig, ax = plt.subplots()

    width = 0.22
    centers = np.arange(len(CARS), dtype=float)

    for idx, (scenario_key, scenario_style) in enumerate(SCENARIOS.items()):
        series = game["series"][scenario_key]
        converged = series["converged_rate"]
        optimal = series["optimal_rate"]
        x = centers + (idx - 1) * width

        ax.bar(
            x,
            converged,
            width=width,
            label=f"{scenario_style['label']} Converged",
            color=scenario_style["color"],
            alpha=0.35,
            edgecolor=scenario_style["color"],
            linewidth=1.0,
        )
        ax.bar(
            x,
            optimal,
            width=width,
            label=f"{scenario_style['label']} Optimal",
            color=scenario_style["color"],
            alpha=0.95,
            edgecolor=scenario_style["color"],
            linewidth=1.0,
            hatch="///",
        )

    ax.set_title(f"{game['title']}: Convergence and Optimality Rates")
    ax.set_xlabel("Number of Cars")
    ax.set_ylabel("Rate")
    ax.set_xticks(centers)
    ax.set_xticklabels([str(int(car)) for car in CARS])
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", ncol=2, frameon=False)
    fig.tight_layout()
    plt.show()


def main():
    for game_key in ("car_merge", "car_racing"):
        plot_runtime(
            game_key,
            runtime_key="overall_mean_ms",
            variance_key="overall_var_ms",
            title_suffix="Overall Runtime",
        )
        plot_runtime(
            game_key,
            runtime_key="converged_mean_ms",
            variance_key="converged_var_ms",
            title_suffix="Runtime for Converged Cases",
        )
        plot_rates(game_key)


if __name__ == "__main__":
    main()
