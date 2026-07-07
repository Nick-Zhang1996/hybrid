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
    # "without PDO-Split": {
    #     "label": "without PDO-Split",
    #     "color": "#8a5a00",
    #     "marker": "s",
    #     "linestyle": "--",
    # },
    "rd3g": {
        "label": "RD3G",
        "color": "#222222",
        "marker": "o",
        "linestyle": "--",
    },
    "ilqgame": {
        "label": "iLQGame",
        "color": "#6d28d9",
        "marker": "D",
        "linestyle": ":",
    },
    "algames": {
        "label": "ALGAMES",
        "color": "#b91c1c",
        "marker": "X",
        "linestyle": "-.",
    },
    "PDO-Split": {
        "label": "PDO-Split",
        "color": "#0f766e",
        "marker": "^",
        "linestyle": "-",
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
            "PDO-Split": {
                "converged_rate": np.array([0.86, 0.88, 0.66, 0.50, 0.56, 0.50, 0.30]),
                "optimal_rate": np.array([0.74, 0.88, 0.66, 0.50, 0.56, 0.50, 0.30]),
                "overall_mean_ms": np.array([12.043766975402832, 23.11375141143799, 56.08804702758789, 76.98510408401489, 124.15194511413574, 170.32540798187256, 229.68162536621094]),
                "overall_var_ms": np.array([33.70589055587061, 116.8487030957976, 342.5112015666855, 651.1064964902914, 761.7493507689231, 1335.0465035846582, 2795.2710108244446]),
                "converged_mean_ms": np.array([9.7, 18.3, 40.6, 56.8, 89.2, 120.1, 168.2]),
                "converged_var_ms": np.array([7.7, 24.2, 135.6, 231.4, 472.2, 1035.4, 1522.9]),
            },
            "rd3g": {
                "converged_rate": np.array([0.81, 0.89, 0.53, 0.55, 0.27, 0.32, 0.28]),
                "optimal_rate": np.array([0.81, 0.88, 0.5, 0.55, 0.26, 0.28, 0.27]),
                "overall_mean_ms": np.array([20.310323238372803, 35.96055030822754, 99.36886548995972, 142.38977670669556, 223.3168387413025, 279.4550395011902, 390.073983669281]),
                "overall_var_ms": np.array([317.8028101820644, 865.6070989405633, 1162.0416497099257, 1623.2909212845864, 4041.2309159993306, 7221.258995733586, 16860.36886469725]),
                "converged_mean_ms": np.array([14.774946518886237, 30.455624119619305, 84.74168237650169, 128.9092497392134, 207.21218321058484, 242.3539236187935, 378.81284952163696]),
                "converged_var_ms": np.array([199.3033972759219, 685.8694034275914, 1160.7605345982915, 1724.3061014548878, 3500.731972913843, 8920.316940867979, 14157.877315644233]),
            },
            "with PDO-Split": {
                "converged_rate": np.array([0.86, 0.88, 0.66, 0.50, 0.56, 0.50, 0.30]),
                "optimal_rate": np.array([0.74, 0.88, 0.66, 0.50, 0.56, 0.50, 0.30]),
                "overall_mean_ms": np.array([7.233061790466309, 11.21722936630249, 23.204176425933838, 29.779224395751953, 46.22326135635376, 59.98722314834595, 74.1278862953186]),
                "overall_var_ms": np.array([12.122721965647543, 27.73494566264957, 56.841766716678414, 97.30240830240291, 107.17392414459822, 166.62285099002366, 264.9633838835768]),
                "converged_mean_ms": np.array([5.8, 8.9, 16.9, 21.9, 33.0, 42.3, 55.3]),
                "converged_var_ms": np.array([3.2, 5.7, 22.9, 34.2, 65.5, 127.9, 154.7]),
            },
            "ilqgame": {
                "converged_rate": np.array([0.86, 0.8, 0.53, 0.59, 0.35, 0.35, 0.3]),
                "optimal_rate": np.array([0.86, 0.8, 0.53, 0.59, 0.35, 0.35, 0.3]),
                "overall_mean_ms": np.array(
                    [
                        60.09026050567627,
                        80.24375200271606,
                        137.41577863693237,
                        284.67933893203735,
                        590.5238628387451,
                        886.4679312705994,
                        1364.2191362380981,
                    ]
                ),
                "overall_var_ms": np.array(
                    [
                        249.99045254774043,
                        608.3441125209615,
                        1850.8539189365079,
                        5919.625403425578,
                        15647.854178624815,
                        34032.448392997016,
                        66434.67735166449,
                    ]
                ),
                "converged_mean_ms": np.array([53.993158562238825, 68.17665696144104, 98.13431074034493, 221.99275534031753, 426.01279531206404, 652.6949746268136, 982.2649399439493]),
                "converged_var_ms": np.array([19.169062864033876, 11.936020246672278, 68.70710800971378, 266.5757959454139, 1217.446656260387, 6792.32164061803, 7220.196635886333]),
            },
            "algames": {
                "converged_rate": np.array([0.64, 0.59, 0.07, 0.11, 0.02, 0.04, 0.01]),
                "optimal_rate": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                "overall_mean_ms": np.array(
                    [
                        42.71582218000001,
                        88.41220651999998,
                        168.15693348,
                        273.19421016999996,
                        407.19855172,
                        648.9726904300001,
                        np.nan,  # 1107.30742397,
                    ]
                ),
                "overall_var_ms": np.array(
                    [
                        413.9907289139453,
                        1326.588876284679,
                        5142.001437376575,
                        4697.550621599723,
                        20467.543315854466,
                        30068.628265965486,
                        np.nan,  # 30514.186615400744,
                    ]
                ),
                # 1150.3694440000002
                "converged_mean_ms": np.array([41.91985978125, 87.31832784745762, 169.4469608571429, 272.6242022727272, 507.875706, 776.49892825, np.nan]),
                # 0.0
                "converged_var_ms": np.array([319.66240747378043, 1196.8321036656055, 3475.5849548736546, 5320.568058046142, 26985.743048077646, 40099.57931078943, np.nan]),
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
            # "without PDO-Split": {
            #     "converged_rate": np.array([0.70, 0.36, 0.32, 0.26, 0.18, 0.10, 0.09]),
            #     "optimal_rate": np.array([0.70, 0.36, 0.32, 0.26, 0.18, 0.10, 0.09]),
            #     "overall_mean_ms": np.array([12.256879806518555, 33.801796436309814, 69.87300395965576, 113.65606784820557, 196.7015838623047, 302.7076244354248, 479.8362326622009]),
            #     "overall_var_ms": np.array([10.512554047954836, 34.435045881190256, 519.6672941096721, 600.1236934088182, 3412.4275750515603, 8148.398154041843, 19645.44732871838]),
            #     "converged_mean_ms": np.array([12.078584943498885, 36.563304754403916, 86.29893660545349, 149.90892012914023, 271.041239009184, 436.7823600769043, 824.3863317701552]),
            #     "converged_var_ms": np.array([11.80600606319934, 39.38504722116478, 485.875486692997, 463.1084861534716, 4226.868115511434, 4210.3826920272995, 44968.09609025761]),

            # },
            "rd3g": {
                "converged_rate": np.array([0.69, 0.39, 0.41, 0.24, 0.17, 0.08, 0.09]),
                "optimal_rate": np.array([0.69, 0.39, 0.41, 0.24, 0.17, 0.07, 0.09]),
                "overall_mean_ms": np.array([11.124329566955566, 29.4661808013916, 63.7111234664917, 109.2027473449707, 182.3928451538086, 295.6836795806885, 478.42753171920776]),
                "overall_var_ms": np.array([12.594514252418776, 41.87983688357235, 100.28102371777545, 328.2219442308815, 2555.2848715277823, 2628.7098878618963, 8957.492904748387]),
                "converged_mean_ms": np.array([10.523595671722855, 29.87206899202787, 70.42077111034858, 132.21648335456848, 236.41382946687585, 405.84060549736023, 698.5148853725858]),
                "converged_var_ms": np.array([3.356977555269103, 15.484893386055996, 82.90944466994762, 337.6653946215076, 2841.048401793572, 1381.736035845016, 7244.126206978482]),
            },
            "PDO-Split": {
                "converged_rate": np.array([0.68, 0.38, 0.26, 0.26, 0.20, 0.10, 0.04]),
                "optimal_rate": np.array([0.68, 0.38, 0.26, 0.26, 0.20, 0.10, 0.04]),
                "overall_mean_ms": np.array([9.931654930114746, 35.39132356643677, 78.18204641342163, 157.92293071746826, 287.123007774353, 504.5762801170349, 790.569634437561]),
                "overall_var_ms": np.array([29.09422330023972, 196.8146093434541, 600.1986860438307, 1769.2693889905286, 3968.805273861313, 4489.196883514177, 10528.18855164421]),
                "converged_mean_ms": np.array([6.7096131188528885, 20.203620195388794, 51.84188014582584, 91.36427442232768, 168.94644849440633, 300.87736674717496, 507.785826921463]),
                "converged_var_ms": np.array([1.3540296365196198, 11.471129616857922, 221.25095238852344, 414.13561677444085, 1742.6459621107974, 4821.123247169527, 12460.768146389077]),
            },
            "ilqgame": {
                "converged_rate": np.array([0.61, 0.3, 0.28, 0.14, 0.12, 0.02, np.nan]),
                "optimal_rate": np.array([0.61, 0.3, 0.28, 0.14, 0.12, 0.02, np.nan]),
                "overall_mean_ms": np.array(
                    [
                        88.1,
                        171.3,
                        340.2,
                        733.2,
                        1210.8,
                        1968.4,
                        np.nan,
                    ]
                ),
                "overall_var_ms": np.array(
                    [
                        1136.642,
                        2791.291,
                        8432.140,
                        14707.349,
                        35849.523,
                        16829.992,
                        np.nan,
                    ]
                ),
                "converged_mean_ms": np.array(
                    [
                        62.1,
                        92.2,
                        205.8,
                        450.9,
                        725.5,
                        np.nan,  # 1108.7,
                        np.nan,
                    ]
                ),
                "converged_var_ms": np.array(
                    [
                        82.296,
                        70.170,
                        1242.848,
                        5054.949,
                        23705.635,
                        np.nan,  # 2.067,
                        np.nan,
                    ]
                ),
            },
            "algames": {
                "converged_rate": np.array([0.93, 0.91, 0.89, 0.87, 0.85, 0.84, 0.85]),
                "optimal_rate": np.array([0.29, 0.13, 0.03, 0.03, 0.01, 0.0, 0.0]),
                "overall_mean_ms": np.array(
                    [
                        51.831329200000006,
                        130.90379899,
                        277.51545565000004,
                        492.1114093700001,
                        889.38516026,
                        1402.53521619,
                        2073.8890129800006,
                    ]
                ),
                "overall_var_ms": np.array(
                    [
                        1220.1397657454208,
                        2007.1527526536672,
                        5379.706297286654,
                        10374.302104793454,
                        17447.512747375647,
                        12842.163526293849,
                        17599.98380462609,
                    ]
                ),
                "converged_mean_ms": np.array(
                    [
                        51.10948795698925,
                        128.01543783516485,
                        279.9525454719101,
                        491.9787923678161,
                        892.6305057529412,
                        1405.9697927380953,
                        2075.079878341177,
                    ]
                ),
                "converged_var_ms": np.array(
                    [
                        1304.4157305114772,
                        1973.6590507945655,
                        5979.3049023547865,
                        11282.343174742993,
                        18789.382230513136,
                        13817.339488231682,
                        17569.84872013155,
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

    scenario_items = [
        (scenario_key, scenario_style)
        for scenario_key, scenario_style in SCENARIOS.items()
        if scenario_key in game["series"]
    ]
    for scenario_key, scenario_style in scenario_items:
        series = game["series"][scenario_key]
        mean = series[runtime_key]
        runtime_std = np.sqrt(np.clip(series[variance_key], a_min=0.0, a_max=None))
        x, y, yerr = _valid_xy(CARS, mean, runtime_std)
        ax.plot(
            x,
            y,
            label=scenario_style["label"],
            color=scenario_style["color"],
            linestyle=scenario_style["linestyle"],
            marker=scenario_style["marker"],
            markerfacecolor="white",
            markeredgewidth=1.4,
        )
        ax.fill_between(
            x,
            np.clip(y - yerr, a_min=0.0, a_max=None),
            y + yerr,
            color=scenario_style["color"],
            alpha=0.14,
            linewidth=0.0,
        )

    ax.set_title(f"{game['title']}: {title_suffix}")
    _style_axis(ax, "Runtime (ms)")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    plt.show()


def plot_rates(game_key):
    game = GAMES[game_key]
    fig, ax = plt.subplots()

    scenario_items = [
        (scenario_key, scenario_style)
        for scenario_key, scenario_style in SCENARIOS.items()
        if scenario_key in game["series"]
    ]
    width = min(0.22, 0.8 / len(scenario_items))
    centers = np.arange(len(CARS), dtype=float)
    offset_center = (len(scenario_items) - 1) / 2.0

    for idx, (scenario_key, scenario_style) in enumerate(scenario_items):
        series = game["series"][scenario_key]
        converged = series["converged_rate"]
        optimal = series["optimal_rate"]
        x = centers + (idx - offset_center) * width

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
