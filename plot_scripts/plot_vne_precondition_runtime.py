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
    "without PDO-Split": {
        "label": "without PDO-Split",
        "color": "#8a5a00",
        "marker": "s",
        "linestyle": "--",
    },
    "with PDO-Split": {
        "label": "with PDO-Split",
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
    "algames": {
        "label": "ALGAMES",
        "color": "#b91c1c",
        "marker": "X",
        "linestyle": "-.",
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
            "without PDO-Split": {
                "converged_rate": np.array([0.86, 0.88, 0.66, 0.50, 0.56, 0.50, 0.30]),
                "optimal_rate": np.array([0.74, 0.88, 0.66, 0.50, 0.56, 0.50, 0.30]),
                "overall_mean_ms": np.array([12.043766975402832, 23.11375141143799, 56.08804702758789, 76.98510408401489, 124.15194511413574, 170.32540798187256, 229.68162536621094]),
                "overall_var_ms": np.array([33.70589055587061, 116.8487030957976, 342.5112015666855, 651.1064964902914, 761.7493507689231, 1335.0465035846582, 2795.2710108244446]),
                "converged_mean_ms": np.array([9.7, 18.3, 40.6, 56.8, 89.2, 120.1, 168.2]),
                "converged_var_ms": np.array([7.7, 24.2, 135.6, 231.4, 472.2, 1035.4, 1522.9]),
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
                # TODO: update the data dict with the values in comment
                # solver_name='ilqgame', cpp=False, vne=True, precond=False
                # time_mean_vec=[60.09026050567627, 80.24375200271606, 137.41577863693237, 284.67933893203735, 590.5238628387451, 886.4679312705994, 1364.2191362380981]
                # time_var_vec=[249.99045254774043, 608.3441125209615, 1850.8539189365079, 5919.625403425578, 15647.854178624815, 34032.448392997016, 66434.67735166449]
                # converged_time_mean_vec=[53.993158562238825, 68.17665696144104, 98.13431074034493, 221.99275534031753, 426.01279531206404, 652.6949746268136, 982.2649399439493]
                # converged_time_var_vec=[19.169062864033876, 11.936020246672278, 68.70710800971378, 266.5757959454139, 1217.446656260387, 6792.32164061803, 7220.196635886333]
                # conv_mean_vec=[0.86, 0.8, 0.53, 0.59, 0.35, 0.35, 0.3]
                # optimal_mean_vec=[0.86, 0.8, 0.53, 0.59, 0.35, 0.35, 0.3]
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
                        543.2048071534779,
                        723.3699463030177,
                        5033.985963476715,
                        7366.273484815135,
                        16239.235671409347,
                        33907.4968033219,
                        65632.33244570698,
                    ]
                ),
                "converged_mean_ms": np.array([59.9, 74.0, 112.8, 227.9, 423.1, 663.6, 990.0]),
                "converged_var_ms": np.array([0.2, 0.1, 0.3, 0.4, 1.5, 6.9, 10.4]),
            },
            "algames": {
                # TODO: update the data dict with the values in comment
                # time_mean_vec=[42.71582218000001, 88.41220651999998, 168.15693348, 273.19421016999996, 407.19855172, 648.9726904300001, 1107.30742397]
                # time_var_vec=[413.9907289139453, 1326.588876284679, 5142.001437376575, 4697.550621599723, 20467.543315854466, 30068.628265965486, 30514.186615400744]
                # converged_time_mean_vec=[41.91985978125, 87.31832784745762, 169.4469608571429, 272.6242022727272, 507.875706, 776.49892825, 1150.3694440000002]
                # converged_time_var_vec=[319.66240747378043, 1196.8321036656055, 3475.5849548736546, 5320.568058046142, 26985.743048077646, 40099.57931078943, 0.0]
                # conv_mean_vec=[0.64, 0.59, 0.07, 0.11, 0.02, 0.04, 0.01]
                # optimal_mean_vec=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                "converged_rate": np.array([0.64, 0.59, 0.07, 0.11, 0.02, 0.04, 0.01]),
                "optimal_rate": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                "overall_mean_ms": np.array(
                    [
                        40.457602519999995,
                        90.28213739000002,
                        161.86449126,
                        285.79389384999996,
                        415.37859432999994,
                        630.8487440000001,
                        1058.26025594,
                    ]
                ),
                "overall_var_ms": np.array(
                    [
                        00057.88705936248584,
                        01492.5203410200176,
                        04528.082086111775,
                        06653.484865339215,
                        20449.219439226287,
                        26547.73652052324,
                        26035.814529433395,
                    ]
                ),
                "converged_mean_ms": np.array([40.2, 90.2, 137.5, 263.2, 518.5, 715.0, 952.1]),
                "converged_var_ms": np.array([0.0, 1.4, 0.1, 3.6, 26.4, 32.1, 0.0]),
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
            "without PDO-Split": {
                "converged_rate": np.array([0.70, 0.36, 0.32, 0.26, 0.18, 0.10, 0.09]),
                "optimal_rate": np.array([0.70, 0.36, 0.32, 0.26, 0.18, 0.10, 0.09]),
                "overall_mean_ms": np.array([12.256879806518555, 33.801796436309814, 69.87300395965576, 113.65606784820557, 196.7015838623047, 302.7076244354248, 479.8362326622009]),
                "overall_var_ms": np.array([10.512554047954836, 34.435045881190256, 519.6672941096721, 600.1236934088182, 3412.4275750515603, 8148.398154041843, 19645.44732871838]),
                "converged_mean_ms": np.array([12.078584943498885, 36.563304754403916, 86.29893660545349, 149.90892012914023, 271.041239009184, 436.7823600769043, 824.3863317701552]),
                "converged_var_ms": np.array([0.20999999999999994, 0.2379, 0.24, 0.18239999999999998, 0.14110000000000003, 0.056399999999999985, 0.08190000000000001]),
            },
            "with PDO-Split": {
                "converged_rate": np.array([0.68, 0.38, 0.26, 0.26, 0.20, 0.10, 0.04]),
                "optimal_rate": np.array([0.68, 0.38, 0.26, 0.26, 0.20, 0.10, 0.04]),
                "overall_mean_ms": np.array([9.931654930114746, 35.39132356643677, 78.18204641342163, 157.92293071746826, 287.123007774353, 504.5762801170349, 790.569634437561]),
                "overall_var_ms": np.array([29.09422330023972, 196.8146093434541, 600.1986860438307, 1769.2693889905286, 3968.805273861313, 4489.196883514177, 10528.18855164421]),
                "converged_mean_ms": np.array([6.7096131188528885, 20.203620195388794, 51.84188014582584, 91.36427442232768, 168.94644849440633, 300.87736674717496, 507.785826921463]),
                "converged_var_ms": np.array([1.3540296365196198, 11.471129616857922, 221.25095238852344, 414.13561677444085, 1742.6459621107974, 4821.123247169527, 12460.768146389077]),
            },
            "ilqgame": {
                # TODO: update the data dict with the values in comment
                # INFO: main: Benchmarking ilqgame
                # repeat 100/100
                # INFO: main: 2 cars converge = 0.61, optimal = 0.61, mean_dt_ms = 88.1ms, var_dt_ms = 1136.642ms, converged_mean_dt_ms = 62.1ms, converged_var_dt_ms = 82.296ms
                # repeat 100/100
                # INFO: main: 3 cars converge = 0.3, optimal = 0.3, mean_dt_ms = 171.3ms, var_dt_ms = 2791.291ms, converged_mean_dt_ms = 92.2ms, converged_var_dt_ms = 70.170ms
                # repeat 100/100
                # INFO: main: 4 cars converge = 0.28, optimal = 0.28, mean_dt_ms = 340.2ms, var_dt_ms = 8432.140ms, converged_mean_dt_ms = 205.8ms, converged_var_dt_ms = 1242.848ms
                # repeat 100/100
                # INFO: main: 5 cars converge = 0.14, optimal = 0.14, mean_dt_ms = 733.2ms, var_dt_ms = 14707.349ms, converged_mean_dt_ms = 450.9ms, converged_var_dt_ms = 5054.949ms
                # repeat 100/100
                # INFO: main: 6 cars converge = 0.12, optimal = 0.12, mean_dt_ms = 1210.8ms, var_dt_ms = 35849.523ms, converged_mean_dt_ms = 725.5ms, converged_var_dt_ms = 23705.635ms
                # repeat 100/100
                # INFO: main: 7 cars converge = 0.02, optimal = 0.02, mean_dt_ms = 1968.4ms, var_dt_ms = 16829.992ms, converged_mean_dt_ms = 1108.7ms, converged_var_dt_ms = 2.067ms
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
                        258.18565158497214,
                        799.9751282930617,
                        1821.4060676045565,
                        2768.6233750095823,
                        6180.023067747533,
                        4564.591651106322,
                        2858.841629938616,
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
                        37.69501961262479,
                        129.19423973900004,
                        558.2652037230221,
                        834.8763115742487,
                        853.3594901095689,
                        0.0,
                        np.nan,
                    ]
                ),
            },
            "algames": {
                INFO: main: Benchmarking algames
                # TODO: update the data dict with the values in comment
                # time_mean_vec=[51.831329200000006, 130.90379899, 277.51545565000004, 492.1114093700001, 889.38516026, 1402.53521619, 2073.8890129800006]
                # time_var_vec=[1220.1397657454208, 2007.1527526536672, 5379.706297286654, 10374.302104793454, 17447.512747375647, 12842.163526293849, 17599.98380462609]
                # converged_time_mean_vec=[51.10948795698925, 128.01543783516485, 279.9525454719101, 491.9787923678161, 892.6305057529412, 1405.9697927380953, 2075.079878341177]
                # converged_time_var_vec=[1304.4157305114772, 1973.6590507945655, 5979.3049023547865, 11282.343174742993, 18789.382230513136, 13817.339488231682, 17569.84872013155]
                # conv_mean_vec=[0.93, 0.91, 0.89, 0.87, 0.85, 0.84, 0.85]
                # conv_var_vec=[0.0651, 0.08190000000000001, 0.09789999999999999, 0.1131, 0.1275, 0.13440000000000002, 0.1275]
                # optimal_mean_vec=[0.29, 0.13, 0.03, 0.03, 0.01, 0.0, 0.0]
                # optimal_var_vec=[0.20589999999999997, 0.11309999999999999, 0.0291, 0.029100000000000004, 0.009899999999999999, 0.0, 0.0]
                "converged_rate": np.array([0.95, 0.92, 0.92, 0.89, 0.89, 0.87, 0.83]),
                "optimal_rate": np.array([0.31, 0.12, 0.02, 0.0, 0.0, 0.0, 0.0]),
                "overall_mean_ms": np.array(
                    [
                        27.017887759999997,
                        64.40915927,
                        137.03554300000002,
                        242.49503834000004,
                        419.82249399999995,
                        679.44504242,
                        1029.70871266,
                    ]
                ),
                "overall_var_ms": np.array(
                    [
                        410.78771436740186,
                        541.7592565899504,
                        1309.067570047436,
                        1135.3144382266127,
                        4001.9021213135035,
                        7445.649376988003,
                        7465.922867720652,
                    ]
                ),
                "converged_mean_ms": np.array(
                    [
                        26.64785664210526,
                        64.16583408695654,
                        137.51580116304348,
                        243.11249110112362,
                        420.7879936516854,
                        677.0141421034482,
                        1030.6997994457834,
                    ]
                ),
                "converged_var_ms": np.array(
                    [
                        429.3937311437692,
                        588.1178603984131,
                        1413.6068871362073,
                        1255.1182361533606,
                        4241.6121262296,
                        6742.123568981864,
                        8084.505045045667,
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
        runtime_std = np.sqrt(np.clip(series[variance_key], a_min=0.0, a_max=None))
        x, y, yerr = _valid_xy(CARS, mean, runtime_std)
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            label=scenario_style["label"],
            color=scenario_style["color"],
            linestyle=scenario_style["linestyle"],
            marker=scenario_style["marker"],
            capsize=3,
            elinewidth=0.9,
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
    offset_center = (len(SCENARIOS) - 1) / 2.0

    for idx, (scenario_key, scenario_style) in enumerate(SCENARIOS.items()):
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
