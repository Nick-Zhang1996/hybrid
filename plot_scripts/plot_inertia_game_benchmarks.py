# With inertia correction (all time in ms)
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "outputs" / "benchmarks"

CARS = np.array([2, 3, 4, 5, 6, 7, 8], dtype=float)

SCENARIOS = {
    "without_inertia": {
        "label": "Without Inertia Correction",
        "color": "black",
        "fmt": "o--",
        "markerfacecolor": "white",
        "markeredgecolor": "black",
        "markeredgewidth": 1.0,
    },
    "with_inertia": {
        "label": "With Inertia Correction",
        "color": "#0055AA",
        "fmt": "s-",
        "markerfacecolor": "#0055AA",
        "markeredgecolor": "#0055AA",
    },
    # "ipm_no_inertia": {
    #     "label": "IPM (no inertia)",
    #     "color": "#D20418",
    #     "fmt": "*-",
    #     "markerfacecolor": "#D20418",
    #     "markeredgecolor": "#D20418",
    # },
    "rd3g_baseline": {
        "label": "RD3G Baseline",
        "color": "#D62728",
        "fmt": "^-.",
        "markerfacecolor": "white",
        "markeredgecolor": "#D62728",
        "markeredgewidth": 1.0,
    },
    "ilqgame": {
        "label": "iLQGame",
        "color": "#6d28d9",
        "fmt": "D:",
        "markerfacecolor": "white",
        "markeredgecolor": "#6d28d9",
        "markeredgewidth": 1.0,
    },
    "algames": {
        "label": "ALGAMES",
        "color": "#b91c1c",
        "fmt": "X-.",
        "markerfacecolor": "white",
        "markeredgecolor": "#b91c1c",
        "markeredgewidth": 1.0,
    },
}

PLOT_SCENARIOS = ("without_inertia", "with_inertia", "ilqgame", "algames")

GAMES = {
    "merge": {
        "title": "Merge Game",
        "series": {
            "with_inertia": {
                "time_mean": np.array(
                    [
                        13.701128959655762,
                        31.11893653869629,
                        60.20157337188721,
                        98.62836837768555,
                        129.09189224243164,
                        203.43015670776367,
                        241.0167407989502,
                    ]
                ),
                "time_var": np.array(
                    [
                        89.05793371070557,
                        460.4966592229175,
                        1388.5599936889998,
                        3168.9017830462034,
                        7014.893784189009,
                        11260.259495204172,
                        16677.92382895923,
                    ]
                ),
                "conv_mean": np.array([0.96, 0.84, 0.74, 0.62, 0.66, 0.54, 0.5]),
                "opt_mean": np.array([0.68, 0.48, 0.36, 0.24, 0.22, 0.12, 0.12]),
            },
            "without_inertia": {
                "time_mean": np.array(
                    [
                        9.951138496398926,
                        24.0029239654541,
                        41.50583744049072,
                        59.10581588745117,
                        81.65302276611328,
                        156.80120468139648,
                        153.08618068695068,
                    ]
                ),
                "time_var": np.array(
                    [
                        0064.0026744537181,
                        0452.5735902510859,
                        1077.1046952942243,
                        2277.3320300247177,
                        4874.19258854743,
                        12633.891934981056,
                        12586.491240343754,
                    ]
                ),
                "conv_mean": np.array([0.94, 0.84, 0.86, 0.84, 0.9, 0.66, 0.76]),
                "opt_mean": np.array([0.6, 0.44, 0.36, 0.24, 0.22, 0.1, 0.08]),
            },
            "rd3g_baseline": {
                "time_mean": np.array(
                    [
                        19.655447006225586,
                        26.420068740844727,
                        60.15329360961914,
                        76.67149066925049,
                        111.84639930725098,
                        196.01680278778076,
                        254.1122341156006,
                    ]
                ),
                "time_var": np.array(
                    [
                        0223.56003851846257,
                        0111.32965713613889,
                        1807.4582180975085,
                        1303.6110900229914,
                        2578.892789979363,
                        11920.197526804737,
                        17022.68117576059,
                    ]
                ),
                "conv_mean": np.array([0.88, 0.86, 0.72, 0.64, 0.68, 0.58, 0.58]),
                "opt_mean": np.array([0.88, 0.86, 0.72, 0.64, 0.68, 0.58, 0.58]),
            },
            "ipm_no_inertia": {
                "time_mean": np.array(
                    [
                        7.8603172302246085,
                        15.662398338317871,
                        35.950140953063965,
                        50.873379707336426,
                        67.74871349334717,
                        90.0473690032959,
                        131.45367622375488,
                    ]
                ),
                "time_var": np.array(
                    [
                        0003.974978963378817,
                        0009.8839142629231,
                        0224.11141903987755,
                        0432.70459708785436,
                        0600.3010039677292,
                        1118.7105300836264,
                        1268.1023916421508,
                    ]
                ),
                "conv_mean": np.array([0.86, 0.88, 0.66, 0.5, 0.56, 0.5, 0.3]),
                "opt_mean": np.array([0.74, 0.88, 0.66, 0.5, 0.56, 0.5, 0.3]),
            },
            "ilqgame": {
                "conv_mean": np.array([0.86, 0.8, 0.53, 0.59, 0.35, 0.35, 0.3]),
                "opt_mean": np.array([0.86, 0.8, 0.53, 0.59, 0.35, 0.35, 0.3]),
                "time_mean": np.array(
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
                "time_var": np.array(
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
                "converged_time_mean": np.array(
                    [
                        53.993158562238825,
                        68.17665696144104,
                        98.13431074034493,
                        221.99275534031753,
                        426.01279531206404,
                        652.6949746268136,
                        982.2649399439493,
                    ]
                ),
                "converged_time_var": np.array(
                    [
                        19.169062864033876,
                        11.936020246672278,
                        68.70710800971378,
                        266.5757959454139,
                        1217.446656260387,
                        6792.32164061803,
                        7220.196635886333,
                    ]
                ),
            },
            "algames": {
                "conv_mean": np.array([0.64, 0.59, 0.07, 0.11, 0.02, 0.04, 0.01]),
                "opt_mean": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                "time_mean": np.array(
                    [
                        42.71582218000001,
                        88.41220651999998,
                        168.15693348,
                        273.19421016999996,
                        407.19855172,
                        648.9726904300001,
                        np.nan,
                    ]
                ),
                "time_var": np.array(
                    [
                        413.9907289139453,
                        1326.588876284679,
                        5142.001437376575,
                        4697.550621599723,
                        20467.543315854466,
                        30068.628265965486,
                        np.nan,
                    ]
                ),
                "converged_time_mean": np.array(
                    [
                        41.91985978125,
                        87.31832784745762,
                        169.4469608571429,
                        272.6242022727272,
                        507.875706,
                        776.49892825,
                        np.nan,
                    ]
                ),
                "converged_time_var": np.array(
                    [
                        319.66240747378043,
                        1196.8321036656055,
                        3475.5849548736546,
                        5320.568058046142,
                        26985.743048077646,
                        40099.57931078943,
                        np.nan,
                    ]
                ),
            },
        },
    },
    "intersection": {
        "title": "Intersection Game",
        "series": {
            "without_inertia": {
                "time_mean": np.array(
                    [
                        19.78468894958496,
                        46.96185827255249,
                        112.48016357421875,
                        158.4071445465088,
                        223.65604400634766,
                        270.9224534034729,
                        356.6710329055786,
                    ]
                ),
                "time_var": np.array(
                    [
                        0.22962950055216425,
                        0.9017173539879139,
                        2.1806401817912047,
                        3.7673912668391036,
                        13.823989561434837,
                        19.252338883549847,
                        22.039619464214436,
                    ]
                ),
                "conv_mean": np.array([0.88, 0.83, 0.58, 0.55, 0.54, 0.57, 0.45]),
                "opt_mean": np.array([0.87, 0.82, 0.55, 0.52, 0.49, 0.53, 0.45]),
            },
            "with_inertia": {
                "time_mean": np.array(
                    [
                        21.021089553833008,
                        48.70379447937012,
                        114.14446830749512,
                        160.25827646255493,
                        230.58194160461426,
                        279.30853843688965,
                        363.4367084503174,
                    ]
                ),
                "time_var": np.array(
                    [
                        0.2933674018444435,
                        1.079661458821283,
                        2.1565434196554634,
                        3.559684214474436,
                        12.342151595634185,
                        19.587298116819827,
                        24.125354241422112,
                    ]
                ),
                "conv_mean": np.array([0.88, 0.82, 0.56, 0.54, 0.5, 0.57, 0.46]),
                "opt_mean": np.array([0.88, 0.82, 0.55, 0.54, 0.49, 0.56, 0.46]),
            },
            "ilqgame": {
                "conv_mean": np.array([0.33, 0.33, 0.09, 0.09, 0.14, 0.13, 0.08]),
                "opt_mean": np.array([0.33, 0.33, 0.09, 0.09, 0.14, 0.13, 0.08]),
                "time_mean": np.array(
                    [
                        73.88120412826538,
                        103.09318780899048,
                        177.57079601287842,
                        376.31266832351685,
                        614.9493026733398,
                        992.4918532371521,
                        1539.0213251113892,
                    ]
                ),
                "time_var": np.array(
                    [
                        602.6831497261527,
                        1299.789576307893,
                        1140.9700014968396,
                        4423.218861585218,
                        15497.33214860439,
                        41822.61452097279,
                        54600.55862794944,
                    ]
                ),
                "converged_time_mean": np.array(
                    [
                        40.516925580573805,
                        53.76552812980883,
                        87.39935027228461,
                        190.29156366984049,
                        321.1491789136614,
                        501.4750407292292,
                        850.0849306583405,
                    ]
                ),
                "converged_time_var": np.array(
                    [
                        41.324656482384384,
                        123.83068073771347,
                        410.535920753004,
                        2308.972206290289,
                        5888.485092363717,
                        2996.963101230709,
                        51275.60152487742,
                    ]
                ),
            },
            "algames": {
                "conv_mean": np.array([0.62, 0.62, 0.38, 0.46, 0.48, 0.52, 0.65]),
                "opt_mean": np.array([0.33, 0.31, 0.03, 0.03, 0.06, 0.06, 0.03]),
                "time_mean": np.array(
                    [
                        33.806607199999995,
                        86.04386529,
                        210.74992079,
                        397.92947175000006,
                        677.2919811099999,
                        1128.4692293399999,
                        1804.7254560099996,
                    ]
                ),
                "time_var": np.array(
                    [
                        404.0963094641855,
                        3018.470645023467,
                        2997.1266881855585,
                        9505.005490880736,
                        32908.10863115004,
                        68859.13122034758,
                        83857.74627732802,
                    ]
                ),
                "converged_time_mean": np.array(
                    [
                        24.948264548387094,
                        67.81525633870966,
                        208.30337692105263,
                        374.73579758695655,
                        645.378779375,
                        1069.377070673077,
                        1780.218086276923,
                    ]
                ),
                "converged_time_var": np.array(
                    [
                        444.2578459569503,
                        3286.2411017312043,
                        4619.815233581639,
                        11565.38019735902,
                        55165.89842617905,
                        115525.05386607517,
                        119604.61257745216,
                    ]
                ),
            },
        },
    },
}

# --- 2. IEEE Formatting Setup ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.figsize': (3.5, 2.5),
    'text.usetex': False,
    'lines.linewidth': 1.5,
    'lines.markersize': 5  # Slightly smaller to accommodate error bars
})


def plot_runtime(game_key, output_name):
    fig, ax = plt.subplots(constrained_layout=True)
    game = GAMES[game_key]

    for scenario_key in PLOT_SCENARIOS:
        if scenario_key not in game["series"]:
            continue

        scenario_style = SCENARIOS[scenario_key]
        series = game["series"][scenario_key]
        time_mean = series["time_mean"]
        time_std = np.sqrt(np.clip(series["time_var"], a_min=0.0, a_max=None))
        finite = np.isfinite(time_mean) & np.isfinite(time_std)
        ax.errorbar(
            CARS[finite],
            time_mean[finite],
            yerr=time_std[finite],
            label=scenario_style["label"],
            fmt=scenario_style["fmt"],
            color=scenario_style["color"],
            ecolor=scenario_style["color"],
            capsize=3,
            markerfacecolor=scenario_style["markerfacecolor"],
            markeredgecolor=scenario_style["markeredgecolor"],
            markeredgewidth=scenario_style.get("markeredgewidth", 1.0),
        )

    ax.set_title(game["title"])
    ax.set_xlabel('Number of Cars ($N$)')
    ax.set_ylabel('Computation Time (ms)')
    ax.set_xticks(CARS)
    ax.grid(True, which='major', linestyle=':', alpha=0.6)
    ax.minorticks_on()
    ax.grid(True, which='minor', linestyle=':', alpha=0.15)
    ax.legend(loc='upper left', frameon=True, fancybox=False, edgecolor='black', framealpha=1.0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / output_name, format='png', dpi=300)
    print(f"Plot generated: {output_name}")
    plt.show()


plot_runtime("merge", "runtime_comparison.png")
plot_runtime("intersection", "intersection_runtime_comparison.png")
