# With inertia correction (all time in ms)
import numpy as np
import matplotlib.pyplot as plt
in_time_mean = [13.701128959655762, 31.11893653869629, 60.20157337188721,
                98.62836837768555, 129.09189224243164, 203.43015670776367, 241.0167407989502]
in_time_var = [0.08905793371070557, 0.4604966592229175, 1.3885599936889998,
               3.1689017830462034, 7.014893784189009, 11.260259495204172, 16.67792382895923]
in_conv_mean = [0.96, 0.84, 0.74, 0.62, 0.66, 0.54, 0.5]
in_conv_var = [0.038400000000000004, 0.1344, 0.19240000000000002,
               0.23559999999999998, 0.22440000000000002, 0.24840000000000004, 0.25]
in_opt_mean = [0.68, 0.48, 0.36, 0.24, 0.22, 0.12, 0.12]
in_opt_var = [0.2176, 0.24959999999999996, 0.23039999999999997,
              0.18240000000000006, 0.17159999999999997, 0.10559999999999999, 0.1056]
# Without inertia correction
noin_time_mean = [9.951138496398926, 24.0029239654541, 41.50583744049072,
                  59.10581588745117, 81.65302276611328, 156.80120468139648, 153.08618068695068]
noin_time_var = [0.0640026744537181, 0.4525735902510859, 1.0771046952942243,
                 2.2773320300247177, 4.87419258854743, 12.633891934981056, 12.586491240343754]
noin_conv_mean = [0.94, 0.84, 0.86, 0.84, 0.9, 0.66, 0.76]
noin_conv_var = [0.056400000000000006, 0.13439999999999996,
                 0.12039999999999995, 0.13439999999999996, 0.09, 0.22440000000000002, 0.1824]
noin_opt_mean = [0.6, 0.44, 0.36, 0.24, 0.22, 0.1, 0.08]
noin_opt_var = [0.24000000000000005, 0.24640000000000004, 0.23039999999999997,
                0.1824, 0.17159999999999997, 0.09, 0.07360000000000001]
# Original RD3G solver
ori_time_mean = [19.655447006225586, 26.420068740844727, 60.15329360961914,
                 76.67149066925049, 111.84639930725098, 196.01680278778076, 254.1122341156006]
ori_time_var = [0.22356003851846257, 0.11132965713613889, 1.8074582180975085,
                1.3036110900229914, 2.578892789979363, 11.920197526804737, 17.02268117576059]
ori_conv_mean = [0.88, 0.86, 0.72, 0.64, 0.68, 0.58, 0.58]
ori_conv_var = [0.1056, 0.12039999999999998, 0.20160000000000003,
                0.23039999999999997, 0.2176, 0.24359999999999998, 0.24359999999999998]
ori_opt_mean = [0.88, 0.86, 0.72, 0.64, 0.68, 0.58, 0.58]
ori_opt_var = [0.1056, 0.12039999999999998, 0.20160000000000003,
               0.23039999999999997, 0.2176, 0.24359999999999998, 0.24359999999999998]


car_count = np.array([2, 3, 4, 5, 6, 7, 8])

# A. Proposed (With Inertia)
in_time_std = np.sqrt(in_time_var)  # Convert Variance to Std Dev

# B. Baseline (No Inertia)
noin_time_std = np.sqrt(noin_time_var)

# C. Original Solver (Existing Paper)
ori_time_std = np.sqrt(ori_time_var)

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

# --- 3. Plotting ---
fig, ax = plt.subplots(constrained_layout=True)

# 1. Plot Original (Existing) - Triangle marker, Red/Gray, Dash-dot
ax.errorbar(car_count, ori_time_mean, yerr=ori_time_std,
            label='RD3G Baseline',
            fmt='^-.',           # Triangle_up marker, dash-dot line
            color='#D62728',     # Muted Red
            ecolor='#D62728',    # Error bar color
            capsize=3,           # End caps on error bars
            markerfacecolor='white',
            markeredgewidth=1.0)

# 2. Plot Baseline (No Inertia) - Circle marker, Black, Dashed
ax.errorbar(car_count, noin_time_mean, yerr=noin_time_std,
            label='Without Inertia Correction',
            fmt='o--',           # Circle marker, dashed line
            color='black',
            ecolor='black',
            capsize=3,
            markerfacecolor='white',
            markeredgewidth=1.0)

# 3. Plot Proposed (With Inertia) - Square marker, Blue, Solid
ax.errorbar(car_count, in_time_mean, yerr=in_time_std,
            label='With Inertia Correction',
            fmt='s-',            # Square marker, solid line
            color='#0055AA',     # IEEE Blue
            ecolor='#0055AA',
            capsize=3,
            markerfacecolor='#0055AA',  # Filled marker for prominence
            markeredgecolor='#0055AA')

# --- 4. Styling & Labeling ---
ax.set_xlabel('Number of Cars ($N$)')
ax.set_ylabel('Computation Time (ms)')

ax.set_xticks(car_count)

# Grid setup
ax.grid(True, which='major', linestyle=':', alpha=0.6)
ax.minorticks_on()
ax.grid(True, which='minor', linestyle=':', alpha=0.15)

# Legend
# 'framealpha=1' ensures the grid lines don't show through the legend box
ax.legend(loc='upper left', frameon=True, fancybox=False, edgecolor='black', framealpha=1.0)

# --- 5. Save ---
# plt.savefig('runtime_comparison.pdf', format='pdf', dpi=300)
plt.savefig('runtime_comparison.png', format='png', dpi=300)

print("Plot generated: runtime_comparison.pdf")
plt.show()
