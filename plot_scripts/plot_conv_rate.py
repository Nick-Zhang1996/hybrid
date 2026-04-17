""" Plot convergence result with error bar """
import matplotlib.pyplot as plt
import numpy as np

# --- Data Configuration ---
cars = np.array([2, 3, 4, 5, 6, 7, 8])

# No repeats
time_mean = [1.4564800262451172, 2.4991655349731445, 4.476861953735352,
             6.286129951477051, 8.024239540100098, 12.449336051940918, 17.200942039489746]
time_var = [0.00038196927412172945, 0.0012791157946821837, 0.0027230088743181117,
            0.0056895781845241805, 0.008684825052682754, 0.016455146615612648, 0.0307590209331238]
conv_mean = [0.91, 0.83, 0.8, 0.74, 0.82, 0.7, 0.66]
conv_var = [0.08190000000000001, 0.14110000000000003, 0.16,
            0.19239999999999996, 0.1476, 0.20999999999999996, 0.22440000000000004]
optimal_mean = [0.54, 0.42, 0.35, 0.15, 0.19, 0.14, 0.09]
optimal_var = [0.2484, 0.24360000000000007,
               0.2275, 0.1275, 0.1539, 0.1204, 0.08189999999999997]
# 10 reps
rep_time_mean = [6.374945640563965, 11.443684101104736, 31.94684267044067,
                 54.96918201446533, 72.01786518096924, 115.74969530105591, 169.36160802841187]
rep_time_var = [0.08359906442299235, 0.2098502167271988, 0.7995897612304986,
                1.423719273849906, 2.3170759618692274, 4.50685141325959, 6.433015573772143]
rep_conv_mean = [0.91, 0.88, 0.76, 0.68, 0.7, 0.56, 0.45]
rep_conv_var = [0.08190000000000001, 0.1056, 0.1824, 0.2176,
                0.20999999999999996, 0.24640000000000004, 0.2475000000000001]
rep_optimal_mean = [0.89, 0.88, 0.71, 0.54, 0.57, 0.48, 0.37]
rep_optimal_var = [0.09790000000000001, 0.1056, 0.2059, 0.24839999999999993,
                   0.2450999999999999, 0.24959999999999993, 0.23310000000000003]

# New interior point method
ipm_time_mean = [7.8603172302246085, 15.662398338317871, 35.950140953063965,
                 50.873379707336426, 67.74871349334717, 90.0473690032959, 131.45367622375488]
ipm_time_var = [0.003974978963378817, 0.0098839142629231, 0.22411141903987755,
                0.43270459708785436, 0.6003010039677292, 1.1187105300836264, 1.2681023916421508]
ipm_conv_mean = [0.86, 0.88, 0.66, 0.5, 0.56, 0.5, 0.3]
ipm_conv_var = [0.1204, 0.1056, 0.22440000000000002,
                0.25, 0.24639999999999998, 0.25, 0.20999999999999996]
ipm_optimal_mean = [0.74, 0.88, 0.66, 0.5, 0.56, 0.5, 0.3]
ipm_optimal_var = [0.1924, 0.1056, 0.22440000000000002,
                   0.25, 0.24639999999999998, 0.25, 0.20999999999999996]

# --- Error Data Configuration ---
# NOTE: The provided variance in your snippet was for TIME, not RATES.
# You must calculate the Standard Deviation (not variance) for your rates
# and replace the dummy zeros below.
# Example: no_rep_conv_err = np.sqrt(your_rate_variancetor)


# Mapped as:
# [No Rep Conv (Light), No Rep Opt (Dark), 10 Rep Conv (Light), 10 Rep Opt (Dark)]
colors = [
    "#a6cee3",  # Light Blue
    "#1f78b4",  # Dark Blue
    "#fdbf6f",  # Light Orange
    "#ff7f00",  # Dark Orange
    "#89fd6f",  # Light Green
    "#1cb508",  # Dark Green
]


# --- IEEE Styling Setup ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 12
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# --- Plotting ---
fig, ax = plt.subplots(figsize=(10, 6))

# Bar layout configuration
bar_width = 0.1
x = np.arange(len(cars))

# Calculate positions for 6 bars centered on the tick
pos1 = x - 2.5 * bar_width
pos2 = x - 1.5 * bar_width
pos3 = x - 0.5 * bar_width
pos4 = x + 0.5 * bar_width
pos5 = x + 1.5 * bar_width
pos6 = x + 2.5 * bar_width

# --- Plot Bars ---
# Using patterns (hatching) helps distinguish bars in B&W print

# 1. No Repeat: Convergence (Actual)
rects1 = ax.bar(pos1, conv_mean, bar_width, yerr=conv_var, capsize=3,
                label='Converge Rate (1 run)', color=colors[0])

# 2. No Repeat: Optimal
rects2 = ax.bar(pos2, optimal_mean, bar_width, yerr=optimal_var, capsize=3,
                label='Optimal Rate (1 run)', color=colors[1])

# 3. 10 Repeat: Convergence (Actual)
rects3 = ax.bar(pos3, rep_conv_mean, bar_width, yerr=rep_conv_var, capsize=3,
                label='Converge Rate (10 run)', color=colors[2])

# 4. 10 Repeat: Optimal
rects4 = ax.bar(pos4, rep_optimal_mean, bar_width, yerr=rep_optimal_var, capsize=3,
                label='Optimal Rate (10 run)', color=colors[3])

# 3. IPM: Convergence (Actual)
rects5 = ax.bar(pos5, rep_conv_mean, bar_width, yerr=rep_conv_var, capsize=3,
                label='Converge Rate (10 run)', color=colors[4])

# 4. IPM: Optimal
rects6 = ax.bar(pos6, ipm_optimal_mean, bar_width, yerr=ipm_optimal_var, capsize=3,
                label='Optimal Rate (10 run)', color=colors[5])

# --- Add Numerical Labels ---


def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height+0.03),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, rotation=0)


# Add labels
# add_labels(rects1)
# add_labels(rects2)
# add_labels(rects3)
# add_labels(rects4)

# --- Final Formatting ---
ax.set_xlabel('Number of Cars')
ax.set_ylabel('Convergence/Optimal Rate')
ax.set_xticks(x)
ax.set_xticklabels(cars)

# Set Y limit slightly higher to fit labels
ax.set_ylim(0, 1.15)

# Add a subtle grid behind the bars
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
ax.set_axisbelow(True)

# Legend
ax.legend(ncol=2, loc='upper right', frameon=True, fontsize=10)

plt.tight_layout()
plt.show()
