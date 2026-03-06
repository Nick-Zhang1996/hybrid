import matplotlib.pyplot as plt
import numpy as np

car_count = [2, 3, 4, 5, 6, 7, 8]
# Convergence and optimality rate with inertia correction
# test_conv = [0.96, 0.84, 0.74, 0.62, 0.66, 0.54, 0.5] # merge
# test_opt = [0.68, 0.48, 0.36, 0.24, 0.22, 0.12, 0.12] # merge

test_conv = [0.88, 0.82, 0.56, 0.54, 0.5, 0.57, 0.46]  # intersection
test_opt = [0.88, 0.82, 0.55, 0.54, 0.49, 0.56, 0.46]  # intersection

# Show this
test_opt_ratio = [test_opt[i]/test_conv[i] for i in range(len(test_conv))]
print(f'{test_opt_ratio=}')
test_ratio = np.array(test_opt_ratio)

# Convergence and optimality rate without inertia correction
# base_conv = [0.94, 0.84, 0.86, 0.84, 0.9, 0.66, 0.76]  # merge
# base_opt = [0.6, 0.44, 0.36, 0.24, 0.22, 0.1, 0.08]  # merge
base_conv = [0.88, 0.83, 0.58, 0.55, 0.54, 0.57, 0.45]  # intersection
base_opt = [0.87, 0.82, 0.55, 0.52, 0.49, 0.53, 0.45]  # intersection

base_opt_ratio = [base_opt[i]/base_conv[i] for i in range(len(base_conv))]
print(f'{base_opt_ratio=}')
base_ratio = np.array(base_opt_ratio)

# RD3G doesn't have notion of optimality, so always 1
ori_conv = [0.88, 0.86, 0.72, 0.64, 0.68, 0.58, 0.58]  # merge
ori_opt = [0.88, 0.86, 0.72, 0.64, 0.68, 0.58, 0.58]  # merge
ori_opt_ratio = [ori_opt[i]/ori_conv[i] for i in range(len(ori_conv))]
ori_ratio = np.array(ori_opt_ratio)


# --- 2. IEEE Formatting Setup ---
# IEEE standard column width is ~3.5 inches.
# We use Times New Roman to match the manuscript text.
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.figsize': (3.5, 2.5),  # Width x Height in inches
    'text.usetex': False,          # Set to True if you have a local LaTeX installation
    'mathtext.fontset': 'stix',    # improved math font rendering
    'lines.linewidth': 1.5,
    'lines.markersize': 6
})

# --- 3. Plotting ---
fig, ax = plt.subplots(constrained_layout=True)

# Plot Baseline
ax.plot(car_count, base_ratio,
        label='Without Inertia Correction)',
        marker='o',          # Circle marker
        linestyle='--',      # Dashed line for baseline
        color='black',       # Black for high contrast
        markerfacecolor='white',
        markeredgewidth=1.0)

# Plot Proposed
ax.plot(car_count, test_ratio,
        label='With Inertia Correction',
        marker='s',          # Square marker
        linestyle='-',       # Solid line for proposed
        color='#0055AA',     # IEEE Blue or distinct dark color
        markeredgecolor='#0055AA')

# Plot Original RD3G
# ax.plot(car_count, ori_ratio,
#         label='RD3G Baseline',
#         marker='s',          # Square marker
#         linestyle='-',       # Solid line for proposed
#         color='#D62728',     # Red
#         markeredgecolor='#D62728')

# --- 4. Styling & labeling ---
ax.set_xlabel('Number of Cars ($N$)')
ax.set_ylabel(r'Optimality Ratio ($\eta_{opt} / \eta_{conv}$)')
ax.set_ybound(lower=0.5, upper=1.1)

# Set integer ticks for x-axis since car counts are discrete
ax.set_xticks(car_count)

# Add grid (standard for engineering plots)
ax.grid(True, which='major', linestyle=':', alpha=0.6)
ax.grid(True, which='minor', linestyle=':', alpha=0.3)
ax.minorticks_on()

# Legend placement - 'best' usually works, but 'upper right' is safe here
ax.legend(loc='best', frameon=True, fancybox=False, edgecolor='black', framealpha=1.0)

# --- 5. Save ---
# Save as PDF (vector graphics) is standard for submission.
# plt.savefig('optimality_ratio_comparison.pdf', format='pdf', dpi=300)
# plt.savefig('optimality_ratio_comparison.png', format='png', dpi=300)  # High-res preview

plt.show()
