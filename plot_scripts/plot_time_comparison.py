""" Plot Solver runtime comparison"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "outputs" / "benchmarks"

# --- Data Configuration ---
cars = [2, 3, 4, 5, 6, 7, 8]

# no repeat
car_time_mean_vec = [1.4612984657287598, 2.510499954223633, 4.480984210968018,
                     6.304130554199219, 8.013310432434082, 12.38377571105957, 17.142696380615234]
car_time_var_vec = [0.0003833598028165852, 0.0012817887454730226, 0.0027443585564640217,
                    0.005764007726520503, 0.00873011579485592, 0.01667516727466136, 0.030778920515649594]
car_conv_vec = [0.91, 0.83, 0.8, 0.74, 0.82, 0.7, 0.66]
car_optimal_vec = [0.61, 0.5, 0.42, 0.27, 0.24, 0.21, 0.13]

# 10 repeats
rep_car_time_mean_vec = [6.355562210083008, 11.503715515136719, 31.42996311187744,
                         54.213383197784424, 72.35618591308594, 118.17569255828857, 168.34531545639038]
rep_car_time_var_vec = [0.08120785370319936, 0.21264537037345688, 0.7765771563383396,
                        1.402769040324864, 2.3564255298216725, 4.795834444910929, 6.427171162734174]
rep_car_conv_vec = [0.91, 0.88, 0.76, 0.68, 0.7, 0.56, 0.45]
rep_car_optimal_vec = [0.94, 0.94, 0.77, 0.66, 0.67, 0.56, 0.51]

# Define all 4 solvers in a list of dictionaries for cleaner code
solvers = [
    {
        "name": "RD3G-SOSC",
        "times": car_time_mean_vec,
        "color": "#ff7f0e",  # Orange
        "marker": "^",      # Triangle
        "style": "--",      # Dashed
        "label_pos": "bottom"  # Put text below to avoid overlap with Legacy
    },
    # {
    #     "name": "RD3G-SOSC 10 rep",
    #     "times": rep_car_time_mean_vec,
    #     "color": "#852e06",  # Dark Orange
    #     "marker": "^",      # Triangle
    #     "style": "--",      # Dashed
    #     "label_pos": "bottom"  # Put text below to avoid overlap with Legacy
    # },
    {
        "name": "Original RD3G Solver",
        "times": [18, 29, 83, 137, 226, 339, 430],
        "color": "#d62728",  # Red
        "marker": "o",      # Circle
        "style": "-",       # Solid
        "label_pos": "top"  # Put text above
    },
    {
        "name": "ALGames",
        "times": [48, 95, 198, 366, 653, 1103, 1833],
        "color": "#2ca02c",  # Green
        "marker": "D",      # Diamond
        "style": "-.",      # Dash-dot
        "label_pos": "top"
    },
    {
        "name": "iLQGame",
        "times": [324, 505, 657, 829, 1037, 2253, np.nan],
        "color": "#1f77b4",  # Blue
        "marker": "s",      # Square
        "style": ":",       # Dotted
        "label_pos": "bottom"
    }
]

# --- Plotting ---
plt.figure(figsize=(12, 8))

for solver in solvers:
    # 1. Plot the line
    plt.plot(cars, solver["times"],
             marker=solver["marker"],
             linestyle=solver["style"],
             color=solver["color"],
             label=solver["name"],
             linewidth=2.5,
             markersize=8)

    # 2. Add the numerical labels
    for x, y in zip(cars, solver["times"]):
        # Determine offset direction
        is_top = (solver["label_pos"] == "top")
        offset = 10 if is_top else -15
        va = 'bottom' if is_top else 'top'

        # Add text with a small white background (bbox) for readability
        plt.text(x, y + offset, f'{y:.0f}ms',
                 ha='center', va=va,
                 fontsize=9,
                 color=solver["color"],
                 fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

# --- Styling ---
plt.title('Performance Benchmark', fontsize=16, pad=20)
plt.xlabel('Number of Cars (Agents)', fontsize=13)
plt.ylabel('Avg Runtime (ms)', fontsize=13)

# Force integer ticks for x-axis
plt.xticks(cars)

# Add a subtle grid
plt.grid(True, linestyle='--', alpha=0.5)

# Add Legend
plt.legend(fontsize=11, loc='upper left')

plt.tight_layout()

# Save and Show
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_DIR / 'four_solvers_comparison.png', dpi=300)
plt.show()
