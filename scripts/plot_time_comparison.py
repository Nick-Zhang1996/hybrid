""" Plot Solver runtime comparison"""
import matplotlib.pyplot as plt
import numpy as np

# --- Data Configuration ---
cars = [2, 3, 4, 5, 6, 7, 8]

# Define all 4 solvers in a list of dictionaries for cleaner code
solvers = [
    {
        "name": "RD3G-SOSC",
        "times": [1.5, 4.5, 5.0, 10.7, 13.0, 17.8, 13.9],
        "color": "#ff7f0e", # Orange
        "marker": "^",      # Triangle
        "style": "--",      # Dashed
        "label_pos": "bottom" # Put text below to avoid overlap with Legacy
    },
    {
        "name": "Original RD3G Solver",
        "times": [18,29,83,137,226,339,430],
        "color": "#d62728", # Red
        "marker": "o",      # Circle
        "style": "-",       # Solid
        "label_pos": "top"  # Put text above
    },
    {
        "name": "ALGames",
        "times": [48,95,198,366,653,1103,1833],
        "color": "#2ca02c", # Green
        "marker": "D",      # Diamond
        "style": "-.",      # Dash-dot
        "label_pos": "top"
    },
    {
        "name": "iLQGame",
        "times": [324, 505, 657, 829, 1037, 2253, np.nan],
        "color": "#1f77b4", # Blue
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
        plt.text(x, y + offset, f'{y}ms', 
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
plt.savefig('four_solvers_comparison.png', dpi=300)
plt.show()