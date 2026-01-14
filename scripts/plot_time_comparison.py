""" Plot Solver runtime comparison"""
import matplotlib.pyplot as plt
import numpy as np

# --- Data Configuration ---
# Replace these lists with your actual benchmark results
num_cars = [2, 3, 4, 5, 6, 7, 8]

# Time in milliseconds
time_rd3g_solver = [18,29,83,137,226,339,430]
time_ldl_solver = [2,3,4,5,6,7,8]

# --- Plotting ---
plt.figure(figsize=(10, 6))

# Plot Old Solver (Red line with circles)
plt.plot(num_cars, time_rd3g_solver, marker='o', linestyle='-', color='#d62728', 
         label='Original RD3G', linewidth=2, markersize=8)

# Plot New Solver (Blue dashed line with squares)
plt.plot(num_cars, time_ldl_solver, marker='s', linestyle='--', color='#1f77b4', 
         label='RD3C-SOSC', linewidth=2, markersize=8)

# --- Add Numerical Labels ---
# Offset determines how far above/below the point the text appears
def add_labels(x_data, y_data, color, vertical_offset=5, is_top=True):
    for x, y in zip(x_data, y_data):
        va = 'bottom' if is_top else 'top'
        offset = vertical_offset if is_top else -vertical_offset
        plt.text(x, y + offset, f'{y} ms', ha='center', va=va, 
                 fontsize=10, color=color, fontweight='bold')

add_labels(num_cars, time_rd3g_solver, '#d62728', vertical_offset=5, is_top=True)
add_labels(num_cars, time_ldl_solver, '#1f77b4', vertical_offset=8, is_top=False)

# --- Styling ---
plt.title('Solver Benchmark: Computation Time vs Number of Agents', fontsize=14, pad=15)
plt.xlabel('Number of Cars', fontsize=12)
plt.ylabel('Computation Time (ms)', fontsize=12)
plt.xticks(num_cars)  # Ensure x-axis only shows integer car numbers
plt.grid(True, linestyle=':', alpha=0.6)  # Light dotted grid
plt.legend(fontsize=12, loc='upper left')

# Adjust layout to prevent clipping
plt.tight_layout()

# Save and Show
plt.savefig('solver_comparison.png', dpi=300)
plt.show()