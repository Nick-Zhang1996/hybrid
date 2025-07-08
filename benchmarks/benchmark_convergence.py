""" benchmark linear convergence of 10 car"""
from time import time
import pickle
import numpy as np
from examples.car_merge_kinematic_bicycle import CarMergeKinematicBicycle
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)
'''
y = np.sin(x)  # Example line
confidence_interval = 0.3  # Example confidence interval

# Compute the upper and lower bounds of the confidence interval
y_upper = y + confidence_interval
y_lower = y - confidence_interval

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Line')
plt.fill_between(x, y_lower, y_upper, color='b', alpha=0.2, 
                label='Confidence Interval',edgecolor='none')
plt.show()
'''

car_count = 10
mean_vec = []
std_vec = []
residual_vec_vec = []
# TODO if diverge, skip

time_vec = []
for i in range(20):
    main = CarMergeKinematicBicycle(car_count)
    #main.silent_mode_enable()
    main.setup()
    t0 = time()
    main.solve(save_gif=False, visualize=False, animate=False)
    residual_vec_vec.append(main.residual_vec)
    time_vec.append(time() - t0)
    print(f'{i}-', end='', flush=True)
print('')

diverge_count = 0

for residual_vec in residual_vec_vec:
    if (len(residual_vec) == 50):
        plt.plot(residual_vec, '-')
    else:
        diverge_count += 1

with open('convergence.p', 'wb') as f:
    pickle.dump(residual_vec_vec, f)

print(f'diverge count {diverge_count}')
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Residual (exp)')
plt.show()
