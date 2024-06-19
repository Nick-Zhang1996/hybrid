# benchmark CarMerge.py, for different total car count
from time import time
import numpy as np
from CarMergeKinematicBicycle import CarMergeKinematicBicycle

car_count_vec = range(2,10)
mean_vec = []
std_vec = []

for car_count in car_count_vec:
    time_vec = []
    for i in range(100):
        main = CarMergeKinematicBicycle(car_count)
        main.silent_mode_enable()
        main.setup()
        t0 = time()
        main.solve(save_gif=False,visualize=False,animate=False)
        time_vec.append(time()-t0)
        print(f'{i}-',end='',flush=True)
    print('')
    mean_vec.append(np.mean(time_vec))
    std_vec.append(np.std(time_vec))
    print(f'car_count {list(car_count_vec)}')
    print(f'mean {mean_vec}')
    print(f'std {std_vec}')




