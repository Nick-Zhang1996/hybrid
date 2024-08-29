# benchmark CarMerge.py, for different total car count
from time import time
import numpy as np
from CarMergeKinematicBicycle import CarMergeKinematicBicycle
#from LQGame_CarMergeKinematicBicycle import LQGame_CarMergeKinematicBicycle as CarMergeKinematicBicycle


#car_count_vec = range(2,10)
car_count_vec = [3,5,8]
mean_vec = []
converged_mean_vec = []
std_vec = []
converge_ratio_vec = []
repeat = 100

for car_count in car_count_vec:
    time_vec = []
    converged_time_vec = []
    converge_count = 0
    for i in range(repeat):
        main = CarMergeKinematicBicycle(car_count)
        main.silent_mode_enable()
        main.setup()
        t0 = time()
        _,_, has_converged = main.solve(save_gif=False,visualize=False,animate=False)
        dt = time()-t0
        time_vec.append(dt)
        if (has_converged):
            converge_count += 1
            converged_time_vec.append(dt)
            print(f'{i}c-',end='',flush=True)
        else:
            print(f'{i}-',end='',flush=True)
    print('')
    converge_ratio_vec.append(converge_count/repeat)
    mean_vec.append(np.mean(time_vec))
    converged_mean_vec.append(np.mean(converged_time_vec))
    std_vec.append(np.std(time_vec))
    '''
    print(f'car_count {list(car_count_vec)}')
    print(f'mean {mean_vec}')
    print(f'std {std_vec}')
    print(f'converge_ratio {converge_ratio_vec}')
    '''
    for i in range(len(car_count_vec)):
        print(f' car count: {car_count_vec[i]} mean {mean_vec[i]} ({std_vec[i]}) conerge ratio: {converge_ratio_vec[i]} converge mean {converged_mean_vec[i]}')
        if (car_count_vec[i] == car_count):
            break




