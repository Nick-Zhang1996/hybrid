""" Profile time performance of SteinMerge"""
import time
import logging
import numpy as np
from examples.stein_merge import SteinMerge

logging.basicConfig(
    filename='logs/profile_stein_merge.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("ProfileSteinMerge")
logger.setLevel(logging.DEBUG)

if __name__ == "__main__":
    data = []
    for car_count in range(2, 8):
        # one record for each experiment
        # (car_count, solution_time_mean, solution_time_std)
        solution_time_vec = []
        converged_solution_time_vec = []
        for i in range(50):
            # create a random initial condition
            main = SteinMerge(car_count=car_count)
            main.silent_mode_enable()
            main.setup()
            t0 = time.time()
            retval = main.solve(save_gif=False, visualize=False, animate=False)
            u_ref, full_x_ref, has_converged = retval
            dt = time.time() - t0
            solution_time_vec.append(dt)
            if (has_converged):
                converged_solution_time_vec.append(dt)
            logger.debug(f'{car_count=}, {i=} {dt=} {has_converged=}')
        mean_solution_time = np.mean(solution_time_vec)
        std_solution_time = np.std(solution_time_vec)
        converge_ratio = len(converged_solution_time_vec) / len(
            solution_time_vec)
        mean_converged_solution_time = np.mean(converged_solution_time_vec)
        std_converged_solution_time = np.std(converged_solution_time_vec)
        data.append((car_count, mean_solution_time, std_solution_time))
        logger.info(
            f'{car_count=},{mean_solution_time=:.4f}, {std_solution_time=:.4f}'
        )
        logger.info(
            f'{car_count=},{mean_converged_solution_time=:.4f}, {std_converged_solution_time=:.4f}'
            f'{converge_ratio=}')
