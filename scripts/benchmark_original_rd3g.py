import logging
import numpy as np

from rd3g.games.car_merge_kinematic_bicycle import create_random_game
from rd3g.solvers.rd3g import RD3G, RD3GConfig

logger = logging.getLogger('main')
logger.setLevel(logging.INFO)

good_u_vec = []
car_conv_mean_vec = []
car_conv_var_vec = []
car_optimal_mean_vec = []
car_optimal_var_vec = []
car_time_mean_vec = []
car_time_var_vec = []
for car_count in range(2, 9):
    converge_vec = []
    optimal_vec = []
    dt_vec = []
    repeat_count = 50
    for i in range(repeat_count):
        np.random.seed(i)
        game = create_random_game(car_count=car_count, horizon=20)
        solver_config = RD3GConfig(USE_CPP=True)
        solver = RD3G(solver_config, game)
        sol = solver.solve()

        # solver.final()
        converge_vec.append(sol.has_converged)
        optimal_vec.append(sol.is_optimal and sol.has_converged)
        dt_vec.append(sol.elapsed_time)
        logger.debug(
            f'run {i}, {sol.iterations=}, {sol.elapsed_time=:.6f}, {sol.residual=:.6f} {sol.has_converged=}, {sol.is_optimal=}')
        if sol.is_optimal and sol.has_converged:
            good_u_vec.append(sol.u)
        print(f'repeat {i + 1}/{repeat_count}', end='\r', flush=True)

        if i % 10 == 9:
            convergence_rate = np.mean(converge_vec)
            optimal_rate = np.mean(optimal_vec)
            median_dt = np.median(dt_vec)
            mean_dt = np.mean(dt_vec)
            logger.debug(
                f'{convergence_rate=}, {optimal_rate=}, {mean_dt*1000=:.1f}ms, {median_dt*1000=:.1f}ms')
    print()
    converge = np.mean(converge_vec)
    optimal = np.mean(optimal_vec)
    median_dt_ms = np.median(dt_vec) * 1000
    mean_dt_ms = np.mean(dt_vec) * 1000
    var_dt_ms = np.var(dt_vec) * 1000
    car_time_mean_vec.append(mean_dt_ms)
    car_time_var_vec.append(var_dt_ms)
    car_conv_mean_vec.append(converge)
    car_optimal_mean_vec.append(optimal)
    car_conv_var_vec.append(np.var(converge_vec))
    car_optimal_var_vec.append(np.var(optimal_vec))
    logger.info(
        f'{car_count} cars {converge=}, {optimal=}, {mean_dt_ms=:.1f}ms, {var_dt_ms=:.1f}ms')

# Report mean and covariance of optimal results, used as param for initial guess
# stacked_u = np.hstack([val.reshape(game.m, game.N*game.T, order='F') for val in good_u_vec])
# mean = np.mean(stacked_u, axis=1)
# cov = np.cov(stacked_u)
# logger.info(f'{mean=}, {cov=}')
print(f'{car_time_mean_vec=}')
print(f'{car_time_var_vec=}')
print(f'{car_conv_mean_vec=}')
print(f'{car_conv_var_vec=}')
print(f'{car_optimal_mean_vec=}')
print(f'{car_optimal_var_vec=}')
