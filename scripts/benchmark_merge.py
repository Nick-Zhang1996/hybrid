""" Benchmark game with RD3G CasADi """
import logging
import numpy as np

from rd3g.games.car_merge_kinematic_bicycle_casadi import create_random_game
from rd3g.solvers.rd3g_casadi import RD3GCasadi, RD3GCasadiConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('rd3g.solvers.rd3g_casadi')
logger.setLevel(logging.WARNING)

logger = logging.getLogger('main')
logger.setLevel(logging.INFO)

good_u_vec = []
car_conv_mean_vec = []
car_conv_var_vec = []
car_optimal_mean_vec = []
car_optimal_var_vec = []
car_time_mean_vec = []
car_time_var_vec = []
for car_count in range(2, 9, 2):
    converge_vec = []
    optimal_vec = []
    dt_vec = []
    for i in range(50):
        np.random.seed(i)
        game = create_random_game(car_count=car_count, horizon=40)
        solver_config = RD3GCasadiConfig(inertia_correction=False)
        solver = RD3GCasadi(solver_config, game, cpp_only=False)
        sol = solver.solve()
        # solver.init_cpp_backend()
        # sol = solver.solve_cpp_backend()
        # sol = solver.solve_cpp_backend_rand_restart(restarts=10)

        # solver.final()
        converge_vec.append(sol.has_converged)
        optimal_vec.append(sol.is_optimal and sol.has_converged)
        dt_vec.append(sol.elapsed_time)
        logger.debug(
            f'run {i}, {sol.iterations=}, {sol.elapsed_time=:.6f}, {sol.residual=:.6f} {sol.has_converged=}, {sol.is_optimal=}')
        if sol.is_optimal and sol.has_converged:
            good_u_vec.append(sol.u)

        if i % 10 == 9:
            convergence_rate = np.mean(converge_vec)
            optimal_rate = np.mean(optimal_vec)
            median_dt = np.median(dt_vec)
            mean_dt = np.mean(dt_vec)
            logger.debug(
                f'{convergence_rate=}, {optimal_rate=}, {mean_dt*1000=:.1f}ms, {median_dt*1000=:.1f}ms')
    converge = np.mean(converge_vec).item()
    optimal = np.mean(optimal_vec).item()
    median_dt_ms = np.median(dt_vec).item() * 1000
    mean_dt_ms = np.mean(dt_vec).item() * 1000
    var_dt_ms = np.var(dt_vec).item() * 1000
    car_time_mean_vec.append(mean_dt_ms)
    car_time_var_vec.append(var_dt_ms)
    car_conv_mean_vec.append(converge)
    car_optimal_mean_vec.append(optimal)
    car_conv_var_vec.append(np.var(converge_vec).item())
    car_optimal_var_vec.append(np.var(optimal_vec).item())
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
