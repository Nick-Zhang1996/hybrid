""" Benchmark game with RD3G CasADi """
import logging
import numpy as np

from rd3g.games.car_merge_kinematic_bicycle_casadi import create_random_game
from rd3g.solvers.rd3g_casadi import RD3GCasadi, RD3GCasadiConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('RD3G_CasADi')
logger.setLevel(logging.WARNING)

logger = logging.getLogger('main')
logger.setLevel(logging.INFO)

converge_vec = []
optimal_vec = []
dt_vec = []
good_u_vec = []
for i in range(100):
    np.random.seed(i)
    game = create_random_game(car_count=5, horizon=40)
    solver_config = RD3GCasadiConfig()
    solver = RD3GCasadi(solver_config, game, cpp_only=True)
    # sol = solver.solve()
    solver.init_cpp_backend()
    # sol = solver.solve_cpp_backend()
    sol = solver.solve_cpp_backend_rand_restart(restarts=100)

    # solver.final()
    converge_vec.append(sol.has_converged)
    optimal_vec.append(sol.is_optimal)
    dt_vec.append(sol.elapsed_time)
    logger.info(
        f'{i=}, {sol.iterations=}, {sol.elapsed_time=:.6f}, {sol.residual=:.6f} {sol.is_optimal=}')
    if sol.is_optimal and sol.has_converged:
        good_u_vec.append(sol.u)

    if i % 10 == 9:
        convergence_rate = np.mean(converge_vec)
        optimal_rate = np.mean(optimal_vec)
        median_dt = np.median(dt_vec)
        mean_dt = np.mean(dt_vec)
        logger.info(f'{convergence_rate=}, {optimal_rate=}, {mean_dt=}, {median_dt=}')

# Report mean and covariance of optimal results, used as param for initial guess
stacked_u = np.hstack([val.reshape(game.m, game.N*game.T, order='F') for val in good_u_vec])
mean = np.mean(stacked_u, axis=1)
cov = np.cov(stacked_u)
logger.info(f'{mean=}, {cov=}')
