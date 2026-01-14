import logging
import numpy as np

from rd3g.games.car_merge_kinematic_bicycle import create_random_game
from rd3g.solvers.rd3g import RD3G, RD3GConfig

logger = logging.getLogger('main')
logger.setLevel(logging.INFO)

converge_vec = []
optimal_vec = []
dt_vec = []
for i in range(100):
    np.random.seed(i)
    game = create_random_game(car_count=5, horizon = 20)
    solver_config = RD3GConfig(USE_CPP=True)
    solver = RD3G(solver_config, game)
    sol = solver.solve()
    # solver.final()
    # print(f'{sol.elapsed_time=}, {sol.has_converged=}, {sol.residual=}')
    # game.visualize(game.x0, sol.u, sol.x, show=True, save=False)
    # game.animate(game.x0, sol.u, sol.x, show=True, save=False)
    # solver.final()
    converge_vec.append(sol.has_converged)
    optimal_vec.append(sol.is_optimal)
    dt_vec.append(sol.elapsed_time)
    logger.info(
        f'{i=}, {sol.iterations=}, {sol.elapsed_time=:.6f}, {sol.residual=:.6f} {sol.is_optimal=}')

    if i % 10 == 9:
        convergence_rate = np.mean(converge_vec)
        optimal_rate = np.mean(optimal_vec)
        median_dt = np.median(dt_vec)
        mean_dt = np.mean(dt_vec)
        logger.info(f'{convergence_rate=}, {optimal_rate=}, {mean_dt=}, {median_dt=}')
