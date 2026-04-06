""" Prototype to solve intersection game with RD3G CasADi """
from time import time
import logging
import numpy as np

from rd3g.games.intersection_casadi import create_random_game
from rd3g.solvers.rd3g_casadi import RD3GCasadi, RD3GCasadiConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('rd3g.solvers.rd3g_casadi')
logger.setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

i = 35
np.random.seed()
cpp = True
logger.info('Creating game...')
game = create_random_game(car_count=4, horizon=40)
logger.info('Setting up solver...')
solver_config = RD3GCasadiConfig(iterations=50, inertia_correction=True)
solver = RD3GCasadi(solver_config, game, cpp_only=False)
if cpp:
    solver.init_cpp_backend()
    sol = solver.solve_cpp_backend()
else:
    sol = solver.solve()
# solver.visualize(sol.u)
# solver.final()
logger.info(f'seed {i}')
logger.info(f'{sol.iterations=}, {sol.elapsed_time=:.6f},'
            f'{sol.residual=:.6f} {sol.is_optimal=}, '
            f'{sol.has_converged=}')
# if sol.is_optimal and sol.has_converged:
solver.animate(sol.u, save_gif=False, save_snapshots=False)
