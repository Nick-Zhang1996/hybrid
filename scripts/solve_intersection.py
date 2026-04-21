""" Prototype to solve intersection game with RD3G CasADi """
from time import time
import logging
import numpy as np

from rd3g.games.intersection_casadi import create_random_game
from rd3g.solvers.rd3g_casadi import RD3GCasadi, RD3GCasadiConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('rd3g.solvers.rd3g_casadi')
logger.setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

i = 10
np.random.seed(i)
cpp = False
variational_gne = False

logger.info('Creating game...')
game = create_random_game(car_count=5, horizon=20, variational_gne=variational_gne)
logger.info('Setting up solver...')
solver_config = RD3GCasadiConfig(
    iterations=50, inertia_correction=False, variational_gne=variational_gne)
solver = RD3GCasadi(solver_config, game, cpp_only=cpp)
if cpp:
    solver.init_cpp_backend()
    sol = solver.solve_cpp_backend()
else:
    sol = solver.solve()
# solver.final()
u_norm = np.linalg.norm(sol.u)
logger.info(f'seed {i}')
logger.info(f'{sol.iterations=}, {sol.elapsed_time=:.6f},'
            f'{sol.residual=:.6f} {sol.is_optimal=}, '
            f'{sol.has_converged=}, {u_norm=}')
solver.visualize(sol.u)
# logger.info('Preparing gif.')
# solver.animate(sol.u, save_gif=True, save_snapshots=False)
