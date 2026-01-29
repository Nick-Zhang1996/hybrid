""" Prototype to solve merge game with RD3G CasADi """
from time import time
import logging
import numpy as np

from rd3g.games.car_merge_kinematic_bicycle_casadi import create_random_game
from rd3g.solvers.rd3g_casadi import RD3GCasadi, RD3GCasadiConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('RD3G_CasADi')
logger.setLevel(logging.WARNING)

logger = logging.getLogger('main')
logger.setLevel(logging.INFO)

np.random.seed(3)
game = create_random_game(car_count=5, horizon=20)
solver_config = RD3GCasadiConfig()
solver = RD3GCasadi(solver_config, game, cpp_only=True)
# sol = solver.solve()
# solver.visualize(sol.u)
# solver.animate(sol.u)
solver.init_cpp_backend()
sol = solver.solve_cpp_backend()

# solver.final()
logger.info(f'{sol.iterations=}, {sol.elapsed_time=:.6f},'
            f'{sol.residual=:.6f} {sol.is_optimal=}, '
            f'{sol.has_converged=}')
