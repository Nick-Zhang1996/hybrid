""" Prototype to solve merge game with RD3G CasADi """
import logging
import numpy as np

from rd3g.games.car_merge_kinematic_bicycle_casadi import create_random_game
from rd3g.solvers.rd3g_casadi import RD3GCasadi, RD3GCasadiConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('rd3g.solvers.rd3g_casadi')
logger.setLevel(logging.DEBUG)

logger = logging.getLogger('main')
logger.setLevel(logging.INFO)

np.random.seed(4)
# np.random.seed()
cpp = True
game = create_random_game(car_count=2, horizon=20)
solver_config = RD3GCasadiConfig(inertia_correction=True)
solver = RD3GCasadi(solver_config, game, cpp_only=cpp)
if cpp:
    solver.init_cpp_backend()
    sol = solver.solve_cpp_backend()
else:
    sol = solver.solve()
# solver.visualize(sol.u)
# solver.animate(sol.u)
# solver.final()
logger.info(f'{sol.iterations=}, {sol.elapsed_time=:.6f},'
            f'{sol.residual=:.6f} {sol.is_optimal=}, '
            f'{sol.has_converged=}')
