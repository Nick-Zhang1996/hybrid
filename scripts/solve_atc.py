"""Prototype to solve the air traffic control game with the interior-point solver."""
import logging
import numpy as np

from rd3g.games.air_traffic_control_casadi import create_random_game
from rd3g.solvers.interior_point_game import InteriorPointGame, InteriorPointGameConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('rd3g.solvers.interior_point_game')
logger.setLevel(logging.DEBUG)

logger = logging.getLogger('main')
logger.setLevel(logging.INFO)

np.random.seed(123421)
cpp = False
variational_gne = True

game = create_random_game(aircraft_count=3, horizon=10, variational_gne=variational_gne)
solver_config = InteriorPointGameConfig(
    inertia_correction=False,
    variational_gne=variational_gne,
    precondition_with_potential=True,
    abs_split=False)
solver = InteriorPointGame(solver_config, game, cpp_only=cpp)

if cpp:
    solver.init_cpp_backend()
    sol = solver.solve_cpp_backend()
else:
    sol = solver.solve()

logger.info(f'{sol.iterations=}, {sol.elapsed_time=:.6f},'
            f'{sol.residual=:.6f} {sol.is_optimal=}, '
            f'{sol.has_converged=}')
solver.visualize(sol.u,)
solver.final()
