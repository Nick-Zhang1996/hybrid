"""Prototype to solve the rocket landing game with the interior-point solver."""
import logging
import numpy as np

from rd3g.games.rocket_landing_casadi import create_random_game
from rd3g.solvers.interior_point_game import InteriorPointGame, InteriorPointGameConfig
from rd3g.solvers.ilqgame import ILQGame, ILQGameConfig
from rd3g.solvers.ipopt_casadi import IpoptCasadi, IpoptCasadiConfig
from rd3g.solvers.newton_casadi import NewtonCasadi, NewtonCasadiConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('rd3g.solvers.interior_point_game')
logger.setLevel(logging.DEBUG)

logger = logging.getLogger('main')
logger.setLevel(logging.INFO)

np.random.seed(1)
cpp = False
variational_gne = True
solver_name = 'interior_point'

game = create_random_game(horizon=20, variational_gne=variational_gne)
u_ref = game.initial_control_guess()

if solver_name == 'ilqgame':
    solver_config = ILQGameConfig(
        variational_gne=variational_gne,
        iterations=20,
        step_size=0.5,
        barrier_weight=1e-2,
    )
    solver = ILQGame(solver_config, game, cpp_only=cpp)
elif solver_name == 'ipopt':
    solver_config = IpoptCasadiConfig(variational_gne=variational_gne)
    solver = IpoptCasadi(solver_config, game, cpp_only=cpp)
elif solver_name == 'newton':
    solver_config = NewtonCasadiConfig(variational_gne=variational_gne)
    solver = NewtonCasadi(solver_config, game, cpp_only=cpp)
elif solver_name == 'interior_point':
    solver_config = InteriorPointGameConfig(
        iterations=40,
        inertia_correction=False,
        variational_gne=variational_gne,
        abs_split=False,
    )
    solver = InteriorPointGame(solver_config, game, cpp_only=cpp)
else:
    raise ValueError(f'Unknown solver_name={solver_name}')

if cpp:
    solver.init_cpp_backend()
    sol = solver.solve_cpp_backend(u_ref)
else:
    sol = solver.solve(u_ref)

logger.info(f'{sol.iterations=}, {sol.elapsed_time=:.6f},'
            f'{sol.residual=:.6f} {sol.is_optimal=}, '
            f'{sol.has_converged=}')
solver.visualize(sol.u)
solver.final()
