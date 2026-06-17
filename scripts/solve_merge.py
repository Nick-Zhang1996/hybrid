"""Prototype to solve the merge game with the interior-point game solver."""
import logging
import numpy as np

from rd3g.games.car_merge_kinematic_bicycle_casadi import create_random_game
from rd3g.solvers.interior_point_game import InteriorPointGame, InteriorPointGameConfig
from rd3g.solvers.ilqgame import ILQGame, ILQGameConfig
from rd3g.solvers.ipopt_casadi import IpoptCasadi, IpoptCasadiConfig
from rd3g.solvers.newton_casadi import NewtonCasadi, NewtonCasadiConfig
from rd3g.solvers.rd3g_casadi import RD3GCasadi, RD3GCasadiConfig
from rd3g.solvers.algames_julia import AlgamesJulia, AlgamesJuliaConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('rd3g.solvers.interior_point_game')
logger.setLevel(logging.DEBUG)

logger = logging.getLogger('main')
logger.setLevel(logging.INFO)

np.random.seed(12)
cpp = False
variational_gne = True
game = create_random_game(car_count=3, horizon=20, variational_gne=variational_gne)
solver_name = 'algames'
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
        inertia_correction=False,
        variational_gne=variational_gne,
        abs_split=True)
    solver = InteriorPointGame(solver_config, game, cpp_only=cpp)
elif solver_name == 'rd3g':
    solver_config = RD3GCasadiConfig(inertia_correction=False)
    solver = RD3GCasadi(solver_config, game, cpp_only=cpp)
elif solver_name == 'algames':
    if cpp:
        raise ValueError('algames does not support cpp=True')
    solver_config = AlgamesJuliaConfig(iterations=20)
    solver = AlgamesJulia(solver_config, game)
else:
    raise ValueError(f'Unknown solver_name={solver_name}')
if cpp:
    solver.init_cpp_backend()
    sol = solver.solve_cpp_backend()
else:
    sol = solver.solve()
# solver.final()
logger.info(f'{sol.iterations=}, {sol.elapsed_time=:.6f},'
            f'{sol.residual=:.6f} {sol.is_optimal=}, '
            f'{sol.has_converged=}')
# solver.animate(sol.u, save_gif=False, save_snapshots=False)
solver.visualize(sol.u)
solver.final()
