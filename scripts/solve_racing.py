"""Prototype to solve the car racing game with the interior-point game solver."""
import logging

import pickle
import numpy as np

from rd3g.utilities.util import talk
from rd3g.games.car_racing_casadi import create_random_game
from rd3g.solvers.interior_point_game import InteriorPointGame, InteriorPointGameConfig
from rd3g.solvers.algames_julia import AlgamesJulia, AlgamesJuliaConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('rd3g.solvers.interior_point_game')
logger.setLevel(logging.DEBUG)

logger = logging.getLogger('main')
logger.setLevel(logging.INFO)

cpp = True
load = True
variational_gne = True
solver_name = 'interior_point'

if solver_name == 'algames':
    cpp = False

np.random.seed()
game = create_random_game(car_count=4, horizon=10, variational_gne=variational_gne)

gc = game.config
# load from pickle
if load:
    with open('/home/nick/dcsl/buzzracer/outputs/input.p', 'rb') as f:
        data = pickle.load(f)
    game.config = data['gc']
    game.config.__post_init__()
    u_ref = data['u_ref']
else:
    u_ref = np.zeros((gc.m*gc.N, gc.T), order='F')

if solver_name == 'interior_point':
    solver_config = InteriorPointGameConfig(
        inertia_correction=False, iterations=20, variational_gne=variational_gne)
    solver = InteriorPointGame(solver_config, game, cpp_only=cpp)
elif solver_name == 'algames':
    solver_config = AlgamesJuliaConfig(iterations=20)
    solver = AlgamesJulia(solver_config, game)
else:
    raise ValueError(f'Unknown solver_name={solver_name}')

if cpp:
    solver.init_cpp_backend()
    sol = solver.solve_cpp_backend(u_ref)
else:
    sol = solver.solve()

# Say something to grep my attention
text = "Solution Ready"
talk(text)
print(f'{sol.elapsed_time=}, {sol.residual=}')
# solver.visualize(u_ref, None)
# game.inspect_h(sol.u, None, solver)
solver.visualize(sol.u, None)
# if sol.residual < 1e-3:
#     solver.animate(sol.u, None, save_gif=True)
