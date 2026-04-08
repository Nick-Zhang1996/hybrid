""" Prototype to solve car racing game with RD3G CasADi """
from dataclasses import replace
import logging

import pickle
import numpy as np

from rd3g.utilities.util import talk
from rd3g.games.car_racing_casadi import create_random_game
from rd3g.solvers.rd3g_casadi import RD3GCasadi, RD3GCasadiConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('rd3g.solvers.rd3g_casadi')
logger.setLevel(logging.DEBUG)

logger = logging.getLogger('main')
logger.setLevel(logging.INFO)

np.random.seed(2)
cpp = False
solver_config = RD3GCasadiConfig(inertia_correction=False, iterations=20)
game = create_random_game(car_count=4, horizon=20)

gc = game.config
# load from pickle
load = False
if load:
    with open('outputs/input.p', 'rb') as f:
        data = pickle.load(f)
    game.config = replace(gc, x0=data['x0'], target_x_ref=data['target_x_ref'])
    u_ref = data['u_ref']

solver = RD3GCasadi(solver_config, game, cpp_only=cpp)
u_ref = np.zeros((gc.m*gc.N, gc.T), order='F')
solver.visualize(u_ref, None)


if cpp:
    solver.init_cpp_backend()
    sol = solver.solve_cpp_backend(u_ref)
else:
    sol = solver.solve()

# Say something to grep my attention
text = "Solution Ready"
talk(text)
print(f'{sol.elapsed_time=}, {sol.residual=}')
# DEBUG
# game.inspect_h(sol.u, None, solver)
solver.visualize(sol.u, None)
# solver.animate(sol.u, None, save_gif=False)
