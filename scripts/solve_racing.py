""" Prototype to solve car racing game with RD3G CasADi """
import logging

import numpy as np

from rd3g.utilities.util import talk
from rd3g.games.car_racing_casadi import create_random_game
from rd3g.solvers.rd3g_casadi import RD3GCasadi, RD3GCasadiConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('rd3g.solvers.rd3g_casadi')
logger.setLevel(logging.DEBUG)

logger = logging.getLogger('main')
logger.setLevel(logging.INFO)

np.random.seed()
cpp = True
solver_config = RD3GCasadiConfig(inertia_correction=False, iterations=20)
game = create_random_game(car_count=4, horizon=20)
solver = RD3GCasadi(solver_config, game, cpp_only=False)
gc = game.config
# u_ref = np.zeros((gc.m*gc.N, gc.T), order='F')
# solver.visualize(u_ref)
if cpp:
    solver.init_cpp_backend()
    sol = solver.solve_cpp_backend()
else:
    sol = solver.solve()

# Say something to grep my attention
text = "Solution Ready"
talk(text)
print(f'{sol.elapsed_time=}')

solver.animate(sol.u)
