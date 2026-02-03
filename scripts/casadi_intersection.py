""" Prototype to solve intersection game with RD3G CasADi """
from time import time
import logging
import numpy as np

from rd3g.games.intersection_casadi import create_random_game
from rd3g.solvers.rd3g_casadi import RD3GCasadi, RD3GCasadiConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('rd3g.solvers.rd3g_casadi')
logger.setLevel(logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

np.random.seed(4)
logger.info('Creating game...')
game = create_random_game(car_count=5, horizon=40)
logger.info('Setting up solver...')
solver_config = RD3GCasadiConfig()
solver = RD3GCasadi(solver_config, game, cpp_only=False)
sol = solver.solve()
guess = np.zeros((game.m, game.N, game.T), order='F')
logger.info('visualizaing...')
solver.animate(sol.u)
# solver.visualize(guess)
# solver.animate(guess)
