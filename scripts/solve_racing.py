""" Prototype to solve car racing game with RD3G CasADi """
import logging
import sys

import numpy as np
# import pyttsx3

from rd3g.games.car_racing_casadi import create_random_game
from rd3g.solvers.rd3g_casadi import RD3GCasadi, RD3GCasadiConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('rd3g.solvers.rd3g_casadi')
logger.setLevel(logging.DEBUG)

logger = logging.getLogger('main')
logger.setLevel(logging.INFO)

np.random.seed(5)
cpp = False
solver_config = RD3GCasadiConfig(inertia_correction=False, iterations=20)
game = create_random_game(car_count=4, horizon=20)
solver = RD3GCasadi(solver_config, game, cpp_only=False)
if cpp:
    solver.init_cpp_backend()
    sol = solver.solve_cpp_backend()
else:
    sol = solver.solve()

# Say something to grep my attention
# engine = pyttsx3.init()
# engine.setProperty('rate', 150)  # Speed in words per minute
# text = "Solution Ready"
# engine.say(text)
# engine.runAndWait()

m = game.config.m
N = game.config.N
T = game.config.T
u = np.zeros((m, N, T))
# x = np.zeros((n, N, T+1))
# solver.visualize(sol.u)
