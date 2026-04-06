import logging
from time import time

import jax

from rd3g.games.car_merge_kinematic_bicycle_jax import create_random_game
from rd3g.solvers.rd3g_jax import RD3GJax, RD3GJaxConfig

logging.basicConfig(level=logging.INFO)

game = create_random_game()
solver_config = RD3GJaxConfig(iterations=20, backtracking_max_iter=10)
solver = RD3GJax(solver_config, game)

solver.jax_compile()  # 52 -> 50 -> 12.5 -> 6s

sol = solver.solve()  # 3.5 -> 2.8 -> 1.8 -> 1.9s
print(f'{sol.elapsed_time=}, {sol.has_converged=}, {sol.residual=}')
# game.visualize(game.x0, sol.u, sol.x, show=True, save=False)
# game.animate(game.x0, sol.u, sol.x, show=True, save=False)
solver.final()
