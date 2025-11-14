import logging
from rd3g.games.car_merge_kinematic_bicycle_jax import create_random_game
from rd3g.solvers.rd3g_jax import RD3GJax, RD3GJaxConfig

logging.basicConfig(level=logging.INFO)

game = create_random_game()
solver_config = RD3GJaxConfig()
solver = RD3GJax(solver_config, game)
sol = solver.solve()
print(f'{sol.elapsed_time=}, {sol.has_converged=}, {sol.residual=}')
# game.visualize(game.x0, sol.u, sol.x, show=True, save=False)
# game.animate(game.x0, sol.u, sol.x, show=True, save=False)
solver.final()
