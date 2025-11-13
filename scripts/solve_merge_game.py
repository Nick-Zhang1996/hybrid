from rd3g.games.car_merge_kinematic_bicycle import create_random_game
from rd3g.solvers.residual_game import RD3G, RD3GConfig

game = create_random_game()
solver_config = RD3GConfig(USE_CPP=True)
solver = RD3G(solver_config, game)
sol = solver.solve()
print(f'{sol.elapsed_time=}, {sol.has_converged=}')
# game.visualize(game.x0, sol.u, sol.x, show=True, save=False)
game.animate(game.x0, sol.u, sol.x, show=True, save=False)
