from rd3g.games.car_merge_kinematic_bicycle import create_random_game
from rd3g.solvers.residual_game import RD3G, RD3GConfig

game = create_random_game()
solver_config = RD3GConfig()
solver = RD3G(solver_config, game)
solver.solve()
