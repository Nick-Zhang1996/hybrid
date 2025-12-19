""" Prototype to solve merge game with RD3G CasADi """
from rd3g.games.car_merge_kinematic_bicycle_casadi import create_random_game
from rd3g.solvers.rd3g_casadi import RD3GCasadi, RD3GCasadiConfig

game = create_random_game(car_count=3, horizon=20)
solver_config = RD3GCasadiConfig()
solver = RD3GCasadi(solver_config, game)

solver.solve()
