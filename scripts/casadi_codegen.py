""" Generate c code for CasADi symbolic game functions """
import os

from casadi import *

from rd3g.utilities.util import BASEDIR
from rd3g.games.car_merge_kinematic_bicycle_casadi import CarMergeKinematicBicycleCasadi
from rd3g.games.car_merge_kinematic_bicycle_casadi import create_random_game
from rd3g.solvers.rd3g_casadi import RD3GCasadi, RD3GCasadiConfig


# CasADi expects fixed dimension, so the exact game config needs to be given apriori
# For any specific game, a separate source file is created for each pair of (T, N)
# This example shows generating src file for one specific pair of (T,N)

game = create_random_game(car_count=3, horizon=20)
config = game.config
solver_config = RD3GCasadiConfig()
solver = RD3GCasadi(solver_config, game)

N = game.config.N
n = game.config.n
m = game.config.m
T = game.config.T

x = SX.sym('x', N*n, T)
u = SX.sym('u', N*m, T)
lamda = SX.sym('lamda', N*n, T)
mu = SX.sym('mu', N*N, T)


config_params = [game.config.get_int_param_sx(), game.config.get_double_param_sx()]
args = [x, u, lamda, mu]
y = vertcat(*[vec(val) for val in args])

get_n_fun = Function('get_n', [], [config.n])
get_m_fun = Function('get_m', [], [config.m])

r = solver.r(*args)
r_fun = Function('r', args+config_params, [r])
dr_dy = jacobian(r, y)
dr_dy_fun = Function('dr_dy', args+config_params, [dr_dy])


# Switch working directory
codegen_dir = os.path.join(BASEDIR, 'rd3g', 'src', 'games', 'casadi_codegen')
if not os.path.exists(codegen_dir):
    os.makedirs(codegen_dir)
old_cwd = os.getcwd()
os.chdir(codegen_dir)

module_name = game.__module__.split('.')[-1]
cg = CodeGenerator(f'{module_name}_N{config.N}_T{config.T}.c',
                   {'with_header': True})
cg.add(r_fun)
cg.add(dr_dy_fun)
cg.add(get_n_fun)
cg.add(get_m_fun)
cg.generate()

os.chdir(old_cwd)
