""" Generate c code for CasADi symbolic game functions """
import os

from casadi import *

from rd3g.utilities.util import BASEDIR
from rd3g.games.car_merge_kinematic_bicycle_casadi import CarMergeKinematicBicycleCasadi
from rd3g.games.car_merge_kinematic_bicycle_casadi import create_random_game

# Generate key functions

# Dimension needs to be fixed for codegen, so the game config needs to be given apriori
# However, for any specific game, a separate source file is created for each pair of (T, N)
# This example shows generating src file for one specific pair of (T,N)

game = create_random_game(car_count=3, horizon=20)
config = game.config

x_k = SX.sym('x_k', config.N, config.n)
x_k_i = SX.sym('x_k_i', config.n)
x_k_j = SX.sym('x_k_j', config.n)
u_k_i = SX.sym('u_k_i', config.m)
i_onehot = SX.sym('i_onehot', config.N)


# J(x_k, u_k_i, i_onehot)
J = Function('J', [x_k, u_k_i, i_onehot, config.get_int_param_sx(),
             config.get_double_param_sx()], [game.J(x_k, u_k_i, i_onehot)])

# Jfi(x_T, i_onehot)
Jfi = Function('Jfi', [x_k, i_onehot, config.get_int_param_sx(),
                       config.get_double_param_sx()], [game.Jfi(x_k, i_onehot)])
# f(x_k_i, u_k_i, i_onehot)
f = Function('f', [x_k_i, u_k_i, i_onehot, config.get_int_param_sx(),
                   config.get_double_param_sx()], [game.f(x_k_i, u_k_i, i_onehot)])
# h(x_i, x_j)
h = Function('h', [x_k_i, x_k_j, config.get_int_param_sx(),
                   config.get_double_param_sx()], [game.h(x_k_i, x_k_j)])


# Switch working directory
codegen_dir = os.path.join(BASEDIR, 'rd3g', 'src', 'games', 'casadi_codegen')
if not os.path.exists(codegen_dir):
    os.makedirs(codegen_dir)
old_cwd = os.getcwd()
os.chdir(codegen_dir)

cg = CodeGenerator(f'car_merge_kinematic_bicycle_casadi_N{config.N}_T{config.T}.c',
                   {'with_header': True})
cg.add(J)
cg.add(Jfi)
cg.add(f)
cg.add(h)
cg.generate()

os.chdir(old_cwd)
