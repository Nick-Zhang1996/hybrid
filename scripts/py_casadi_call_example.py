""" Toy example to 
1. Generate c code for CasADi symbolic game
2. Load compiled game/solver dll (.so)
3. Call compiled residual function with numpy array
4. Reconstruct csc matrix from returned value from compiled residual fun
"""
from time import time

from casadi import DM
import numpy as np
from scipy.sparse import csc_matrix

from rd3g.utilities.util import BASEDIR
from rd3g.utilities.casadi_util import generate_code
from rd3g.games.car_merge_kinematic_bicycle_casadi import create_random_game
from rd3g.solvers.rd3g_casadi import RD3GCasadi, RD3GCasadiConfig

from rd3g.src.build.lib import rd3g_casadi


if __name__ == "__main__":
    # CasADi expects fixed dimension, so the exact game config needs to be given apriori
    # For any specific game, a separate source file is created for each pair of (T, N)
    # This example shows generating src file for one specific pair of (T,N)
    game = create_random_game(car_count=3, horizon=20)
    solver_config = RD3GCasadiConfig()
    solver = RD3GCasadi(solver_config, game)
    generate_code(solver)

    # The generated source code need to be compiled before the following code can be run

    # Load compiled solver, compare results
    # Example for sending matrices to/from casadi
    module_name = game.__module__.rsplit('.', maxsplit=1)[-1]

    cpp_solver = rd3g_casadi.Rd3gCasadi(
        game.config.N,
        game.config.T,
        game.config.n_hi,
        game.config.dt,
        solver_config.bc_a,
        solver_config.bc_b,
        solver_config.reg,
        solver_config.reg_inertia,
        solver_config.inertia_correction,
        solver_config.rollout_each_step,
        solver_config.precondition_with_potential,
        solver_config.variational_gne,
        solver_config.tolerance,
        solver_config.tau_decay,
        solver_config.line_search_max_iter,
        solver_config.max_failed_line_search,
        solver_config.iterations,
        solver_config.max_in_reg_iter,
        solver_config.max_in_reg_val,
        solver_config.linear_solver_method,
        0,
        BASEDIR,
        module_name
    )

    N = game.config.N
    n = game.config.n
    m = game.config.m
    T = game.config.T
    n_hi = game.config.n_hi

    x = np.zeros((N*n, T))
    u = np.zeros((N*m, T))
    lamda = np.zeros((N*n, T))
    mu = np.zeros((n_hi*N, 1))
    t0 = time()
    # Call compiled casadi function, with numpy arrays
    res = cpp_solver.dr_dy(x, u, lamda, mu,
                           game.config.get_int_param_np(),
                           game.config.get_double_param_np())

    # Construct sparse and dense matrix from returned value
    matrix_sp = csc_matrix(
        (res.data, res.row, res.colind),
        shape=res.shape
    )
    cpp_retval = matrix_sp.toarray()
    dt = time() - t0
    print(f'cpp dr_dy = {dt=}')  # 1000x -> 0.05, lots of overhead

    # Replicate call in Python CasADi
    x_dm = DM(x)
    u_dm = DM(u)
    lamda_dm = DM(lamda)
    mu_dm = DM(mu)
    int_param_dm = DM(game.config.get_int_param_np())
    double_param_dm = DM(game.config.get_double_param_np())
    t0 = time()
    py_retval = solver.dr_dy_casadi(x, u, lamda, mu, int_param_dm, double_param_dm)
    dt = time() - t0
    print(f'py dr_dy = {dt=}')  # 1000x -> 0.19s
    # Check results
    assert np.linalg.norm(py_retval - cpp_retval) < 1e-8
