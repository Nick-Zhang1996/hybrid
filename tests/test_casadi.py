""" Test casadi game """
from dataclasses import fields
from functools import partial
import casadi as cas
import numpy as np
from rd3g.games.car_merge_kinematic_bicycle_casadi import CarMergeKinematicBicycleCasadiConfig
from rd3g.games.car_merge_kinematic_bicycle_casadi import create_random_game
from rd3g.solvers.rd3g_casadi import RD3GCasadi, RD3GCasadiConfig


def test_casadi_config():
    """ Test casadi param dict can recover attributes packed into flat param vectors """
    N = 3
    n = 4
    m = 2
    x0 = np.random.uniform(low=-1.0, high=1.0, size=(N, n))
    target_x_ref = np.random.uniform(low=-1.0, high=1.0, size=(N, n))
    J_Qr = np.random.uniform(low=-1.0, high=1.0, size=(n, n))
    J_Q = np.random.uniform(low=-1.0, high=1.0, size=(n, n))
    J_R = np.random.uniform(low=-1.0, high=1.0, size=(m, m))
    config = CarMergeKinematicBicycleCasadiConfig(T=5,
                                                  dt=0.1,
                                                  N=N,
                                                  n=n,
                                                  m=m,
                                                  track_width=1.0,
                                                  collision_radius=2.0,
                                                  x0=x0,
                                                  target_x_ref=target_x_ref,
                                                  J_Qr=J_Qr,
                                                  J_Q=J_Q,
                                                  J_R=J_R)

    params_val = [cas.SX(config.get_double_param_np()), cas.SX(config.get_int_param_np())]
    params_sx = [config.get_double_param_sx(), config.get_int_param_sx()]

    # config.get_param(var_name) retrieve sliced SX object
    # params_sx is the flattened param SX vector
    # params_val is the flattened param values from numpy array, cast to DM
    # Verify get_param can retrieve the original value
    assert isinstance(config.get_param('T'), cas.SX)
    cas_T = cas.DM(cas.substitute([config.get_param('T')], params_sx, params_val)).full()
    np.testing.assert_allclose(cas_T, config.T)

    for _field in fields(config):
        if 'param' in _field.name:
            continue
        print(f'checking {_field.name}')
        val_sx = cas.substitute([config.get_param(_field.name)],
                                params_sx,
                                params_val)
        val_dm = cas.DM(val_sx[0]).full()
        np.testing.assert_allclose(val_dm, getattr(config, _field.name))


def test_casadi_game():
    """ Test auto-differentiation correctness for RD3G CasADi solver"""
    game = create_random_game(car_count=3, horizon=20)
    config = RD3GCasadiConfig()
    solver = RD3GCasadi(config, game)
    N = game.config.N
    n = game.config.n
    m = game.config.m
    T = game.config.T

    x = cas.SX.sym('x', N*n, T)
    u = cas.SX.sym('u', N*m, T)
    lamda = cas.SX.sym('lamda', N*n, T)
    mu = cas.SX.sym('mu', N*N, T)

    x_k = cas.SX.sym('x_k', N, n)
    # x_k_i = cas.SX.sym('x_k_i', n)
    x_k1_i = cas.SX.sym('x_k1_i', n)
    u_k_i = cas.SX.sym('u_k_i', m)

    lamda_k = cas.SX.sym('lamda_k', N, n)
    mu_k = cas.SX.sym('mu_k', N, N)

    config_params = [game.config.get_int_param_sx(), game.config.get_double_param_sx()]

    # L, since i is a constant, use a partial function
    args = [x_k, u_k_i, x_k1_i, lamda_k, mu_k]

    def L_fixed_i(x_k, u_k_i, x_k1_i, lamda_k, mu_k):
        return solver.L(x_k, u_k_i, x_k1_i, lamda_k, mu_k, i=1)

    L_val = L_fixed_i(*args)
    L = cas.Function('L', args+config_params, [L_val])

    # LLi
    args = [x, u, lamda, mu]

    def LLi_fixed_i(x, u, lamda, mu):
        return solver.LLi(x, u, lamda, mu, i=1)
    LLi_val = LLi_fixed_i(*args)
    LLi = cas.Function('LLi', args+config_params, [LLi_val])

    # r
    r_val = solver.r(*args)
    r = cas.Function('r', args+config_params, [r_val])
    print(r_val.shape)
    T = solver.T
    N = solver.N
    n = solver.n
    m = solver.m
    r_dim = N*(T*(n+m) + T*n + T*N)
    print(r_dim)
