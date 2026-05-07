""" Test casadi game """
import pytest
from dataclasses import fields

import casadi as cas
import numpy as np
from rd3g.games.car_merge_kinematic_bicycle_casadi import CarMergeKinematicBicycleCasadiConfig
from rd3g.games.car_merge_kinematic_bicycle_casadi import create_random_game
from rd3g.solvers.interior_point_game import InteriorPointGame, InteriorPointGameConfig


def test_casadi_config():
    """ Test casadi param dict can recover attributes packed into flat param vectors """
    N = 3
    n = 4
    m = 2
    x0 = np.random.uniform(low=-1.0, high=1.0, size=(n, N))
    target_x_ref = np.random.uniform(low=-1.0, high=1.0, size=(n, N))
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


@pytest.mark.skip('currently broken')
def test_casadi_game():
    """Test auto-differentiation correctness for the interior-point game solver."""
    game = create_random_game(car_count=3, horizon=20)
    config = InteriorPointGameConfig()
    solver = InteriorPointGame(config, game)
    N = game.config.N
    n = game.config.n
    m = game.config.m
    T = game.config.T
    n_hi = game.config.n_hi

    x = cas.SX.sym('x', n*N, T)
    u = cas.SX.sym('u', m*N, T)
    lamda = cas.SX.sym('lamda', N*n, T)
    mu = cas.SX.sym('mu', n_hi, N)

    x_k = cas.SX.sym('x_k', n, N)
    # x_k_i = cas.SX.sym('x_k_i', n)
    x_k1_i = cas.SX.sym('x_k1_i', n)
    u_k_i = cas.SX.sym('u_k_i', m)

    lamda_k = cas.SX.sym('lamda_k', n, N)
    mu_k = cas.SX.sym('mu_k', N, N)

    config_params = [game.config.get_int_param_sx(), game.config.get_double_param_sx()]

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
    r_dim = N*(T*(n+m) + T*n + n_hi)
    print(r_dim)

    # r derivative
    y = cas.vertcat(*[cas.vec(val) for val in args])
    dr_dy = cas.jacobian(r_val, y)
    print(dr_dy.shape)
