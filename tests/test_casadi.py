""" Test casadi game """
from dataclasses import fields
import casadi as cas
import numpy as np
from rd3g.games.car_merge_kinematic_bicycle_casadi import CarMergeKinematicBicycleCasadiConfig


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
