''' Test CarMergeKinematicBicyel with Jax '''
from ..examples.car_merge_kinematic_bicycle import CarMergeKinematicBicycle
from ..stein_game import SteinGameConfig
import jax.numpy as jnp
from jax import jit, grad, jacfwd, jacrev, hessian, jacobian, vmap
from unittest.mock import MagicMock

import pytest
import numpy as np
from time import time


def assert_flattened_allclose(val1, val2):
    # np.testing.assert_allclose(val1, val2)
    np.testing.assert_allclose(np.array(val1).flatten(), np.array(val2).flatten())


def assert_allclose(val1, val2):
    np.testing.assert_allclose(val1, val2, rtol=1e-5, atol=1e-5)


@pytest.mark.skip(reason="python solver broken during refactoring")
def test_python_solve():
    np.random.seed(0)
    config = SteinGameConfig(USE_CPP=False, iterations=2, particles=3)
    main = CarMergeKinematicBicycle(config, car_count=2, T=4)
    main.setup()
    main.solve()


def test_cpp_solve():
    np.random.seed(0)
    config = SteinGameConfig(USE_CPP=True, iterations=2, particles=3)
    main = CarMergeKinematicBicycle(config, car_count=2, T=4)
    main.setup()
    _, _, has_converged = main.solve()
    assert has_converged


@pytest.mark.skip(reason="python solver broken during refactoring")
def test_cpp_impl_is_called_and_correct():
    config = SteinGameConfig(USE_CPP=False, iterations=2, particles=3)
    python_main = CarMergeKinematicBicycle(config, car_count=2, T=4)

    x_k = np.ones((python_main.N, python_main.n)) * 1.1
    u_k_i = np.ones(python_main.m) * 1.2
    i = 0
    j = 1
    x = np.ones(python_main.n) * 1.3
    u = np.ones(python_main.m) * 1.4
    x_i = np.ones(python_main.n) * 1.5
    x_j = np.ones(python_main.n) * 1.6

    py_retval_J = python_main.J(x_k, u_k_i, i)
    py_retval_dJi_dxi = python_main.dJi_dxi(x_k, u_k_i, i)
    py_retval_dJi_dxj = python_main.dJi_dxj(x_k, u_k_i, i, j)
    py_retval_dJi_du = python_main.dJi_du(x_k, u_k_i, i)
    py_retval_dJi_dxi_dxi = python_main.dJi_dxi_dxi(x_k, u, i)
    py_retval_dJi_dxi_dxj = python_main.dJi_dxi_dxj(x_k, u_k_i, i, j)
    py_retval_dJi_dxj_dxj = python_main.dJi_dxj_dxj(x_k, u_k_i, i, j)
    py_retval_dJi_dudu = python_main.dJi_dudu(x_k, u_k_i, i)
    py_retval_f = python_main.f(x, u, i)
    py_retval_df_dx = python_main.df_dx(x, u, i)
    py_retval_df_du = python_main.df_du(x, u, i)
    py_retval_h = python_main.h(x_i, x_j)
    py_retval_dh_dxi = python_main.dh_dxi(x_i, x_j)
    py_retval_dh_dxj = python_main.dh_dxj(x_i, x_j)
    py_retval_dh_dxi_dxi = python_main.dh_dxi_dxi(x_i, x_j)
    py_retval_dh_dxj_dxi = python_main.dh_dxj_dxi(x_i, x_j)
    py_retval_dh_dxi_dxj = python_main.dh_dxi_dxj(x_i, x_j)
    py_retval_dh_dxj_dxj = python_main.dh_dxj_dxj(x_i, x_j)

    py_retval_Bh = python_main.Bh(x_i, x_j)
    py_retval_dBh_dxi = python_main.dBh_dxi(x_i, x_j)
    py_retval_dBh_dxj = python_main.dBh_dxj(x_i, x_j)
    py_retval_dBh_dxi_dxi = python_main.dBh_dxi_dxi(x_i, x_j)
    py_retval_dBh_dxi_dxj = python_main.dBh_dxi_dxj(x_i, x_j)
    py_retval_dBh_dxj_dxj = python_main.dBh_dxj_dxj(x_i, x_j)

    config = SteinGameConfig(USE_CPP=True, iterations=2, particles=3)
    cpp_mocked_main = CarMergeKinematicBicycle(config, car_count=2, T=4)
    cpp_mocked_main.cpp = MagicMock()

    cpp_mocked_main.cpp.J = MagicMock(return_value=py_retval_J)
    cpp_mocked_main.cpp.dJi_dxi = MagicMock(return_value=py_retval_dJi_dxi)
    cpp_mocked_main.cpp.dJi_dxj = MagicMock(return_value=py_retval_dJi_dxj)
    cpp_mocked_main.cpp.dJi_du = MagicMock(return_value=py_retval_dJi_du)
    cpp_mocked_main.cpp.dJi_dxi_dxi = MagicMock(return_value=py_retval_dJi_dxi_dxi)
    cpp_mocked_main.cpp.Ji_dxi_dxj = MagicMock(return_value=py_retval_dJi_dxi_dxj)
    cpp_mocked_main.cpp.dJi_dxj_dxj = MagicMock(return_value=py_retval_dJi_dxj_dxj)
    cpp_mocked_main.cpp.dJi_dudu = MagicMock(return_value=py_retval_dJi_dudu)
    cpp_mocked_main.cpp.f = MagicMock(return_value=py_retval_f)
    cpp_mocked_main.cpp.df_dx = MagicMock(return_value=py_retval_df_dx)
    cpp_mocked_main.cpp.df_du = MagicMock(return_value=py_retval_df_du)
    cpp_mocked_main.cpp.h = MagicMock(return_value=py_retval_h)
    cpp_mocked_main.cpp.dh_dxi = MagicMock(return_value=py_retval_dh_dxi)
    cpp_mocked_main.cpp.dh_dxj = MagicMock(return_value=py_retval_dh_dxj)
    cpp_mocked_main.cpp.dh_dxi_dxi = MagicMock(return_value=py_retval_dh_dxi_dxi)
    cpp_mocked_main.cpp.dh_dxj_dxi = MagicMock(return_value=py_retval_dh_dxj_dxi)
    cpp_mocked_main.cpp.dh_dxi_dxj = MagicMock(return_value=py_retval_dh_dxi_dxj)
    cpp_mocked_main.cpp.dh_dxj_dxj = MagicMock(return_value=py_retval_dh_dxj_dxj)

    cpp_mocked_main.cpp.Bh = MagicMock(return_value=py_retval_Bh)
    cpp_mocked_main.cpp.dBh_dxi = MagicMock(return_value=py_retval_dBh_dxi)
    cpp_mocked_main.cpp.dBh_dxj = MagicMock(return_value=py_retval_dBh_dxj)
    cpp_mocked_main.cpp.dBh_dxi_dxi = MagicMock(return_value=py_retval_dBh_dxi_dxi)
    cpp_mocked_main.cpp.dBh_dxi_dxj = MagicMock(return_value=py_retval_dBh_dxi_dxj)
    cpp_mocked_main.cpp.dBh_dxj_dxj = MagicMock(return_value=py_retval_dBh_dxj_dxj)

    cpp_mocked_main.J(x_k, u_k_i, i)
    cpp_mocked_main.dJi_dxi(x_k, u_k_i, i)
    cpp_mocked_main.dJi_dxj(x_k, u_k_i, i, j)
    cpp_mocked_main.dJi_du(x_k, u_k_i, i)
    cpp_mocked_main.dJi_dxi_dxi(x_k, u, i)
    cpp_mocked_main.dJi_dxi_dxj(x_k, u_k_i, i, j)
    cpp_mocked_main.dJi_dxj_dxj(x_k, u_k_i, i, j)
    cpp_mocked_main.dJi_dudu(x_k, u_k_i, i)
    cpp_mocked_main.f(x, u, i)
    cpp_mocked_main.df_dx(x, u, i)
    cpp_mocked_main.df_du(x, u, i)
    cpp_mocked_main.h(x_i, x_j)
    cpp_mocked_main.dh_dxi(x_i, x_j)
    cpp_mocked_main.dh_dxj(x_i, x_j)
    cpp_mocked_main.dh_dxi_dxi(x_i, x_j)
    cpp_mocked_main.dh_dxj_dxi(x_i, x_j)
    cpp_mocked_main.dh_dxi_dxj(x_i, x_j)
    cpp_mocked_main.dh_dxj_dxj(x_i, x_j)

    cpp_mocked_main.Bh(x_i, x_j)
    cpp_mocked_main.dBh_dxi(x_i, x_j)
    cpp_mocked_main.dBh_dxj(x_i, x_j)
    cpp_mocked_main.dBh_dxi_dxi(x_i, x_j)
    cpp_mocked_main.dBh_dxi_dxj(x_i, x_j)
    cpp_mocked_main.dBh_dxj_dxj(x_i, x_j)

    cpp_mocked_main.cpp.J.assert_called_once_with(x_k, u_k_i, i)
    cpp_mocked_main.cpp.dJi_dxi.assert_called_once_with(x_k, u_k_i, i)
    cpp_mocked_main.cpp.dJi_dxj.assert_called_once_with(x_k, u_k_i, i, j)
    cpp_mocked_main.cpp.dJi_du.assert_called_once_with(x_k, u_k_i, i)
    cpp_mocked_main.cpp.dJi_dxi_dxi.assert_called_once_with(x_k, u, i)
    cpp_mocked_main.cpp.dJi_dxi_dxj.assert_called_once_with(x_k, u_k_i, i, j)
    cpp_mocked_main.cpp.dJi_dxj_dxj.assert_called_once_with(x_k, u_k_i, i, j)
    cpp_mocked_main.cpp.dJi_dudu.assert_called_once_with(x_k, u_k_i, i)
    cpp_mocked_main.cpp.f.assert_called_once_with(x, u, i)
    cpp_mocked_main.cpp.df_dx.assert_called_once_with(x, u, i)
    cpp_mocked_main.cpp.df_du.assert_called_once_with(x, u, i)
    cpp_mocked_main.cpp.h.assert_called_once_with(x_i, x_j)
    cpp_mocked_main.cpp.dh_dxi.assert_called_once_with(x_i, x_j)
    cpp_mocked_main.cpp.dh_dxj.assert_called_once_with(x_i, x_j)
    cpp_mocked_main.cpp.dh_dxi_dxi.assert_called_once_with(x_i, x_j)
    cpp_mocked_main.cpp.dh_dxj_dxi.assert_called_once_with(x_i, x_j)
    cpp_mocked_main.cpp.dh_dxi_dxj.assert_called_once_with(x_i, x_j)
    cpp_mocked_main.cpp.dh_dxj_dxj.assert_called_once_with(x_i, x_j)

    cpp_mocked_main.cpp.Bh.assert_called_once_with(x_i, x_j)
    cpp_mocked_main.cpp.dBh_dxi.assert_called_once_with(x_i, x_j)
    cpp_mocked_main.cpp.dBh_dxj.assert_called_once_with(x_i, x_j)
    cpp_mocked_main.cpp.dBh_dxi_dxi.assert_called_once_with(x_i, x_j)
    cpp_mocked_main.cpp.dBh_dxi_dxj.assert_called_once_with(x_i, x_j)
    cpp_mocked_main.cpp.dBh_dxj_dxj.assert_called_once_with(x_i, x_j)

    config = SteinGameConfig(USE_CPP=True, iterations=2, particles=3)
    cpp_main = CarMergeKinematicBicycle(config, car_count=2, T=4)
    cpp_main.setup()
    cpp_retval_J = cpp_main.J(x_k, u_k_i, i)
    cpp_retval_dJi_dxi = cpp_main.dJi_dxi(x_k, u_k_i, i)
    cpp_retval_dJi_dxj = cpp_main.dJi_dxj(x_k, u_k_i, i, j)
    cpp_retval_dJi_du = cpp_main.dJi_du(x_k, u_k_i, i)
    cpp_retval_dJi_dxi_dxi = cpp_main.dJi_dxi_dxi(x_k, u, i)
    cpp_retval_dJi_dxi_dxj = cpp_main.dJi_dxi_dxj(x_k, u_k_i, i, j)
    cpp_etval_dJi_dxj_dxj = cpp_main.dJi_dxj_dxj(x_k, u_k_i, i, j)
    cpp_retval_dJi_dudu = cpp_main.dJi_dudu(x_k, u_k_i, i)
    cpp_retval_f = cpp_main.f(x, u, i)
    cpp_retval_df_dx = cpp_main.df_dx(x, u, i)
    cpp_retval_df_du = cpp_main.df_du(x, u, i)
    cpp_retval_h = cpp_main.h(x_i, x_j)
    cpp_retval_dh_dxi = cpp_main.dh_dxi(x_i, x_j)
    cpp_retval_dh_dxj = cpp_main.dh_dxj(x_i, x_j)
    cpp_retval_dh_dxi_dxi = cpp_main.dh_dxi_dxi(x_i, x_j)
    cpp_retval_dh_dxj_dxi = cpp_main.dh_dxj_dxi(x_i, x_j)
    cpp_retval_dh_dxi_dxj = cpp_main.dh_dxi_dxj(x_i, x_j)
    cpp_retval_dh_dxj_dxj = cpp_main.dh_dxj_dxj(x_i, x_j)

    cpp_retval_Bh = cpp_main.Bh(x_i, x_j)
    cpp_retval_dBh_dxi = cpp_main.dBh_dxi(x_i, x_j)
    cpp_retval_dBh_dxj = cpp_main.dBh_dxj(x_i, x_j)
    cpp_retval_dBh_dxi_dxi = cpp_main.dBh_dxi_dxi(x_i, x_j)
    cpp_retval_dBh_dxi_dxj = cpp_main.dBh_dxi_dxj(x_i, x_j)
    cpp_retval_dBh_dxj_dxj = cpp_main.dBh_dxj_dxj(x_i, x_j)

    assert_flattened_allclose(py_retval_J, cpp_retval_J)
    assert_flattened_allclose(py_retval_dJi_dxi, cpp_retval_dJi_dxi)
    assert_flattened_allclose(py_retval_dJi_dxj, cpp_retval_dJi_dxj)
    assert_flattened_allclose(py_retval_dJi_du, cpp_retval_dJi_du)
    assert_flattened_allclose(py_retval_dJi_dxi_dxi, cpp_retval_dJi_dxi_dxi)
    assert_flattened_allclose(py_retval_dJi_dxi_dxj, cpp_retval_dJi_dxi_dxj)
    assert_flattened_allclose(py_retval_dJi_dxj_dxj, cpp_etval_dJi_dxj_dxj)
    assert_flattened_allclose(py_retval_dJi_dudu, cpp_retval_dJi_dudu)
    assert_flattened_allclose(py_retval_f, cpp_retval_f)
    assert_flattened_allclose(py_retval_df_dx, cpp_retval_df_dx)
    assert_flattened_allclose(py_retval_df_du, cpp_retval_df_du)
    assert_flattened_allclose(py_retval_h, cpp_retval_h)
    assert_flattened_allclose(py_retval_dh_dxi, cpp_retval_dh_dxi)
    assert_flattened_allclose(py_retval_dh_dxj, cpp_retval_dh_dxj)
    assert_flattened_allclose(py_retval_dh_dxi_dxi, cpp_retval_dh_dxi_dxi)
    assert_flattened_allclose(py_retval_dh_dxj_dxi, cpp_retval_dh_dxj_dxi)
    assert_flattened_allclose(py_retval_dh_dxi_dxj, cpp_retval_dh_dxi_dxj)
    assert_flattened_allclose(py_retval_dh_dxj_dxj, cpp_retval_dh_dxj_dxj)

    assert_flattened_allclose(py_retval_Bh, cpp_retval_Bh)
    assert_flattened_allclose(py_retval_dBh_dxi, cpp_retval_dBh_dxi)
    assert_flattened_allclose(py_retval_dBh_dxj, cpp_retval_dBh_dxj)
    assert_flattened_allclose(py_retval_dBh_dxi_dxi, cpp_retval_dBh_dxi_dxi)
    assert_flattened_allclose(py_retval_dBh_dxi_dxj, cpp_retval_dBh_dxi_dxj)
    assert_flattened_allclose(py_retval_dBh_dxj_dxj, cpp_retval_dBh_dxj_dxj)


def test_jax_autodiff():
    ''' Test how autodiff works'''

    cpp_config = SteinGameConfig(USE_CPP=True, iterations=2, particles=3)
    cpp_main = CarMergeKinematicBicycle(cpp_config, car_count=2, T=4)
    cpp_main.setup()

    py_config = SteinGameConfig(USE_CPP=False, iterations=2, particles=3)
    py_main = CarMergeKinematicBicycle(py_config, car_count=2, T=4)

    i = 1
    # Test values
    x = np.random.uniform(-1, 1, (py_main.T, py_main.N, py_main.n))
    u = np.random.uniform(-1, 1,  (py_main.T, py_main.N, py_main.m))
    x_j = np.random.uniform(-1, 1, py_main.n)
    x_k = np.random.uniform(-1, 1, (py_main.N, py_main.n))
    x_k_i = np.random.uniform(-1, 1, py_main.n)
    u_k_i = np.random.uniform(-1, 1, py_main.m)
    i = 0
    j = 1
    k = 2
    x_k1_i = np.random.uniform(-1, 1, py_main.n)
    ''' n '''
    lamda = np.random.uniform(-1, 1, (py_main.T, py_main.N, py_main.n))
    ''' T, N, n'''
    mu = np.random.uniform(-1, 1, (py_main.T, py_main.N, py_main.N))
    ''' T, N, N'''
    lamda_k = lamda[k]
    mu_k = mu[k]

    # Create jit version of functions
    f = jit(py_main.jax_f)
    df_dx = jit(jacobian(py_main.jax_f, argnums=0))
    df_du = jit(jacobian(py_main.jax_f, argnums=1))
    h = jit(py_main.h)
    dh_dxi = jit(jacobian(py_main.h, argnums=0))
    dh_dxj = jit(jacobian(py_main.h, argnums=1))
    dh_dxi_dxi = jit(jacfwd(jacrev(py_main.h, argnums=0), argnums=0))
    dh_dxi_dxj = jit(jacfwd(jacrev(py_main.h, argnums=0), argnums=1))
    dh_dxj_dxi = jit(jacfwd(jacrev(py_main.h, argnums=1), argnums=0))
    dh_dxj_dxj = jit(jacfwd(jacrev(py_main.h, argnums=1), argnums=1))

    J = jit(py_main.jax_J)

    def dJi_dxk(x_k, u_k_i, i):
        return jacrev(py_main.jax_J, argnums=0)(x_k, u_k_i, i)

    def dJi_dxk_dxk(x_k, u_k_i, i):
        return jacfwd(jacrev(py_main.jax_J, argnums=0), argnums=0)(x_k, u_k_i, i)

    dJi_dxi = jit(lambda x_k, u_k_i, i: dJi_dxk(x_k, u_k_i, i)[i])
    dJi_dxj = jit(lambda x_k, u_k_i, i, j: dJi_dxk(x_k, u_k_i, i)[j])
    dJi_du = jit(jacrev(py_main.jax_J, argnums=1))
    dJi_dudu = jit(jacfwd(jacrev(py_main.jax_J, argnums=1), argnums=1))
    dJi_dxi_dxi = jit(lambda x_k, u_k_i, i: dJi_dxk_dxk(x_k, u_k_i, i)[i, :, i, :])
    dJi_dxi_dxj = jit(lambda x_k, u_k_i, i, j: dJi_dxk_dxk(x_k, u_k_i, i)[i, :, j, :])
    dJi_dxj_dxi = jit(lambda x_k, u_k_i, i, j: dJi_dxk_dxk(x_k, u_k_i, i)[j, :, i, :])
    dJi_dxj_dxj = jit(lambda x_k, u_k_i, i, j: dJi_dxk_dxk(x_k, u_k_i, i)[j, :, j, :])

    Bh = jit(py_main.Bh)
    dBh_dxi = jit(jacrev(py_main.dBh_dxi, argnums=0))
    dBh_dxj = jit(jacrev(py_main.dBh_dxi, argnums=1))
    dBh_dxi_dxi = jit(jacfwd(jacrev(py_main.dBh_dxi, argnums=0), argnums=0))
    dBh_dxi_dxj = jit(jacfwd(jacrev(py_main.dBh_dxi, argnums=0), argnums=1))
    dBh_dxj_dxi = jit(jacfwd(jacrev(py_main.dBh_dxi, argnums=1), argnums=0))
    dBh_dxj_dxj = jit(jacfwd(jacrev(py_main.dBh_dxi, argnums=1), argnums=1))

    # Evaluate cpp reference values
    cpp_retval_f = cpp_main.f(x_k_i, u_k_i, i)
    cpp_retval_df_dx = cpp_main.df_dx(x_k_i, u_k_i, i)
    cpp_retval_df_du = cpp_main.df_du(x_k_i, u_k_i, i)
    cpp_retval_h = cpp_main.h(x_k_i, x_j)
    cpp_retval_dh_dxi = cpp_main.dh_dxi(x_k_i, x_j)
    cpp_retval_dh_dxj = cpp_main.dh_dxj(x_k_i, x_j)
    cpp_retval_dh_dxi_dxi = cpp_main.dh_dxi_dxi(x_k_i, x_j)
    cpp_retval_dh_dxi_dxj = cpp_main.dh_dxi_dxj(x_k_i, x_j)
    cpp_retval_dh_dxj_dxi = cpp_main.dh_dxj_dxi(x_k_i, x_j)
    cpp_retval_dh_dxj_dxj = cpp_main.dh_dxj_dxj(x_k_i, x_j)
    cpp_retval_J = cpp_main.J(x_k, u_k_i, i)
    cpp_retval_dJi_dxi = cpp_main.dJi_dxi(x_k, u_k_i, i)
    cpp_retval_dJi_dxj = cpp_main.dJi_dxj(x_k, u_k_i, i, j)
    cpp_retval_dJi_du = cpp_main.dJi_du(x_k, u_k_i, i)
    cpp_retval_dJi_dxi_dxi = cpp_main.dJi_dxi_dxi(x_k, u_k_i, i)
    cpp_retval_dJi_dxi_dxj = cpp_main.dJi_dxi_dxj(x_k, u_k_i, i, j)
    cpp_retval_dJi_dxj_dxi = cpp_main.dJi_dxi_dxj(x_k, u_k_i, i, j).T  # pylint: disable=no-member
    cpp_retval_dJi_dxj_dxj = cpp_main.dJi_dxj_dxj(x_k, u_k_i, i, j)
    cpp_retval_dJi_dudu = cpp_main.dJi_dudu(x_k, u_k_i, i)
    # cpp_retval_Bh = cpp_main.Bh( x_i, x_j)
    # cpp_retval_dBh_dxi = cpp_main.dBh_dxi( x_i, x_j)
    # cpp_retval_dBh_dxj = cpp_main.dBh_dxj( x_i, x_j)
    # cpp_retval_dBh_dxi_dxi = cpp_main.dBh_dxi_dxi( x_i, x_j)
    # cpp_retval_dBh_dxi_dxj = cpp_main.dBh_dxi_dxj( x_i, x_j)
    # cpp_retval_dBh_dxj_dxi = cpp_main.dBh_dxi_dxj( x_i, x_j).T # pylint: disable=no-member
    # cpp_retval_dBh_dxj_dxj = cpp_main.dBh_dxj_dxj( x_i, x_j)

    # Compile and evaluate jax values
    jax_retval_f = f(x_k_i, u_k_i, i)
    jax_retval_df_dx = df_dx(x_k_i, u_k_i, i)
    jax_retval_df_du = df_du(x_k_i, u_k_i, i)
    jax_retval_h = h(x_k_i, x_j)
    jax_retval_dh_dxi = dh_dxi(x_k_i, x_j)
    jax_retval_dh_dxj = dh_dxj(x_k_i, x_j)
    jax_retval_dh_dxi_dxi = dh_dxi_dxi(x_k_i, x_j)
    jax_retval_dh_dxi_dxj = dh_dxi_dxj(x_k_i, x_j)
    jax_retval_dh_dxj_dxi = dh_dxj_dxi(x_k_i, x_j)
    jax_retval_dh_dxj_dxj = dh_dxj_dxj(x_k_i, x_j)
    jax_retval_J = J(x_k, u_k_i, i)
    jax_retval_dJi_dxi = dJi_dxi(x_k, u_k_i, i)
    jax_retval_dJi_dxj = dJi_dxj(x_k, u_k_i, i, j)
    jax_retval_dJi_dxi_dxi = dJi_dxi_dxi(x_k, u_k_i, i)
    jax_retval_dJi_dxi_dxj = dJi_dxi_dxj(x_k, u_k_i, i, j)
    jax_retval_dJi_dxj_dxi = dJi_dxj_dxi(x_k, u_k_i, i, j)
    jax_retval_dJi_dxj_dxj = dJi_dxj_dxj(x_k, u_k_i, i, j)
    jax_retval_dJi_du = dJi_du(x_k, u_k_i, i)
    jax_retval_dJi_dudu = dJi_dudu(x_k, u_k_i, i)

    jax_retval_Bh = Bh(x_k_i, x_j)
    jax_retval_dBh_dxi = dBh_dxi(x_k_i, x_j)
    jax_retval_dBh_dxj = dBh_dxj(x_k_i, x_j)
    jax_retval_dBh_dxi_dxi = dBh_dxi_dxi(x_k_i, x_j)
    jax_retval_dBh_dxi_dxj = dBh_dxi_dxj(x_k_i, x_j)
    jax_retval_dBh_dxj_dxi = dBh_dxj_dxi(x_k_i, x_j)
    jax_retval_dBh_dxj_dxj = dBh_dxj_dxj(x_k_i, x_j)

    assert_allclose(cpp_retval_f.reshape(py_main.n), jax_retval_f)
    assert_allclose(cpp_retval_df_dx, jax_retval_df_dx)
    assert_allclose(cpp_retval_df_du, jax_retval_df_du)
    assert_allclose(cpp_retval_h, jax_retval_h)
    assert_allclose(cpp_retval_dh_dxi[0], jax_retval_dh_dxi)
    assert_allclose(cpp_retval_dh_dxj[0], jax_retval_dh_dxj)
    assert_allclose(cpp_retval_dh_dxi_dxi, jax_retval_dh_dxi_dxi)
    assert_allclose(cpp_retval_dh_dxi_dxj, jax_retval_dh_dxi_dxj)
    assert_allclose(cpp_retval_dh_dxj_dxi, jax_retval_dh_dxj_dxi)
    assert_allclose(cpp_retval_dh_dxj_dxj, jax_retval_dh_dxj_dxj)

    assert_allclose(cpp_retval_J, jax_retval_J)
    assert_allclose(cpp_retval_dJi_dxi[0], jax_retval_dJi_dxi)
    assert_allclose(cpp_retval_dJi_dxj[0], jax_retval_dJi_dxj)
    assert_allclose(cpp_retval_dJi_du[0], jax_retval_dJi_du)
    assert_allclose(cpp_retval_dJi_dudu, jax_retval_dJi_dudu)
    assert_allclose(cpp_retval_dJi_dxi_dxi, jax_retval_dJi_dxi_dxi)
    assert_allclose(cpp_retval_dJi_dxi_dxj, jax_retval_dJi_dxi_dxj)
    assert_allclose(cpp_retval_dJi_dxj_dxi, jax_retval_dJi_dxj_dxi)
    assert_allclose(cpp_retval_dJi_dxj_dxj, jax_retval_dJi_dxj_dxj)

    jax_x = jnp.array(x)
    h_map_i = vmap(lambda k, i, j: py_main.h(
        jax_x[k, i], jax_x[k, j]), in_axes=(None, 0, None), out_axes=0)
    h_map_ij = vmap(h_map_i, in_axes=(None, None, 0), out_axes=0)
    h_map_kij = vmap(h_map_ij, in_axes=(0, None, None), out_axes=0)

    h_map_val = h_map_kij(jnp.arange(py_main.T),
                          jnp.arange(py_main.N),
                          jnp.arange(py_main.N))
    jax_retval_h_plus_mask = jnp.where(jnp.eye(py_main.N), False, h_map_val >= 0)
    py_retval_h_plus_mask = py_main.get_h_plus_mask(x)
    assert_allclose(py_retval_h_plus_mask, jax_retval_h_plus_mask)

    # Bh doesn't have cpp interface
    # assert_allclose(cpp_retval_Bh, jax_retval_Bh)
    # assert_allclose(cpp_retval_dBh_dxi[0], jax_retval_dBh_dxi)
    # assert_allclose(cpp_retval_dBh_dxj[0], jax_retval_dBh_dxj)
    # assert_allclose(cpp_retval_dBh_dxi_dxi, jax_retval_dBh_dxi_dxi)
    # assert_allclose(cpp_retval_dBh_dxi_dxj, jax_retval_dBh_dxi_dxj)
    # assert_allclose(cpp_retval_dBh_dxj_dxi, jax_retval_dBh_dxj_dxi)
    # assert_allclose(cpp_retval_dBh_dxj_dxj, jax_retval_dBh_dxj_dxj)

    # jit(main.Bh)
    # jit(main.dBh_dxi)
    # jit(main.dBh_dxj)
    # jit(main.dBh_dxi_dxi)
    # jit(main.dBh_dxi_dxj)
    # jit(main.dBh_dxj_dxj)

    # new format put one function together
    L = jit(py_main.jax_L)
    h_k_plus_mask = jax_retval_h_plus_mask[2]  # slice h_k, k=2
    py_retval_L = py_main.L(x_k, u_k_i, x_k1_i, h_k_plus_mask, lamda_k, mu_k, i)
    jax_retval_L = L(x_k, u_k_i, x_k1_i, h_k_plus_mask, lamda_k, mu_k, i)
    assert_allclose(py_retval_L, jax_retval_L)

    dL_dx_ik = jit(lambda *args: jacrev(py_main.jax_L, argnums=0)(*args)[i])
    py_retval_dL_dx_ik = py_main.dL_dx_ik(x_k, u_k_i, x_k1_i, h_k_plus_mask, lamda_k, mu_k, i)
    jax_retval_dL_dx_ik = dL_dx_ik(x_k, u_k_i, x_k1_i, h_k_plus_mask, lamda_k, mu_k, i)
    assert_allclose(py_retval_dL_dx_ik[0], jax_retval_dL_dx_ik)

    dL_dx_ik1 = jit(jacrev(py_main.jax_L, argnums=2))
    py_retval_dL_dx_ik1 = py_main.dL_dx_ik1(x_k, u_k_i, x_k1_i, h_k_plus_mask, lamda_k, mu_k, i)
    jax_retval_dL_dx_ik1 = dL_dx_ik1(x_k, u_k_i, x_k1_i, h_k_plus_mask, lamda_k, mu_k, i)
    assert_allclose(py_retval_dL_dx_ik1, jax_retval_dL_dx_ik1)
    # jit(main.dL_dx_jk)
    # jit(main.dL_du)

    # dr_dy y:x, u, lamda, mu
    # r needs dLLi_dxi, dLLi_dui, which needs LLi
    LLi = jit(py_main.jax_LLi)
    h_plus_mask = jax_retval_h_plus_mask
    py_retval_LLi = py_main.LLi(x, u, h_plus_mask, lamda, mu, i)
    jax_retval_LLi = LLi(x, u, h_plus_mask, lamda, mu, i)
    assert_allclose(py_retval_LLi, jax_retval_LLi)

    dLLi_dx = jit(jacrev(py_main.jax_LLi, argnums=0))
    dLLi_du = jit(jacrev(py_main.jax_LLi, argnums=1))

    def dLLi_dxi(x, u, h_plus_mask, lamda, mu, i):
        return dLLi_dx(x, u, h_plus_mask, lamda, mu, i)[:, i, :].reshape(1, -1)

    def dLLi_dui(x, u, h_plus_mask, lamda, mu, i):
        return dLLi_du(x, u, h_plus_mask, lamda, mu, i)[:, i, :].reshape(1, -1)

    py_retval_dLLi_dxi = py_main.dLLi_dxi(x, u, h_plus_mask, lamda, mu, i)
    jax_retval_dLLi_dxi = dLLi_dxi(x, u, h_plus_mask, lamda, mu, i)
    assert_allclose(py_retval_dLLi_dxi, jax_retval_dLLi_dxi)
    py_retval_dLLi_dui = py_main.dLLi_dui(x, u, h_plus_mask, lamda, mu, i)
    jax_retval_dLLi_dui = dLLi_dui(x, u, h_plus_mask, lamda, mu, i)
    assert_allclose(py_retval_dLLi_dui, jax_retval_dLLi_dui)

    # r = jit(py_main.r)

    # jit(main.L)
    # jit(main.dL_dx_ik)
    # jit(main.dL_dx_ik1)
    # jit(main.dL_du)

    # jit(main.LLi)
    # jit(main.dLLi_dxi)
    # jit(main.dLLi_dx)
    # jit(main.dLLi_dui)
    # jit(main.dLLi_du)
    # jit(main.dLLi_dxi_dmu)
    # jit(main.dLLi_dx_dmu)

    # jit(main.rollout)
    # jit(main.dr_dy)
    # jit(main.getCollisionResidual)
    # jit(main.getHplusMask)

    # jit(main.r)
    # jit(main.dLLi_dxi_dx)
    # jit(main.dLLi_dxdx)
    # jit(main.dLLi_dudx)
    # jit(main.dx_du)
    # jit(main.dF_dx)
    # jit(main.dF0_dx)
    # jit(main.dh_dx)
    # jit(main.dr_dx)
    # jit(main.dr_dx_old)
    # jit(main.dr_du)
    # jit(main.dr_dlamda)
    # jit(main.dr_dmu)
