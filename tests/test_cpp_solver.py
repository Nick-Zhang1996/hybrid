''' Test cpp implementation'''
from unittest.mock import MagicMock

import pytest
import numpy as np

from ..examples.car_merge_kinematic_bicycle import CarMergeKinematicBicycle
from ..stein_game import SteinGameConfig


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

    # prepare dummy variables
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
