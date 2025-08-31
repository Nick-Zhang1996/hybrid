''' Test CarMergeKinematicBicyel with Jax '''
from unittest.mock import MagicMock

from jax import jit
import numpy as np

from ..stein_game import SteinGameConfig
from ..examples.car_merge_kinematic_bicycle import CarMergeKinematicBicycle


def test_python_impl():
    np.random.seed(0)
    config = SteinGameConfig(USE_CPP=False, iterations=2, particles=3)
    main = CarMergeKinematicBicycle(config, car_count=2, T=4)
    main.setup()
    main.solve()

def test_cpp_impl_is_called_when_use_cpp():
    config = SteinGameConfig(USE_CPP=False, iterations=2, particles=3)
    main = CarMergeKinematicBicycle(config, car_count=2, T=4)
    x_k = np.ones((main.N,main.n))
    u_k_i = np.ones(main.m)
    i = 0
    py_retval = main.J(x_k, u_k_i, i)
    config = SteinGameConfig(USE_CPP=True, iterations=2, particles=3)
    main = CarMergeKinematicBicycle(config, car_count=2, T=4)
    main.cpp = MagicMock()
    main.cpp.J = MagicMock(return_value=py_retval)
    cpp_mocked_retval = main.J(x_k, u_k_i, i)
    main.cpp.J.assert_called_once_with(x_k, u_k_i, i)

def test_cpp_impl():
    np.random.seed(0)
    config = SteinGameConfig(USE_CPP=True, iterations=2, particles=3)
    main = CarMergeKinematicBicycle(config, car_count=2, T=4)
    main.setup()
    main.solve()

def test_J():
    config = SteinGameConfig(USE_CPP=False, iterations=2, particles=3)
    main = CarMergeKinematicBicycle(config, car_count=2, T=4)
    main.setup()
    jit(main.J)

'''
def dJi_dxi(self, x_k, u_k_i, i):
def dJi_dxj(self, x_k, u_k_i, i, j):
def dJi_dxj(self, x_k, u_k_i, i, j):
def dJi_du(self, x_k, u_k_i, i):
def dJi_dxi_dxi(self, x_k, u, i):
def dJi_dxi_dxj(self, x_k, u_k_i, i, j):
def dJi_dxj_dxj(self, x_k, u_k_i, i, j):
def dJi_dudu(self, x_k, u_k_i, i):
def f(self, x, u, i):
def df_dx(self, x, u, i):
def df_du(self, x, u, i):
def h(self, x_i, x_j):
def dh_dxi(self, x_i, x_j):
def dh_dxj(self, x_i, x_j):
def dh_dxi_dxi(self, x_i, x_j):
def dh_dxj_dxi(self, x_i, x_j):
def dh_dxi_dxj(self, x_i, x_j):
def dh_dxj_dxj(self, x_i, x_j):
'''