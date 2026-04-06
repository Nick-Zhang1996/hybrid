from jax import jit, grad, jacfwd, jacrev, hessian, jacobian
import numpy as np

from ..stein_game import SteinGameConfig
from ..examples.car_merge_kinematic_bicycle import CarMergeKinematicBicycle


def test_jax_autodiff():
    ''' Test how autodiff works'''

    cpp_config = SteinGameConfig(USE_CPP=True, iterations=2, particles=3)
    cpp_main = CarMergeKinematicBicycle(cpp_config, car_count=2, T=4)
    cpp_main.setup()

    py_config = SteinGameConfig(USE_CPP=False, iterations=2, particles=3)
    py_main = CarMergeKinematicBicycle(py_config, car_count=2, T=4)
    f = jit(py_main.f)
    df_dx = jit(grad(py_main.f, argnums=0))
    df_du = jit(grad(py_main.f, argnums=1))
    i = 1

    # Test values
    x = np.random.uniform(-1, 1, py_main.n)
    u = np.random.uniform(-1, 1, py_main.m)
    cpp_retval_f = cpp_main.f(x, u, i)
    cpp_retval_df_dx = cpp_main.df_dx(x, u, i)
    cpp_retval_df_du = cpp_main.df_du(x, u, i)

    jax_retval_f = f(x, u, i)
    jax_retval_df_dx = df_dx(x, u, i)
    jax_retval_df_du = df_du(x, u, i)
