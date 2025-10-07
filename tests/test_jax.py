''' Test Jax '''
from time import time
import numpy as np
import jax.numpy as jnp
from jax import jit, jacfwd, jacrev, jacobian, vmap
from concurrent.futures import ThreadPoolExecutor

from ..examples.car_merge_kinematic_bicycle import CarMergeKinematicBicycle
from ..stein_game import SteinGameConfig


def assert_flattened_allclose(val1, val2):
    # np.testing.assert_allclose(val1, val2)
    np.testing.assert_allclose(np.array(val1).flatten(), np.array(val2).flatten())


def assert_allclose(val1, val2):
    np.testing.assert_allclose(val1, val2, rtol=1e-5, atol=1e-5)


def test_jax_autodiff():
    ''' Test how autodiff works'''

    cpp_config = SteinGameConfig(USE_CPP=True, iterations=2, particles=3)
    cpp_main = CarMergeKinematicBicycle(cpp_config, car_count=2, T=4)
    cpp_main.setup()

    py_config = SteinGameConfig(USE_CPP=False, iterations=2, particles=3)
    py_main = CarMergeKinematicBicycle(py_config, car_count=2, T=4)
    py_main.prepare_jax_functions()

    # Test values
    i = 1
    x = np.random.uniform(-1, 1, (py_main.T, py_main.N, py_main.n))
    u = np.random.uniform(-1, 1,  (py_main.T, py_main.N, py_main.m))
    x_j = np.random.uniform(-1, 1, py_main.n)
    x_k = np.random.uniform(-1, 1, (py_main.N, py_main.n))
    x_k_i = np.random.uniform(-1, 1, py_main.n)
    u_k_i = np.random.uniform(-1, 1, py_main.m)
    i = 0
    j = 1
    k = 2
    # n
    x_k1_i = np.random.uniform(-1, 1, py_main.n)
    # T, N, n
    lamda = np.random.uniform(-1, 1, (py_main.T, py_main.N, py_main.n))
    # T, N, N
    mu = np.random.uniform(-1, 1, (py_main.T, py_main.N, py_main.N))
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
    jax_retval_J = py_main.jax_J(x_k, u_k_i, i)
    jax_retval_dJi_dxi = dJi_dxi(x_k, u_k_i, i)
    jax_retval_dJi_dxj = dJi_dxj(x_k, u_k_i, i, j)
    jax_retval_dJi_dxi_dxi = dJi_dxi_dxi(x_k, u_k_i, i)
    jax_retval_dJi_dxi_dxj = dJi_dxi_dxj(x_k, u_k_i, i, j)
    jax_retval_dJi_dxj_dxi = dJi_dxj_dxi(x_k, u_k_i, i, j)
    jax_retval_dJi_dxj_dxj = dJi_dxj_dxj(x_k, u_k_i, i, j)
    jax_retval_dJi_du = dJi_du(x_k, u_k_i, i)
    jax_retval_dJi_dudu = dJi_dudu(x_k, u_k_i, i)

    # cpp impl always returns a matrix, so it will have two dimensions even for vectors
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
    h_map_val = py_main.jax_h_map_fun(jax_x)
    jax_retval_h_plus_mask = jnp.where(jnp.eye(py_main.N), False, h_map_val >= 0)
    py_retval_h_plus_mask = py_main.get_h_plus_mask(x)
    assert_allclose(py_retval_h_plus_mask, jax_retval_h_plus_mask)

    # new format put one function together
    h_k_plus_mask = jax_retval_h_plus_mask[2]  # slice h_k, k=2
    py_retval_L = py_main.L(x_k, u_k_i, x_k1_i, h_k_plus_mask, lamda_k, mu_k, i)
    jax_retval_L = py_main.jax_L(x_k, u_k_i, x_k1_i, h_k_plus_mask, lamda_k, mu_k, i)
    assert_allclose(py_retval_L, jax_retval_L)

    py_retval_dL_dx_ik = py_main.dL_dx_ik(x_k, u_k_i, x_k1_i, h_k_plus_mask, lamda_k, mu_k, i)
    jax_retval_dL_dx_ik = py_main.jax_dL_dx_ik(x_k, u_k_i, x_k1_i, h_k_plus_mask, lamda_k, mu_k, i)
    assert_allclose(py_retval_dL_dx_ik[0], jax_retval_dL_dx_ik)

    py_retval_dL_dx_ik1 = py_main.dL_dx_ik1(x_k, u_k_i, x_k1_i, h_k_plus_mask, lamda_k, mu_k, i)
    jax_retval_dL_dx_ik1 = py_main.jax_dL_dx_ik1(
        x_k, u_k_i, x_k1_i, h_k_plus_mask, lamda_k, mu_k, i)
    assert_allclose(py_retval_dL_dx_ik1, jax_retval_dL_dx_ik1)

    # dr_dy
    # r: dLL_dxi, dLL_dui, f(x,u) - x+
    # r needs dLLi_dxi, dLLi_dui, which needs LLi
    # y:x, u, lamda, mu
    h_plus_mask = jax_retval_h_plus_mask
    py_retval_LLi = py_main.LLi(x, u, h_plus_mask, lamda, mu, i)
    jax_retval_LLi = py_main.jax_LLi(x, u, h_plus_mask, lamda, mu, i)
    assert_allclose(py_retval_LLi, jax_retval_LLi)

    py_retval_dLLi_dxi = py_main.dLLi_dxi(x, u, h_plus_mask, lamda, mu, i)
    jax_retval_dLLi_dxi = py_main.jax_dLLi_dxi(x, u, h_plus_mask, lamda, mu, i)
    assert_allclose(py_retval_dLLi_dxi, jax_retval_dLLi_dxi)
    py_retval_dLLi_dui = py_main.dLLi_dui(x, u, h_plus_mask, lamda, mu, i)
    jax_retval_dLLi_dui = py_main.jax_dLLi_dui(x, u, h_plus_mask, lamda, mu, i)
    assert_allclose(py_retval_dLLi_dui, jax_retval_dLLi_dui)

    # NOTE the dimension is different because jax has static dimension and can't use active set
    t0 = time()
    py_retval_r = py_main.r(x, u, lamda, mu, h_plus_mask)
    py_dt = time()-t0
    py_pos = py_retval_r[np.abs(py_retval_r) > 0]
    t0 = time()
    cpp_retval_r = cpp_main.r(x, u, lamda, mu, h_plus_mask)
    cpp_dt = time()-t0
    # cpp_pos = cpp_retval_r[np.abs(cpp_retval_r) > 0]
    py_main.jax_r(x, u, lamda, mu, h_plus_mask)  # 200x faster than pure python
    t0 = time()
    jax_retval_r = py_main.jax_r(x, u, lamda, mu, h_plus_mask)  # 200x faster than pure python
    jax_dt = time()-t0
    jax_pos = jax_retval_r[np.abs(jax_retval_r) > 0]
    assert_allclose(py_pos, jax_pos)
    print('runtime for r')
    print(f'{py_dt=}')
    print(f'{cpp_dt=}')
    print(f'{jax_dt=}')

    # Dr dy
    py_retval_dr_dy = py_main.dr_dy(x, u, lamda, mu, h_plus_mask)
    jax_retval_dr_dy = py_main.jax_dr_dy(x, u, lamda, mu, h_plus_mask)
    # FIXME check supposedly empty row values
    # assert_allclose(np.sum(py_retval_dr_dy), np.sum(jax_retval_dr_dy))

    jax_step = jit(py_main._jax_step)
    # TODO test if residual indeed decreased
    stepped = jax_step(x, u, lamda, mu)
    t0 = time()
    py_stepped = py_main.step(x, u, lamda, mu)
    py_dt = time()-t0
    t0 = time()
    jax_stepped = jax_step(x, u, lamda, mu)  # 2x faster than cpp, ~100x faster than py
    jax_dt = time()-t0
    t0 = time()
    cpp_stepped = cpp_main.step(x, u, lamda, mu)
    cpp_dt = time()-t0
    print('runtime for step')
    print(f'{py_dt=}')
    print(f'{cpp_dt=}')
    print(f'{jax_dt=}')

    batch_size = 64
    print(f'runtime for vmapped particle x{batch_size}')
    batch_x = np.random.uniform(-1, 1, (batch_size, py_main.T, py_main.N, py_main.n))
    batch_u = np.random.uniform(-1, 1,  (batch_size, py_main.T, py_main.N, py_main.m))
    batch_lamda = np.random.uniform(-1, 1, (batch_size, py_main.T, py_main.N, py_main.n))
    batch_mu = np.random.uniform(-1, 1, (batch_size, py_main.T, py_main.N, py_main.N))
    jax_batch_step = jit(vmap(jax_step, in_axes=(0, 0, 0, 0), out_axes=0))
    stepped = jax_batch_step(batch_x, batch_u, batch_lamda, batch_mu)
    t0 = time()
    for i in range(batch_size):
        cpp_main.step(batch_x[i], batch_u[i], batch_lamda[i], batch_mu[i])
    cpp_dt = time() - t0
    t0 = time()
    stepped = jax_batch_step(batch_x, batch_u, batch_lamda, batch_mu)  # ~4x faster
    jax_dt = time() - t0

    with ThreadPoolExecutor() as executor:
        t0 = time()
        results = list(
            executor.map(
                lambda i: cpp_main.step(batch_x[i], batch_u[i], batch_lamda[i], batch_mu[i]),
                range(batch_size)))
        cpp_threaded_dt = time() - t0
    print(f'{cpp_dt=}')
    print(f'{jax_dt=}')
    print(f'{cpp_threaded_dt=}')

    # assert_allclose(py_pos, cpp_pos)
