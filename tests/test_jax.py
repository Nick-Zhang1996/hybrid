''' Test Jax '''
import logging
from dataclasses import replace
from time import time
import numpy as np
import jax.numpy as jnp
from jax import jit, jacfwd, jacrev, jacobian, vmap

from rd3g.games.car_merge_kinematic_bicycle import create_random_game as rgame
from rd3g.games.car_merge_kinematic_bicycle import CarMergeKinematicBicycle, CarMergeKinematicBicycleConfig
from rd3g.games.car_merge_kinematic_bicycle_jax import CarMergeKinematicBicycleJaxConfig
from rd3g.games.car_merge_kinematic_bicycle_jax import CarMergeKinematicBicycleJax

from rd3g.solvers.rd3g_jax import RD3GJax, RD3GJaxConfig
from rd3g.solvers.rd3g import RD3G, RD3GConfig

logging.basicConfig(level=logging.INFO)


def assert_flattened_allclose(val1, val2):
    # np.testing.assert_allclose(val1, val2)
    np.testing.assert_allclose(np.array(val1).flatten(), np.array(val2).flatten())


def assert_allclose(val1, val2):
    np.testing.assert_allclose(val1, val2, rtol=1e-3, atol=1e-3)


def jax_game_from_regular(game):
    c = game.config
    target_y = c.target_y
    x_ref = np.zeros((c.N, c.n))
    x_ref[:, 2] = 2.0  # target speed
    x_ref[:, 1] = np.array(target_y)  # target y position
    config = CarMergeKinematicBicycleJaxConfig(
        T=c.T,
        dt=game.dt,
        N=c.N,
        n=c.n,
        m=c.m,
        track_width=c.track_width,
        collision_radius=c.collision_radius,
        x0=jnp.array(c.x0),
        target_x_ref=jnp.array(x_ref),
        J_Qr=jnp.array(c.J_Qr),
        J_Q=jnp.array(c.J_Q),
        J_R=jnp.array(c.J_R)
    )
    return CarMergeKinematicBicycleJax(config)


def copy_game(game):
    c = game.config
    config = CarMergeKinematicBicycleConfig(
        T=c.T,
        dt=c.dt,
        N=c.N,
        n=c.n,
        m=c.m,
        x0=c.x0,
        track_width=c.track_width,
        collision_radius=c.collision_radius,
        target_y=c.target_y,
        J_Qr=c.J_Qr,
        J_Q=c.J_Q,
        J_R=c.J_R
    )
    return CarMergeKinematicBicycle(config)


def test_jax_rollout():
    game = rgame()
    game_jax = jax_game_from_regular(game)
    u = np.random.uniform(-1, 1,  (game_jax.T, game_jax.N, game_jax.m))

    rollout = jit(game_jax.rollout)

    jax_state_traj = rollout(game_jax.x0, u)
    py_state_traj = game.rollout(game.x0, u)
    assert_allclose(jax_state_traj, py_state_traj)


def test_jax_py_equality():
    solver_config = RD3GConfig()
    game = rgame()
    solver = RD3G(solver_config, game)

    cpp_solver_config = replace(solver_config, USE_CPP=True)
    game = copy_game(game)
    game.setup_rd3g_cpp(cpp_solver_config)
    solver_cpp = RD3G(cpp_solver_config, game)

    game_jax = jax_game_from_regular(game)
    solver_jax_config = RD3GJaxConfig()
    solver_jax = RD3GJax(solver_jax_config, game_jax)

    # Create dummy test values
    i = 1
    x = np.random.uniform(-1, 1, (solver.T, solver.N, solver.n))
    u = np.random.uniform(-1, 1,  (solver.T, solver.N, solver.m))
    # u = np.zeros((solver.T, solver.N, solver.m), dtype=float)
    # x = np.vstack([game.x0[np.newaxis, :, :], game.rollout(game.x0, u)])[:-1]
    x_j = np.random.uniform(-1, 1, solver.n)
    x_k = np.random.uniform(-1, 1, (solver.N, solver.n))
    x_k_i = np.random.uniform(-1, 1, solver.n)
    u_k_i = np.random.uniform(-1, 1, solver.m)
    i = 0
    j = 1
    k = 2
    # n
    x_k1_i = np.random.uniform(-1, 1, solver.n)
    # T, N, n
    lamda = np.random.uniform(-1, 1, (solver.T, solver.N, solver.n))
    # T, N, N
    mu = np.random.uniform(-1, 1, (solver.T, solver.N, solver.N))
    lamda_k = lamda[k]
    mu_k = mu[k]

    if False:
        t0 = time()
        val = solver_jax.step(x, u, lamda, mu)
        val[0].block_until_ready()
        dt = time() - t0
        print(f'before compile {dt=}')

        t0 = time()
        val = solver_jax.step(x, u, lamda, mu)
        val[0].block_until_ready()
        dt = time() - t0
        print(f'after compile {dt=}')

    # Create jit version of functions
    f = jit(game_jax.f)
    df_dx = jit(jacobian(game_jax.f, argnums=0))
    df_du = jit(jacobian(game_jax.f, argnums=1))

    # Evaluate py reference values
    py_retval_f = game.f(x_k_i, u_k_i, i)
    py_retval_df_dx = game.df_dx(x_k_i, u_k_i, i)
    py_retval_df_du = game.df_du(x_k_i, u_k_i, i)

    # Compile and evaluate jax values
    jax_retval_f = f(x_k_i, u_k_i, i)
    jax_retval_df_dx = df_dx(x_k_i, u_k_i, i)
    jax_retval_df_du = df_du(x_k_i, u_k_i, i)

    # py impl always returns a matrix, so it will have two dimensions even for vectors
    assert_allclose(py_retval_f.reshape(game.n), jax_retval_f)
    assert_allclose(py_retval_df_dx, jax_retval_df_dx)
    assert_allclose(py_retval_df_du, jax_retval_df_du)

    py_retval_h_plus_mask = solver.get_h_plus_mask(x)
    jax_retval_h_plus_mask = jnp.where(jnp.eye(game.N), False, solver_jax.h_map_fun(x) > 0)
    assert_allclose(py_retval_h_plus_mask, jax_retval_h_plus_mask)

    h_k_plus_mask = jax_retval_h_plus_mask[2]  # slice h_k, k=2
    py_retval_L = solver.L(x_k, u_k_i, x_k1_i, h_k_plus_mask, lamda_k, mu_k, i)
    jax_retval_L = solver_jax.L(jnp.array(x_k), u_k_i, x_k1_i,
                                h_k_plus_mask, lamda_k, jnp.array(mu_k), i)
    assert_allclose(py_retval_L, jax_retval_L)

    py_retval_dL_dx_ik = solver.dL_dx_ik(x_k, u_k_i, x_k1_i, h_k_plus_mask, lamda_k, mu_k, i)
    jax_retval_dL_dx_ik = solver_jax.dL_dx_ik(
        x_k, u_k_i, x_k1_i, h_k_plus_mask, lamda_k, jnp.array(mu_k), i)
    assert_allclose(py_retval_dL_dx_ik[0], jax_retval_dL_dx_ik)

    py_retval_dL_dx_ik1 = solver.dL_dx_ik1(x_k, u_k_i, x_k1_i, h_k_plus_mask, lamda_k, mu_k, i)
    jax_retval_dL_dx_ik1 = solver_jax.dL_dx_ik1(
        x_k, u_k_i, x_k1_i, h_k_plus_mask, lamda_k, mu_k, i)
    assert_allclose(py_retval_dL_dx_ik1, jax_retval_dL_dx_ik1)

    # dr_dy
    # r: dLL_dxi, dLL_dui, f(x,u) - x+
    # r needs dLLi_dxi, dLLi_dui, which needs LLi
    # y:x, u, lamda, mu
    h_plus_mask = jax_retval_h_plus_mask
    py_retval_LLi = solver.LLi(x, u, h_plus_mask, lamda, mu, i)
    jax_retval_LLi = solver_jax.LLi(x, u, h_plus_mask, lamda, mu, i)
    assert_allclose(py_retval_LLi, jax_retval_LLi)

    py_retval_dLLi_dxi = solver.dLLi_dxi(x, u, h_plus_mask, lamda, mu, i)
    jax_retval_dLLi_dxi = solver_jax.dLLi_dxi(x, u, h_plus_mask, lamda, mu, i)
    assert_allclose(py_retval_dLLi_dxi, jax_retval_dLLi_dxi)
    py_retval_dLLi_dui = solver.dLLi_dui(x, u, h_plus_mask, lamda, mu, i)
    jax_retval_dLLi_dui = solver_jax.dLLi_dui(x, u, h_plus_mask, lamda, mu, i)
    assert_allclose(py_retval_dLLi_dui, jax_retval_dLLi_dui)

    # NOTE the dimension is different because jax has static dimension and can't use active set
    t0 = time()
    py_retval_r = solver.r(x, u, lamda, mu, h_plus_mask)  # 0.4s
    py_dt = time()-t0
    py_pos = py_retval_r[np.abs(py_retval_r) > 0]

    jax_retval_r = solver_jax.r(x, u, lamda, mu, h_plus_mask)  # 0.0002s
    jax_pos = jax_retval_r[np.abs(jax_retval_r) > 0]
    assert_allclose(py_pos, jax_pos)

    py_retval_dr_dy = solver.dr_dy(x, u, lamda, mu, h_plus_mask)
    jax_retval_dr_dy = solver_jax.dr_dy(x, u, lamda, mu, h_plus_mask)
    t0 = time()
    jax_retval_dr_dy = solver_jax.dr_dy(x, u, lamda, mu, h_plus_mask)
    dt = time() - t0
    print(f'dr_dy {dt=}')
    # FIXME check supposedly empty row values
    # py version is slightly higher, why?
    # assert_allclose(np.sum(np.abs(py_retval_dr_dy)), np.sum(np.abs(jax_retval_dr_dy)))

    t0 = time()
    py_stepped = solver.step(x, u, lamda, mu)  # 0.13s
    dt = time() - t0
    print(f'step py {dt=}')

    t0 = time()
    try:
        cpp_stepped = solver_cpp.cpp.step(x, u, lamda, mu)  # 0.09s not representative as x is bad
    except StopIteration:
        pass
    dt = time() - t0
    print(f'step cpp {dt=}')

    jax_stepped = solver_jax.step(x, u, lamda, mu)
    t0 = time()
    jax_stepped = solver_jax.step(x, u, lamda, mu)  # 0.11s
    dt = time() - t0
    print(f'step jax{dt=}')

    if False:
        batch_size = 64
        print(f'runtime for vmapped particle x{batch_size}')
        batch_x = np.random.uniform(-1, 1, (batch_size, solver.T, solver.N, solver.n))
        batch_u = np.random.uniform(-1, 1,  (batch_size, solver.T, solver.N, solver.m))
        batch_lamda = np.random.uniform(-1, 1, (batch_size, solver.T, solver.N, solver.n))
        batch_mu = np.random.uniform(-1, 1, (batch_size, solver.T, solver.N, solver.N))
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
