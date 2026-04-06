""" Residual Descent Differential Dynamic Game Solver (RD3G) with JAX """
# pylint: disable=invalid-name, forgotten-debug-statement

import logging
from time import time
from dataclasses import dataclass
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, jacrev, vmap
import matplotlib.pyplot as plt

from rd3g.utilities.time_util import TimeUtil
from rd3g.core.base_solver import BaseSolver, BaseSolverConfig, Solution
from rd3g.core.base_game import BaseGame

logger = logging.getLogger('RD3GJax')
logger.setLevel(logging.INFO)


@dataclass(frozen=True)
class RD3GJaxConfig(BaseSolverConfig):
    """Configs for Residual Game."""
    tolerance: float = 5e-4
    iterations: int = 30
    # backtracking line search param
    bc_a: float = 0.1  # alpha
    bc_b: float = 0.5  # beta
    backtracking_max_iter: int = 20
    # NOTE this is not implemented in cpp
    dynamics_residual_weight: float = 1.0
    # barrier function scaling schedule
    rho_0: float = 20.0
    # scaling rate for rho, rho+ = rho * rho_b
    rho_b: float = 1.0


class RD3GJax(BaseSolver):
    """Residual Descent Differential Dynamic Game Solver (RD3G) with JAX

    Attributes:
        config (ResidualGameConfig): Configuration nametuple to
            specify solver iterations, cpp binary, tolerance etc.
        N (int): Number of agents
        T (int): Horizon length
        dt (float): Time step length
        n (int): Dimension of state for a single agent
        m (int): Dimension of control for a single agent
        x0 (np.ndarray): [N, n] Initial State for all agents
        guess (np.ndarray): [T,N,m] Initial guess for control traj

    """

    def __init__(self, config: RD3GJaxConfig, game: BaseGame):
        BaseSolver.__init__(self, config, game)

        self.N = self.game.config.N
        self.T = self.game.config.T
        self.dt = self.game.config.dt
        self.n = self.game.config.n
        self.m = self.game.config.m
        self.x0 = self.game.config.x0

        self.dim_theta = self.T * self.N * self.m
        self.guess = np.zeros((self.T, self.N, self.m))
        self.violations = None

        self.rho = self.config.rho_0
        self.profiler = TimeUtil(True)
        # logger.debug_enable()
        self.residual_vec = []

        # Define jax functions
        self.dLLi_dx = jit(jacrev(self.LLi, argnums=0))
        self.dLLi_du = jit(jacrev(self.LLi, argnums=1))
        self.dL_dx_ik = jit(lambda *args: jacrev(self.L, argnums=0)
                            (*args)[args[-1]])
        self.dL_dx_ik1 = jit(jacrev(self.L, argnums=2))

        h_map_i = vmap(lambda x, k, i, j: self.game.h(
            x[k, i], x[k, j]), in_axes=(None, None, 0, None), out_axes=0)
        h_map_ij = vmap(h_map_i, in_axes=(None, None, None, 0), out_axes=0)
        h_map_kij = vmap(h_map_ij, in_axes=(None, 0, None, None), out_axes=0)

        self.h_map_fun = jit(lambda x: h_map_kij(x, jnp.arange(self.T),
                                                 jnp.arange(self.N),
                                                 jnp.arange(self.N)),
                             )
        self.validate()

    def validate(self):
        """Check the dimension of initial state x0, guess for control."""
        assert self.x0.shape == (self.N, self.n), (
            'Incorrect self.x0 dimension, '
            f'should be {(self.N, self.n)}, but got {self.x0.shape}')
        assert self.guess.shape == (self.T, self.N, self.m), (
            'Incorrect self.guess dimension, '
            f'should be {(self.T, self.N, self.m)}, but got {self.guess.shape}'
        )
        assert self.dim_theta == self.T * self.N * self.m
        assert isinstance(self.n, int) and self.n > 0
        assert isinstance(self.m, int) and self.m > 0
        assert isinstance(self.T, int) and self.T > 0
        assert isinstance(self.N, int) and self.N > 0
        return

    def init(self):
        """Setup solver parameters that changes between iterations, call this
        funtion to reset the solver."""
        raise NotImplementedError

    def jax_compile(self):
        """ Run key functions once to compile """
        logger.info('Compiling functions with jax jit')
        t0 = time()
        p = TimeUtil(True)
        p.s()

        u_ref = self.guess
        # x_ref = x_1 .. x_T, NOTE the array index is offset from the math notation
        p.s('rollout')
        x_ref = self.game.rollout(self.x0, u_ref)
        p.e('rollout')
        lambda_ref = np.zeros((self.T, self.N, self.n))
        # defined for all h_k_i_j, but all values may not be used
        mu_ref = np.zeros((self.T, self.N, self.N))
        p.s('step')
        x_ref, u_ref, lambda_ref, mu_ref, _ = self.step(x_ref, u_ref, lambda_ref, mu_ref)
        p.e('step')
        p.s('h_mask')
        h_plus_mask = jnp.where(jnp.eye(self.N),
                                False,
                                self.h_map_fun(x_ref) > 0)
        p.e('h_mask')

        p.s('r')
        _ = self.r(x_ref, u_ref, lambda_ref, mu_ref, h_plus_mask)
        p.e('r')
        p.e()

        dt = time() - t0
        logger.info(f'jax compile {dt=}s')
        p.summary()

    def solve(self):
        N = self.N
        T = self.T
        n = self.n
        m = self.m
        # y: x(T*N*n) ,u(T*N*m), lambda(T,N,n),mu(T,N,N)
        logger.debug(
            f'primal variables:{(T*N*n) +(T*N*m)} dual variables:{(N*T*n)+(T*N*N)}'
        )
        # TODO call all jit function once to compile before profiling
        u_ref = self.guess
        # x_ref = x_1 .. x_T, NOTE the array index is offset from the math notation
        x_ref = self.game.rollout(self.x0, u_ref)
        lambda_ref = np.zeros((T, N, n))
        # defined for all h_k_i_j, but all values may not be used
        mu_ref = np.zeros((T, N, N))

        t0 = time()
        t = self.profiler
        has_converged = False
        for i in range(self.config.iterations):
            logger.info(f'------ iter {i+1} ------')
            t.s()
            t.s('step')
            x_ref, u_ref, lambda_ref, mu_ref, step_size = self.step(
                x_ref, u_ref, lambda_ref, mu_ref)
            t.e('step')
            t.s('h_plus_mask')
            h_plus_mask = jnp.where(jnp.eye(self.N),
                                    False,
                                    self.h_map_fun(x_ref) > 0)
            t.e('h_plus_mask')
            t.s('residual')
            r0 = self.r(x_ref, u_ref, lambda_ref, mu_ref, h_plus_mask)
            t.e('residual')
            t.s('tolerance')
            residual = np.linalg.norm(r0)
            logger.info(f'{residual=}, {step_size=}')
            if residual < self.config.tolerance:
                t.e()
                has_converged = True
                break
            t.e('tolerance')
            # NOTE may not be necessary
            t.s('final rollout')
            x_ref = self.game.rollout(self.x0, u_ref)
            t.e('final rollout')
            t.e()
            logger.debug(f'------ {N} agents, iter {i} ------')

        t_solve = time() - t0
        logger.info(f'Total solve time: {t_solve}s')
        if i == self.config.iterations - 1:
            logger.warning(' algorithm did not reach stopping criterion')
        full_x_ref = np.vstack([self.x0[np.newaxis, :, :], x_ref])

        # NOTE we don't check for optimality in this solver
        sol = Solution(elapsed_time=t_solve,
                       u=u_ref,
                       x=full_x_ref,
                       residual=residual,
                       has_converged=has_converged,
                       is_optimal=has_converged)
        return sol

    def final(self):
        self.profiler.summary()
        plt.plot(self.residual_vec, '*-')
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Residual (exp)')
        plt.show()

    # @partial(jit, static_argnums=0)
    def get_h_plus_mask(self, x):
        ''' Get a matrix mask of currently active constraints
        Args:
            x: (T, N, n) Agent states
        Return:
            h_plus_mask: (N, N) where val(i,j) = h(x_i, h_j) > 0
        '''
        # h(i,i) should not be considered in either h_plus or h_minus
        # we check it in h_minux
        # for x 1-T, NOTE index to retval start from 1
        h_plus_mask = jnp.fromfunction(
            lambda k, i, j: jnp.logical_and(self.game.h(x[k, i], x[k, j]) >= 0, i != j),
            shape=(self.T, self.N, self.N),
            dtype=int
        )
        return h_plus_mask

    # @partial(jit, static_argnums=0)
    def L(self, x_k, u_k_i, x_k1_i, h_k_plus_mask, lamda_k, mu_k, i: int):
        ''' Lagrangian for agent i
        Args:
            x_k: (N,n) state vector at step k
            u_k_i: (n) control vector for agent i at step k
            k_k1_i: (n) state vector for agent i at step k+1
            h_k_plus_mask: (N,N) Boolean matrix, [i,j] True if h(x_i, x_j) > 0
            lamda_k: (N,n) Multiplier for dynamics constraint
            mu_k: (N,N) multiplier for positive h 
            i: agent index i
        '''
        # TODO: refactor, use h_k instead of h_k_plus_mask to avoid calculating h many times
        # feasibility for h>0
        h_plus_comp = jnp.fromfunction(
            lambda j: jnp.where(i != j, mu_k[i, j] * self.game.h(x_k[i], x_k[j]), 0),
            shape=self.N,
            dtype=jnp.int32
        )
        h_plus_comp = jnp.where(h_k_plus_mask[i], h_plus_comp, 0)
        h_plus = jnp.sum(h_plus_comp)

        # barrier for h < 0
        h_minus_comp = jnp.fromfunction(
            lambda j: jnp.where(i != j, self.Bh(x_k[i], x_k[j]), 0),
            shape=self.N,
            dtype=int
        )
        h_minus_comp = jnp.where(~h_k_plus_mask[i], h_minus_comp, 0)
        h_minus = jnp.sum(h_minus_comp)
        dynamics = lamda_k[i].T @ (self.game.f(x_k[i], u_k_i, i).flatten() - x_k1_i)
        return self.game.J(x_k, u_k_i, i) + h_plus + h_minus + dynamics

    @partial(jit, static_argnums=0)
    def LLi(self, x, u, h_plus_mask, lamda, mu, i):
        ''' Lagrangian for agent i across all time steps 1-T
        Args:
            x: (T,N,n) State for all agents, all time step
            u: (T,N,m) Control for all agents, all time step 
            h_plus_mask: (T,N,N) Boolean matrix, [k,i,j] True if h(x_k_i, x_k_j) > 0
            lamda: (T,N,n) Multiplier for dynamics constraint
            mu: (T,N,N) multiplier for positive h 
            i: agent index i
        Return:
            retval: scalar
        '''
        T = self.T
        LLi_val = jnp.sum(
            jnp.fromfunction(
                lambda k: self.L(
                    x[k], u[k+1, i], x[k+1, i], h_plus_mask[k], lamda[k+1], mu[k], i),
                shape=T-1,
                dtype=int
            ),
        )
        # x0 related terms
        lamda_0_i = jax.lax.dynamic_index_in_dim(lamda[0], i, keepdims=False)
        LLi_val += (self.game.J(self.x0, u[0, i], i)
                    + lamda_0_i.T @ (self.game.f(self.x0[i], u[0, i], i).flatten() - x[0, i])
                    )
        # x_T related terms
        LLi_val += self.game.Jfi(x[T - 1], i)
        h_T_val = jnp.fromfunction(
            lambda j: self.game.h(x[T-1, i], x[T-1, j]) * (i != j),
            shape=(self.N,),
            dtype=int
        )
        h_plus_elements = jnp.where(
            h_T_val >= 0,
            mu[T-1, i] * h_T_val,
            0
        )
        h_plus = jnp.sum(h_plus_elements)
        h_T_val_minus_clipped = jnp.where(h_T_val < -1e-100, h_T_val, -1e-100)
        h_minus_elements = -1 / self.rho * jnp.where(
            h_T_val < 0,
            jnp.log(-h_T_val_minus_clipped),
            0
        )
        h_minus = jnp.sum(h_minus_elements)
        LLi_val += h_plus + h_minus
        return LLi_val

    def Bh(self, x_i, x_j):
        h_val = self.game.h(x_i, x_j)
        return -1 / self.rho * jnp.log(-jnp.where(h_val < -1e-100, h_val, -1e-100))

    @partial(jit, static_argnums=0)
    def step(self, x_ref, u_ref, lambda_ref, mu_ref):
        print('jit step()')
        h_map_val = self.h_map_fun(x_ref)
        h_plus_mask = jnp.where(jnp.eye(self.N), False, h_map_val >= 0)

        r0 = self.r(x_ref, u_ref, lambda_ref, mu_ref, h_plus_mask)
        y0 = jnp.hstack([
            x_ref.flatten(),
            u_ref.flatten(),
            lambda_ref.flatten(),
            mu_ref.flatten()
        ])
        Dr = self.dr_dy(x_ref, u_ref, lambda_ref, mu_ref, h_plus_mask)

        N = self.N
        T = self.T
        n = self.n
        m = self.m

        def split_y(y):
            return (y[:T * N * n].reshape(T, N, n),
                    y[T * N * n:T * N * n + T * N * m].reshape(T, N, m), y[
                    T * N * n + T * N * m:T * N * n + T * N * m + N * T * n
                    ].reshape(T, N, n), y[T * N * n + T * N * m + N * T * n:].reshape(
                    T, N, N))

        def r_y_fun(y):
            return self.r(*split_y(y), h_plus_mask)
        # pylint: disable-next=unused-variable
        dy, residual, rank, s = jax.numpy.linalg.lstsq(Dr, y0)
        # line search
        dy = dy.flatten()
        r0_norm = jnp.linalg.norm(r0)
        apriori_h_res = jnp.sum(h_map_val)

        def cond_fun(step):
            y_new = y0 + step * dy
            r_t = r_y_fun(y_new)
            r_t_norm = jnp.linalg.norm(r_t)
            search_h_res = jnp.sum(h_map_val)
            return jnp.logical_or(r_t_norm > (1 - self.config.bc_a * step) * r0_norm,
                                  search_h_res > apriori_h_res)

        def body_fun(step):
            # NOTE do we still need to rollout here? maybe for nonlinear dynamics?
            # x_new, _, _, _ = split_y(y_new)
            # x_new = self.game.rollout(self.x0, u_new)
            # y_new[:dim_x] = x_new.flatten()
            step = step*self.config.bc_b
            return step

        # line search step size
        step = jax.lax.while_loop(cond_fun, body_fun, 1.0)
        y_new = y0 + step * dy

        x_ref, u_ref, lambda_ref, mu_ref = split_y(y_new)

        return x_ref, u_ref, lambda_ref, mu_ref, step

    @partial(jit, static_argnums=0)
    def dr_dy(self, x, u, lamda, mu, h_plus_mask):
        n = self.n
        m = self.m
        # dLLi/dxi, dLLi/dui, f(x,u)-x, h
        full_r_dim = self.T*self.N*(n+m+n+self.N)
        fun = jacrev(self.r, argnums=[0, 1, 2, 3])
        drdx, drdu, drdlamda, drdmu = fun(x, u, lamda, mu, h_plus_mask)
        Dr = jnp.hstack([drdx.reshape(full_r_dim, -1),
                         drdu.reshape(full_r_dim, -1),
                         drdlamda.reshape(full_r_dim, -1),
                         drdmu.reshape(full_r_dim, -1)])
        return Dr

    @partial(jit, static_argnums=0)
    def r(self, x, u, lamda, mu, h_plus_mask):
        """ Find residual vector for KKT conditions"""
        # NOTE we don't do active set here since jax doesn't work with variable size array
        h_val = self.h_map_fun(x)

        def residual_i(i):
            """ residual for agent i """
            # residual for dLLi/dxi and dLLi/dui
            dLLi_dxi_val = self.dLLi_dxi(x, u, h_plus_mask, lamda, mu, i)
            dLLi_dui_val = self.dLLi_dui(x, u, h_plus_mask, lamda, mu, i)
            # residual for f(x0, u0) - x1
            r_f0 = self.game.f(self.x0[i], u[0, i], i).flatten() - x[0, i]

            # residual for f(x_k,u_k) - x k+1
            def _get_fk(k):
                fk = self.game.f(x[k - 1, i], u[k, i], i).flatten() - x[k, i]
                return fk
            get_fk = vmap(_get_fk, in_axes=0)
            r_fk = jnp.hstack(get_fk(jnp.arange(1, self.T)))

            # residual for h(x_i, x_j)
            def _get_h(k):
                mask = jnp.logical_and(h_val[k-1, i] > 0, jnp.eye(self.N)[i] == 0)
                return jnp.where(mask, h_val[k-1, i], 0)
            get_h = vmap(_get_h, in_axes=0)
            r_h = jnp.hstack(get_h(jnp.arange(1, self.T+1)))

            r = jnp.hstack([dLLi_dxi_val.flatten(),
                            dLLi_dui_val.flatten(),
                            r_f0,
                            r_fk,
                            r_h
                            ])
            return r
        get_residual_all_agents = vmap(residual_i, in_axes=(0,))
        r_all_agents = get_residual_all_agents(jnp.arange(self.N))

        return jnp.hstack(r_all_agents)

    @partial(jit, static_argnums=0)
    def dLLi_dxi(self, x, u, h_plus_mask, lamda, mu, i):
        return self.dLLi_dx(x, u, h_plus_mask, lamda, mu, i)[:, i, :].reshape(1, -1)

    @partial(jit, static_argnums=0)
    def dLLi_dui(self, x, u, h_plus_mask, lamda, mu, i):
        return self.dLLi_du(x, u, h_plus_mask, lamda, mu, i)[:, i, :].reshape(1, -1)

    def J_x_ref_fun(self, i):
        return self.config.target_x_ref[i]
