""" Residual Descent Differential Dynamic Game Solver (RD3G)"""
# pylint: disable=invalid-name, forgotten-debug-statement

import logging
from time import time
from dataclasses import dataclass
from itertools import chain

import numpy as np
import scipy.sparse  # sparse matrix operations
import scipy.sparse.linalg
import matplotlib.pyplot as plt

from rd3g.utilities.util import jacobian_numerical
from rd3g.utilities.time_util import TimeUtil
from rd3g.core.base_solver import BaseSolver, BaseSolverConfig, Solution
from rd3g.core.base_game import BaseGame

logger = logging.getLogger('ResidualGame')
logger.setLevel(logging.INFO)


@dataclass(frozen=True)
class RD3GConfig(BaseSolverConfig):
    """Configs for Residual Game."""
    USE_CPP: bool = False
    CPP_DEBUG: bool = False
    DEBUG: bool = False
    FORCE_PYTHON_SOLVER: bool = False
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


class RD3G(BaseSolver):
    """Residual Descent Differential Dynamic Game Solver (RD3G)

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

    def __init__(self, config: RD3GConfig, game: BaseGame):
        BaseSolver.__init__(self, config, game)
        self.USE_CPP = config.USE_CPP

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
        self.profiler = TimeUtil(False)
        # logger.debug_enable()
        self.residual_vec = []
        if self.config.USE_CPP:
            self.cpp = self.game.setup_rd3g_cpp(config)
            if self.cpp is None:
                logger.error('User specified USE_CPP, but cpp backend for this game is unavailable')
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

    def cpp_solve(self):
        logger.debug('solve using cpp.solve()')
        u_ref = self.guess
        t0 = time()
        retval = self.cpp.solve(u_ref)
        x_ref, u_ref, lambda_ref, mu_ref = [
            np.array(val) for val in retval[:-1]
        ]
        has_converged = retval[-1]
        logger.debug(f'has_converged: {has_converged}')
        t_solve = time() - t0
        logger.info(f'Total solve time: {t_solve}s')

        h_plus_mask = self.get_h_plus_mask(x_ref)
        r0 = self.r(x_ref, u_ref, lambda_ref, mu_ref, h_plus_mask)
        r0_norm = np.linalg.norm(r0)
        logger.debug(f' residual = {r0_norm}')
        return

    def solve(self):
        """main entry point for solver, will call cpp version if available,
        will fallback to python if cpp does not provide a solution, I forgot
        why I did the fallback."""
        logger.info(f'USE_CPP: {self.config.USE_CPP}')
        logger.info(f'FORCE_PYTHON_SOLVER: {self.config.FORCE_PYTHON_SOLVER}')

        N = self.N
        T = self.T
        n = self.n
        m = self.m
        # y: x(T*N*n) ,u(T*N*m), lambda(T,N,n),mu(T,N,N)
        logger.debug(
            f'primal variables:{(T*N*n) +(T*N*m)} dual variables:{(N*T*n)+(T*N*N)}'
        )

        u_ref = self.guess
        # x_ref = x_1 .. x_T, NOTE the array index is offset from the math notation
        x_ref = self.game.rollout(self.x0, u_ref)
        lambda_ref = np.zeros((T, N, self.n))
        # defined for all h_k_i_j, but all values may not be used
        mu_ref = np.zeros((T, N, N))
        t = self.profiler
        has_converged = False
        t0 = time()
        for i in range(self.config.iterations):
            logger.info(f'------ iter {i+1} ------')
            t.s()
            if self.config.USE_CPP and not self.config.FORCE_PYTHON_SOLVER:
                t.s('cpp step')
                try:
                    try:
                        retval = self.cpp.step(x_ref, u_ref, lambda_ref,
                                               mu_ref)
                        x_ref, u_ref, lambda_ref, mu_ref = [
                            np.array(val) for val in retval
                        ]
                        # put update here because in case solver failed,
                        # self.step() will call cpp.post_step_update()
                        self.cpp.post_step_update()
                    except RuntimeError as e:
                        logger.warning('-----------------------------------')
                        logger.warning(f'cpp.step() error: {e}')
                        logger.warning('-----------------------------------')
                        x_ref, u_ref, lambda_ref, mu_ref = self.step(
                            x_ref, u_ref, lambda_ref, mu_ref)
                except StopIteration as e:
                    logger.debug(e)
                    if 'criteria met' in str(e):
                        has_converged = True
                    break
                finally:
                    t.e('cpp step')
            else:
                try:
                    x_ref, u_ref, lambda_ref, mu_ref = self.step(
                        x_ref, u_ref, lambda_ref, mu_ref)
                except StopIteration as e:
                    logger.debug(e)
                    has_converged = True
                    break
            # NOTE may not be necessary
            x_ref = self.game.rollout(self.x0, u_ref)
            t.e()
            logger.debug(f'------ {N} agents, iter {i} ------')

        t_solve = time() - t0
        logger.info(f'Total solve time: {t_solve}s')
        if i == self.config.iterations - 1:
            logger.warning(' algorithm did not reach stopping criterion ')
        full_x_ref = np.vstack([self.x0[np.newaxis, :, :], x_ref])

        # check second order conditions
        h_plus_mask = np.zeros((self.T, self.N, self.N), dtype=bool)
        for i in range(self.N):
            # x: T,N,n
            idx = []
            for k in range(self.T):
                idx.append(
                    range(k * self.N * self.n + i * self.n,
                          k * self.N * self.n + (i + 1) * self.n))
            idx = [i for item in idx for i in item]

            # dLL / dxdx, hessian
            M = self.dLLi_dxi_dx(x_ref, u_ref, h_plus_mask, lambda_ref, mu_ref,
                                 i)[:, idx]
            pde = not np.all(np.linalg.eigvals(M) < 1e-3)
            logger.debug(f'{i} eig val: {np.linalg.eigvals(M)}')
            # check the second order condition for J at each time step
            # for k in range(self.T):
            #     H = self.dJfi_dxi_dxi(x_ref[k], i)
            #     logger.info(f'{i, k} eig val: {np.linalg.eigvals(H)}')
            has_converged = pde and has_converged
        r0 = self.r(x_ref, u_ref, lambda_ref, mu_ref, h_plus_mask)
        r0 = np.linalg.norm(r0)

        sol = Solution(elapsed_time=t_solve,
                       u=u_ref,
                       x=full_x_ref,
                       residual=r0,
                       has_converged=has_converged,
                       is_optimal=has_converged)

        return sol

    def step(self, x_ref, u_ref, lambda_ref, mu_ref):
        t = self.profiler
        t.s('setup')
        N = self.N
        T = self.T
        n = self.n
        m = self.m
        dim_x = T * N * n
        # r0 + Dr*dr = 0
        h_plus_mask = self.get_h_plus_mask(x_ref)

        r0 = self.r(x_ref, u_ref, lambda_ref, mu_ref, h_plus_mask)
        y0 = np.hstack([
            x_ref.flatten(),
            u_ref.flatten(),
            lambda_ref.flatten(),
            mu_ref.flatten()
        ])
        # x,u,lamda,mu = split_y(y)

        def split_y(y):
            return (y[:T * N * n].reshape(T, N, n),
                    y[T * N * n:T * N * n + T * N * m].reshape(T, N, m), y[
                    T * N * n + T * N * m:T * N * n + T * N * m + N * T * n
                    ].reshape(T, N, n), y[T * N * n + T * N * m + N * T * n:].reshape(
                    T, N, N))

        def r_y_fun(y):
            return self.r(*split_y(y), h_plus_mask)

        t.e('setup')
        Dr = self.dr_dy(x_ref, u_ref, lambda_ref, mu_ref, h_plus_mask)
        # after we remove the cols associated with unused mu, Dr will be square

        if self.config.DEBUG:
            t0 = time()
            Dr_alt = jacobian_numerical(r_y_fun, y0, dim=r0.shape[0])
            logger.debug(f't: Dr numerical {time()-t0}')
            logger.debug(np.linalg.norm(Dr - Dr_alt))
            assert np.linalg.norm(Dr - Dr_alt) < 1e-4

        # find newton direction, dense matrix
        # t.s('lstsq')
        # dy, residuals, rank, s = np.linalg.lstsq(Dr,-r0)
        # t.e('lstsq')
        # find newton direction, Sparse lsqr
        # t.s('sparse-lstsq')
        # sparse_Dr = scipy.sparse.csc_matrix(Dr, dtype=float)
        # dy, istop, itn, normr = scipy.sparse.linalg.lsqr(sparse_Dr,-r0)[:4]
        # t.e('sparse-lstsq')

        # remove zero col/rows first, then use Sparse lsqr
        t.s('nonzero reduction')
        nonzero_rows = np.nonzero(np.sum(np.abs(Dr), axis=1))[0]
        nonzero_cols = np.nonzero(np.sum(np.abs(Dr), axis=0))[0]
        reduced_Dr = Dr[nonzero_rows, :][:, nonzero_cols]
        t.e('nonzero reduction')
        # NOTE debug heatmap
        # abs_matrix = np.abs(reduced_Dr)
        # # Plotting the heatmap
        # plt.imshow(abs_matrix, cmap='viridis', interpolation='none')
        # # Adding a color bar
        # plt.colorbar(label='Absolute Value')
        # plt.title('Heatmap of Matrix Values')
        # plt.xlabel('Column Index')
        # plt.ylabel('Row Index')
        # plt.show()

        # use python's Sparse lsqr
        t.s('reduced-sparse-lstsq')
        sparse_Dr = scipy.sparse.csc_matrix(reduced_Dr, dtype=float)
        reduced_dy, istop, itn, normr = scipy.sparse.linalg.lsqr(
            sparse_Dr, -r0[nonzero_rows])[:4]
        del normr
        t.e('reduced-sparse-lstsq')
        logger.debug(
            f'Solver status: {"exact solution" if istop==1 else "Least Square Solution"},'
            f'iterations: {itn}'
        )
        # use cpp's sparse QR
        # t.s('cpp SparseQR')
        # reduced_dy_sqr = self.cpp.SparseQR(reduced_Dr, -r0[nonzero_rows])
        # t.e('cpp SparseQR')

        # use cpp's lscg (fastest)
        # if self.config.USE_CPP:
        #     t.s('cpp lscg')
        #     reduced_dy = self.cpp.LeastSquaresConjugateGradient(reduced_Dr, -r0[nonzero_rows])
        #     t.e('cpp lscg')

        dy = np.zeros_like(y0)
        dy[nonzero_cols] = reduced_dy.flatten()

        # projection onto dynamics null space
        # extract control constraint F
        # assert that x,u are separated from the rest
        # F @ [x,u] = Fx @ x + Fu @ u= -r_F
        # NOTE this is extremely expensive, only do this if we can't obtain an exact solution
        if istop == 2:
            index = 0
            for i in range(self.N):
                index += T * n + T * m
                # x: T*N*n
                x_indices = list(
                    chain.from_iterable([
                        list(range(t * N * n + i * n, t * N * n + (i + 1) * n))
                        for t in range(T)
                    ]))
                # u: T*N*m
                u_indices = list(
                    chain.from_iterable([
                        list(
                            range(dim_x + t * N * m + i * m,
                                  dim_x + t * N * m + (i + 1) * m))
                        for t in range(T)
                    ]))

                Fx = Dr[index:index + n * T, x_indices]
                Fu = Dr[index:index + n * T, u_indices]
                dx_i = dy[x_indices].flatten()
                du_i = dy[u_indices].flatten()
                F = np.hstack([Fx, Fu])
                z = np.hstack([dx_i, du_i])[:, np.newaxis]
                FFT_inv = np.linalg.inv(
                    F @ F.T
                )  # TODO add regularization if this in singular, or use pseudoinverse
                z_null = (np.eye(z.shape[0]) - F.T @ FFT_inv @ F) @ z
                dx_i_after = z_null[:dx_i.shape[0], 0]
                du_i_after = z_null[dx_i.shape[0]:, 0]

                # apriori = Fu @ du_i + Fx @ dx_i
                # posterior = Fu @ du_i_after + Fx @ dx_i_after
                dy[u_indices] = du_i_after
                dy[x_indices] = dx_i_after
                index += n * T + np.sum(h_plus_mask[:, i])

        # Backtracking line search
        t.s('line search')
        apriori_h_res = self.get_collision_residual(x_ref)  # NOTE optimize?
        # backtracking line search
        step = 1.0  # step size
        dy = dy.flatten()
        r0_norm = np.linalg.norm(r0)
        flag_no_step = True
        for i in range(self.config.backtracking_max_iter):
            y_new = y0 + step * dy
            x_new, _, _, _ = split_y(y_new)
            # NOTE do we still need to rollout here? maybe for nonlinear dynamics?
            # x_new = self.rollout(self.x0, u_new)
            # y_new[:dim_x] = x_new.flatten()
            search_h_res = self.get_collision_residual(x_new)
            r_t = r_y_fun(y_new)
            r_t_norm = np.linalg.norm(r_t)
            if (r_t_norm > (1 - self.config.bc_a * step) * r0_norm
                    or search_h_res > apriori_h_res):
                step *= self.config.bc_b
            else:
                flag_no_step = False
                break
        t.e('line search')

        self.residual_vec.append(r0_norm)
        self.violations = violations = np.sum(h_plus_mask) / 2
        expected_posterior_norm = np.linalg.norm(r0 + Dr @ dy)
        logger.info(
            f'r0_norm {r0_norm} expected full step {expected_posterior_norm}'
            f'rt_norm {r_t_norm}, h>0 {violations}'
        )

        # stopping criterion
        if np.abs(r_t_norm) < self.config.tolerance and violations == 0:
            raise StopIteration('stopping criterion met!')
        if flag_no_step:
            raise StopIteration('iteration not making progress')

        self.rho *= self.config.rho_b

        if self.config.USE_CPP:
            # normally we won't reach here because we'd use  the cpp.step(),
            # but if we are only using the "subfunctions", then this will be called
            self.cpp.post_step_update()

        return split_y(y_new)

    def final(self):
        self.profiler.summary()
        if self.config.USE_CPP:
            self.cpp.summary()
        plt.plot(self.residual_vec, '*-')
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Residual (exp)')
        plt.show()

    def dr_dy(self, x, u, lamda, mu, h_plus_mask):
        t = self.profiler
        if self.config.USE_CPP:
            # t.s('drdy-stacked')
            # drdx = self.cpp.dr_dx(x, u, lamda, mu, h_plus_mask)
            # drdu = self.cpp.dr_du(x, u, lamda, mu, h_plus_mask)
            # drdlamda = self.cpp.dr_dlamda(x, u, lamda, mu, h_plus_mask)
            # drdmu = self.cpp.dr_dmu(x, u, lamda, mu, h_plus_mask)
            # Dr = np.hstack([drdx,drdu,drdlamda,drdmu])
            # t.e('drdy-stacked')
            t.s('drdy-cpp')
            Dr = self.cpp.dr_dy(x, u, lamda, mu, h_plus_mask)
            t.e('drdy-cpp')
        else:
            t.s('drdx')
            drdx = self.dr_dx(x, u, lamda, mu, h_plus_mask)
            t.e('drdx')
            t.s('drdu')
            drdu = self.dr_du(x, u, lamda, mu, h_plus_mask)
            t.e('drdu')
            t.s('drdlamda')
            drdlamda = self.dr_dlamda(x, u, lamda, mu, h_plus_mask)
            t.e('drdlamda')
            t.s('drdmu')
            drdmu = self.dr_dmu(x, u, lamda, mu, h_plus_mask)
            t.e('drdmu')
            t.s('stack')
            Dr = np.hstack([drdx, drdu, drdlamda, drdmu])
            t.e('stack')
        return Dr

    def get_collision_residual(self, x):
        h_res = 0
        for k in range(1, self.T + 1):
            for i in range(self.N):
                for j in range(i + 1, self.N):
                    this_h = self.game.h(x[k - 1, i], x[k - 1, j])
                    if this_h > 0:
                        h_res += this_h
        return h_res

    def get_h_plus_mask(self, x):
        ''' Get a matrix mask of currently active constraints
        Args:
            x: (T, N, n) Agent states
        Return:
            retval: (N, N) where val(i,j) = h(x_i, h_j) > 0
        '''
        h_plus_mask = np.zeros((self.T, self.N, self.N), dtype=bool)
        for k in range(0, self.T):
            for i in range(self.N):
                for j in range(i + 1, self.N):
                    h_plus_mask[k, i,
                                j] = h_plus_mask[k, j, i] = self.game.h(
                                    x[k, i], x[k, j]) >= 0
        if self.config.CPP_DEBUG:
            alt = self.cpp.getHplusMask([xx for xx in x])
            if np.linalg.norm(alt - h_plus_mask) > 1e-4:
                breakpoint()

        return h_plus_mask

    # ----- derivatives and other generic math functions ----
    def L(self, x_k, u_k_i, x_k1_i, h_k_plus_mask, lamda_k, mu_k, i):
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
        # feasibility for h>0
        h_plus = np.sum([
            mu_k[i, j.item()] * (self.game.h(x_k[i], x_k[j.item()]))
            for j in np.nonzero(h_k_plus_mask[i])[0]
        ],
            axis=0)
        if self.config.CPP_DEBUG:
            for j in np.nonzero(h_k_plus_mask[i])[0]:
                val = self.game.h(x_k[i], x_k[j.item()])
                val_cpp = self.cpp.h(x_k[i], x_k[j.item()])
                if np.linalg.norm(val - val_cpp) > 1e-4:
                    breakpoint()

        # barrier for h < 0
        h_minus = -1 / self.rho * np.sum([
            np.log(-min(self.game.h(x_k[i], x_k[j.item()]), -1e-100))
            if j.item() != i else 0 for j in np.nonzero(~h_k_plus_mask[i])[0]
        ])
        dynamics = lamda_k[i].T @ (self.game.f(x_k[i], u_k_i, i).flatten() - x_k1_i)
        return self.game.J(x_k, u_k_i, i) + h_plus + h_minus + dynamics

    def dL_dx_ik(self, x_k, u_k_i, x_k1_i, h_k_plus_mask, lamda_k, mu_k, i):
        """dL / dx_i_k
        Args:
            x_k: [N, n]
            u_k_i: [m]
            x_k1_i: x_k+1_i: [n]
            lambda_k:
            mu_k: [N,N]
            i: agent index
        """
        if self.config.USE_CPP:
            return self.cpp.dL_dx_ik(x_k, u_k_i, x_k1_i, h_k_plus_mask,
                                     lamda_k, mu_k, i)

        val = self.game.dJi_dxi(x_k, u_k_i,
                                i) + lamda_k[i].T @ self.game.df_dx(x_k[i], u_k_i, i)
        val += np.sum([
            mu_k[i, j.item()] * (self.game.dh_dxi(x_k[i], x_k[j.item()]))
            for j in np.nonzero(h_k_plus_mask[i])[0]
        ],
            axis=0)
        val += -1.0 / self.rho * np.sum([
            min(1 / self.game.h(x_k[i], x_k[j.item()]), 1e20) *
            self.game.dh_dxi(x_k[i], x_k[j.item()]) * (j.item() != i)
            for j in np.nonzero(~h_k_plus_mask[i])[0]
        ],
            axis=0)

        # NOTE the behavior of barrier function near boundary may need tuning
        if self.config.DEBUG:
            # dJi_dx -- passed
            num = jacobian_numerical(
                lambda xx: self.game.J(xx.reshape(x_k.shape), u_k_i, i),
                x_k.flatten())
            num = num.reshape((1, self.N, self.n))[:, i]
            ana = self.game.dJi_dxi(x_k, u_k_i, i)
            assert np.linalg.norm(num - ana) < 1e-4
            # df_dx -- inconclusive
            num = jacobian_numerical(lambda uu: self.game.J(x_k, uu, i), u_k_i)
            ana = self.game.dJi_du(x_k, u_k_i, i)
            assert np.linalg.norm(num - ana) < 1e-4
            # dh_dxi -- inconclusive
            for j in np.nonzero(h_k_plus_mask[i])[0]:
                if i == j:
                    continue
                ana = self.game.dh_dxi(x_k[i], x_k[j])
                # pylint: disable-next=cell-var-from-loop
                num = jacobian_numerical(lambda xx: self.game.h(xx, x_k[j]), x_k[i])
                assert np.linalg.norm(num - ana) < 1e-4

            num = jacobian_numerical(
                lambda xx: self.L(xx.reshape(x_k.shape), u_k_i, x_k1_i,
                                  h_k_plus_mask, lamda_k, mu_k, i),
                x_k.flatten())
            num = num[0, i * self.n:(i + 1) * self.n]
            assert np.linalg.norm(num - val) < 1e-4

        if self.config.CPP_DEBUG:
            alt = self.cpp.dJi_dxi(x_k, u_k_i, i)
            if np.linalg.norm(self.game.dJi_dxi(x_k, u_k_i, i) - alt) > 1e-4:
                breakpoint()
            alt = self.cpp.dL_dx_ik(x_k, u_k_i, x_k1_i, h_k_plus_mask, lamda_k,
                                    mu_k, i)
            if np.linalg.norm(val - alt) > 1e-4:
                breakpoint()
        return val

    def dL_dx_ik1(self, x_k, u_k_i, x_k1_i, h_k_plus_mask, lamda_k, mu_k, i):
        """dL/dx_i_k+1."""
        del x_k
        del u_k_i
        del x_k1_i
        del h_k_plus_mask
        del mu_k
        return -lamda_k[i].T

    # NOTE deprecated, usually dJi_du is called directly
    def dL_du(self, x_k, u_k_i, x_k1_i, h_k_plus_mask, lamda_k, mu_k, i):
        del x_k1_i
        del h_k_plus_mask
        del mu_k
        val = self.game.dJi_du(x_k, u_k_i,
                               i) + lamda_k[i].T @ self.game.df_du(x_k[i], u_k_i, i)
        return val

    def LLi(self, x, u, h_plus_mask, lamda, mu, i):
        ''' Lagrangian for agent i across all time steps
        only used in debug, LLi's derivative is used more prevalently
        '''
        T = self.T
        LLi_val = np.sum([
            self.L(x[k - 1], u[k, i], x[k, i], h_plus_mask[k - 1], lamda[k],
                   mu[k - 1], i) for k in range(1, T)
        ],
            axis=0)

        # x0 related terms
        LLi_val += self.game.J(
            self.x0, u[0, i],
            i) + lamda[0, i].T @ (self.game.f(self.x0[i], u[0, i], i).flatten() - x[0, i])
        # x_T related terms
        LLi_val += self.game.Jfi(x[T - 1], i)

        h_plus = np.sum([
            mu[T - 1, i, j.item()] *
            (self.game.h(x[T - 1, i], x[T - 1, j.item()]))
            for j in np.nonzero(h_plus_mask[T - 1, i])[0]
        ],
            axis=0)
        h_minus = -1 / self.rho * np.sum([
            np.log(-min(self.game.h(x[T - 1, i], x[T - 1, j.item()]), -1e-100)) *
            (j.item() != i) for j in np.nonzero(~h_plus_mask[T - 1, i])[0]
        ],
            axis=0)
        LLi_val += h_plus + h_minus
        return LLi_val

    def dLLi_dxi(self, x, u, h_plus_mask, lamda, mu, i):
        ''' return: 1*(T*n)  Note index of x starts with 1'''
        if self.config.USE_CPP:
            return self.cpp.dLLi_dxi([xx for xx in x], [uu for uu in u],
                                     [hh for hh in h_plus_mask],
                                     [ll for ll in lamda], [mmm
                                                            for mmm in mu], i)
        T = self.T
        N = self.N
        n = self.n
        der = np.zeros(T * n)

        def submtx_k(k):
            return der[(k - 1) * n:k * n]
        # dLLi_dxi
        for k in range(1, T):
            sub = submtx_k(k)
            sub[:] = self.dL_dx_ik(x[k - 1], u[k, i], x[k,
                                                        i], h_plus_mask[k - 1],
                                   lamda[k], mu[k - 1], i) - lamda[k - 1, i].T
            if self.config.DEBUG:
                # pylint: disable=cell-var-from-loop
                num = jacobian_numerical(
                    lambda xx: self.L(xx.reshape(N, n), u[k, i], x[
                        k, i], h_plus_mask[k - 1], lamda[k], mu[k - 1], i),
                    x[k - 1].flatten())
                num = num[0, i * n:(i + 1) * n] - lamda[k - 1, i].T
                assert np.linalg.norm(num - sub) < 1e-4

        # dLLi_dxi_T
        sub = submtx_k(T)
        h_pos_term = np.sum(
            [
                mu[T-1, i, j.item()] * (self.game.dh_dxi(x[T-1, i], x[T-1, j.item()]))
                for j in np.nonzero(h_plus_mask[T-1, i])[0]
            ],
            axis=0
        )

        h_neg_term = - 1/self.rho*np.sum(
            [
                min(1/self.game.h(x[T-1, i], x[T-1, j.item()]), 1e20)*self.game.dh_dxi(
                    x[T-1, i], x[T-1, j.item()]) * (j.item() != i)
                for j in np.nonzero(~h_plus_mask[T-1, i])[0]
            ],
            axis=0
        )

        sub[:] = -lamda[T-1, i].T + \
            self.game.dJfi_dxi(x[T-1], i) + h_pos_term + h_neg_term

        val = der.reshape(1, -1)
        if self.config.CPP_DEBUG:
            alt = self.cpp.dLLi_dxi([xx for xx in x], [uu for uu in u],
                                    [hh for hh in h_plus_mask],
                                    [ll for ll in lamda], [mmm
                                                           for mmm in mu], i)
            if np.linalg.norm(alt - val) > 1e-4:
                breakpoint()
        return val

    def dLLi_dui(self, x, u, h_plus_mask, lamda, mu, i):
        ''' return: 1*(T*m) '''
        if self.config.USE_CPP:
            return self.cpp.dLLi_dui([xx for xx in x], [uu for uu in u],
                                     [hh for hh in h_plus_mask],
                                     [ll for ll in lamda], [mmm
                                                            for mmm in mu], i)
        T = self.T
        m = self.m
        der = np.zeros(T * m)

        def submtx_k(k):
            return der[k * m:(k + 1) * m]
        # dLLi_dui_0
        sub = submtx_k(0)
        sub[:] = self.game.dJi_du(self.x0, u[0, i], i) + lamda[0, i].T @ self.game.df_du(
            self.x0[i], u[0, i], i)
        # dLLi_dui_k
        for k in range(1, T):
            sub = submtx_k(k)
            # dL_du
            sub[:] = self.game.dJi_du(
                x[k - 1], u[k, i],
                i) + lamda[k, i].T @ self.game.df_du(x[k - 1, i], u[k, i], i)
        val = der.reshape(1, -1)
        if self.config.CPP_DEBUG:
            alt = self.cpp.dLLi_dui([xx for xx in x], [uu for uu in u],
                                    [hh for hh in h_plus_mask],
                                    [ll for ll in lamda], [mmm
                                                           for mmm in mu], i)
            if np.linalg.norm(alt - der) > 1e-4:
                breakpoint()
        return val

    def dLLi_dxi_dmu(self, x, u, h_plus_mask, lamda, mu, i):
        ''' return: dim: dim_x*dim_mu '''
        if self.config.USE_CPP:
            return self.cpp.dLLi_dxi_dmu([xx for xx in x], [uu for uu in u],
                                         [hh for hh in h_plus_mask],
                                         [ll for ll in lamda],
                                         [mmm for mmm in mu], i)
        T = self.T
        N = self.N
        n = self.n
        dim_mu = T * N * N
        dLLi_dxi_dmu = np.zeros((T * n, dim_mu))
        for k in range(1, T + 1):
            for j in np.nonzero(h_plus_mask[k - 1, i])[0]:
                dLLi_dxki_dmuijk = self.game.dh_dxi(x[k - 1, i], x[k - 1, j])
                dLLi_dxi_dmu[(k - 1) * n:k * n,
                             (k - 1) * N * N + i * N + j] = dLLi_dxki_dmuijk
        if self.config.CPP_DEBUG:
            alt = self.cpp.dLLi_dxi_dmu([xx for xx in x], [uu for uu in u],
                                        [hh for hh in h_plus_mask],
                                        [ll for ll in lamda],
                                        [mmm for mmm in mu], i)
            if np.linalg.norm(alt - dLLi_dxi_dmu) > 1e-4:
                breakpoint()
        return dLLi_dxi_dmu

    def r(self, x, u, lamda, mu, h_plus_mask):
        if self.config.USE_CPP:
            return self.cpp.r([xx for xx in x], [uu for uu in u],
                              [ll for ll in lamda], [mmm for mmm in mu],
                              [hh for hh in h_plus_mask])
        T = self.T
        r = np.zeros(0)
        for i in range(self.N):
            dLL_dxi = self.dLLi_dxi(x, u, h_plus_mask, lamda, mu, i)
            dLL_dui = self.dLLi_dui(x, u, h_plus_mask, lamda, mu, i)
            # this needs to be updated
            # if self.config.DEBUG:
            #     dLL_du_num = jacobianNumerical(lambda uu:self.LLi(x,uu.reshape(u.shape),
            #                               h_plus_mask,lamda,mu,i), u.flatten())
            #     assert(np.linalg.norm(dLL_du-dLL_du_num)<1e-4)
            #     dLL_dx_num = jacobianNumerical(lambda xx:self.LLi(xx.reshape(x.shape),
            #                                   u,h_plus_mask,lamda,mu,i), x.flatten())
            #     assert(np.linalg.norm(dLL_dx-dLL_dx_num)<1e-4)
            r = np.hstack([r, dLL_dxi.flatten(), dLL_dui.flatten()])
            # dynamics for f(x0,u0) = x1
            r = np.hstack([
                r, self.config.dynamics_residual_weight *
                self.game.f(self.x0[i], u[0, i], i).flatten() - x[0, i]
            ])
            for k in range(1, self.T):
                r = np.hstack([
                    r, self.config.dynamics_residual_weight *
                    self.game.f(x[k - 1, i], u[k, i], i).flatten() - x[k, i]
                ])  # dual for dynamics
            for k in range(1, self.T):
                r = np.hstack([r] + [
                    self.game.h(x[k - 1, i], x[k - 1, j.item()])
                    for j in np.nonzero(h_plus_mask[k - 1, i])[0]
                ])
            # h(x_T_i, x_T_j)
            r = np.hstack([r] + [
                self.game.h(x[T - 1, i], x[T - 1, j.item()])
                for j in np.nonzero(h_plus_mask[T - 1, i])[0]
            ])

        if self.config.CPP_DEBUG:
            alt = self.cpp.r([xx for xx in x], [uu for uu in u],
                             [ll for ll in lamda], [mmm for mmm in mu],
                             [hh for hh in h_plus_mask])
            if np.linalg.norm(alt.flatten() - r) > 1e-4:
                breakpoint()
        return r

    def Bh(self, x_i, x_j):
        h_val = self.game.h(x_i, x_j)
        return -1 / self.rho * np.log(-np.where(h_val < -1e-100, h_val, -1e-100))

    def dBh_dxi(self, x_i, x_j):
        # B(h) = -rho^-1 log(-h)
        # dB(h)/dx = -rho^-1 h^-1 dhdx
        val = -1 / (self.rho * self.game.h(x_i, x_j)) * self.game.dh_dxi(x_i, x_j)
        if self.config.DEBUG:
            val_num = jacobian_numerical(
                lambda xx: self.Bh(xx.reshape(x_i.shape), x_j), x_i.flatten())
            assert np.linalg.norm(val - val_num) < 1e-4
        return val

    def dBh_dxj(self, x_i, x_j):
        # B(h) = -rho^-1 log(-h)
        # dB(h)/dx = -rho^-1 h^-1 dhdx
        val = -1 / (self.rho * self.game.h(x_i, x_j)) * self.game.dh_dxj(x_i, x_j)
        if self.config.DEBUG:
            val_num = jacobian_numerical(
                lambda xx: self.Bh(x_i, xx.reshape(x_j.shape)), x_j.flatten())
            assert np.linalg.norm(val - val_num) < 1e-4
        return val

    def dBh_dxi_dxi(self, x_i, x_j):
        h = self.game.h(x_i, x_j)
        dhdxi = self.game.dh_dxi(x_i, x_j).reshape(1, self.n)
        val = 1 / (self.rho * h) * (-self.game.dh_dxi_dxi(x_i, x_j) +
                                    1 / h * dhdxi.T @ dhdxi)
        if self.config.DEBUG:
            val_num = jacobian_numerical(
                lambda xx: self.dBh_dxi(xx.reshape(x_i.shape), x_j),
                x_i.flatten(),
                dim=self.n)
            assert np.linalg.norm(val - val_num) < 1e-4
        return val

    def dBh_dxi_dxj(self, x_i, x_j):
        h = self.game.h(x_i, x_j)
        dhdxi = self.game.dh_dxi(x_i, x_j).reshape(1, self.n)
        dhdxj = self.game.dh_dxj(x_i, x_j).reshape(1, self.n)
        val = 1 / (self.rho * h) * (-self.game.dh_dxi_dxj(x_i, x_j) +
                                    1 / h * dhdxi.T @ dhdxj)
        if self.config.DEBUG:
            val_num = jacobian_numerical(
                lambda xx: self.dBh_dxi(x_i, xx.reshape(x_j.shape)),
                x_j.flatten(),
                dim=self.n)
            assert np.linalg.norm(val - val_num) < 1e-4
        return val

    def dBh_dxj_dxj(self, x_i, x_j):
        h = self.game.h(x_i, x_j)
        dhdxj = self.game.dh_dxj(x_i, x_j).reshape(1, self.n)
        val = 1 / (self.rho * h) * (-self.game.dh_dxj_dxj(x_i, x_j) +
                                    1 / h * dhdxj.T @ dhdxj)
        if self.config.DEBUG:
            val_num = jacobian_numerical(
                lambda xx: self.dBh_dxj(x_i, xx.reshape(x_j.shape)),
                x_j.flatten(),
                dim=self.n)
            assert np.linalg.norm(val - val_num) < 1e-4
        return val

    def dLLi_dxi_dx(self, x, u, h_plus_mask, lamda, mu, i):
        if self.config.USE_CPP:
            return self.cpp.dLLi_dxi_dx([xx for xx in x], [uu for uu in u],
                                        [hh for hh in h_plus_mask],
                                        [ll for ll in lamda],
                                        [mm for mm in mu], i)
        T = self.T
        N = self.N
        n = self.n
        dim_x = T * N * n
        dLL_dxi_dx = np.zeros((T * n, dim_x))

        def submtx(k, j):
            return dLL_dxi_dx[
                (k - 1) * n:k * n, (k - 1) * N * n + j * n:(k - 1) * N * n +
                (j + 1) * n]
        for k in range(1, T):
            # dLLi_dxki_dxki
            mtx = submtx(k, i)
            val1 = self.game.dJi_dxi_dxi(x[k - 1], u[k, i], i)
            val2 = np.sum([
                mu[k - 1, i, j.item()] *
                (self.game.dh_dxi_dxi(x[k - 1, i], x[k - 1, j.item()]))
                for j in np.nonzero(h_plus_mask[k - 1, i])[0]
            ],
                axis=0)
            val3 = np.sum([
                self.dBh_dxi_dxi(x[k - 1, i], x[k - 1, j.item()]) *
                (j.item() != i) for j in np.nonzero(~h_plus_mask[k - 1, i])[0]
            ],
                axis=0)
            mtx[:, :] = val1 + val2 + val3

        # dLLi_dxki_dxki, k=T, u_T is undefined, use 0 to penalize J(x) only
        mtx = submtx(T, i)
        h_neg_terms = np.sum(
            [
                self.dBh_dxi_dxi(x[T-1, i], x[T-1, j.item()]) * (j.item() != i)
                for j in np.nonzero(~h_plus_mask[T-1, i])[0]
            ],
            axis=0
        )
        h_pos_terms = np.sum(
            [
                mu[T-1, i, j.item()] * (self.game.dh_dxi_dxi(x[T-1, i], x[T-1, j.item()]))
                for j in np.nonzero(h_plus_mask[T-1, i])[0]
            ],
            axis=0
        )

        mtx[:, :] = self.game.dJfi_dxi_dxi(x[T-1], i) + h_neg_terms + h_pos_terms

        # dLLi_dxi_dxj
        for k in range(1, T):
            for j in range(N):
                if i == j:
                    continue
                if j in np.nonzero(h_plus_mask[k - 1, i])[0]:
                    # dLLi_dxki_dxkj
                    val = self.game.dJi_dxi_dxj(
                        x[k - 1], u[k, i], i,
                        j) + mu[k - 1, i, j] * self.game.dh_dxi_dxj(
                            x[k - 1, i], x[k - 1, j])
                    mtx = submtx(k, j)
                    mtx[:, :] = val
                else:
                    # dLLi_dxki_dxkj
                    val = self.game.dJi_dxi_dxj(x[k - 1], u[k, i], i,
                                                j) + self.dBh_dxi_dxj(
                        x[k - 1, i], x[k - 1, j])
                    mtx = submtx(k, j)
                    mtx[:, :] = val
        k = T
        for j in range(N):
            if i == j:
                continue
            if j in np.nonzero(h_plus_mask[k - 1, i])[0]:
                # dLLi_dxki_dxkj
                val = self.game.dJfi_dxi_dxj(x[k - 1], i,
                                             j) + mu[k - 1, i, j] * self.game.dh_dxi_dxj(
                    x[k - 1, i], x[k - 1, j])
                mtx = submtx(k, j)
                mtx[:, :] = val
            else:
                # dLLi_dxki_dxkj
                val = self.game.dJfi_dxi_dxj(x[k - 1], i, j) + self.dBh_dxi_dxj(
                    x[k - 1, i], x[k - 1, j])
                mtx = submtx(k, j)
                mtx[:, :] = val

        if self.config.DEBUG:
            dLLi_dxi_dx_num = jacobian_numerical(lambda xx: self.dLLi_dxi(
                xx.reshape(x.shape), u, h_plus_mask, lamda, mu, i),
                x.flatten(),
                dim=T * n)
            logger.debug(
                f'dLL_dxdx err {np.linalg.norm(dLLi_dxi_dx_num - dLL_dxi_dx)}')
            assert np.linalg.norm(dLLi_dxi_dx_num - dLL_dxi_dx) < 1e-4
        if self.config.CPP_DEBUG:
            alt = self.cpp.dLLi_dxi_dx([xx for xx in x], [uu for uu in u],
                                       [hh for hh in h_plus_mask],
                                       [ll for ll in lamda], [mm
                                                              for mm in mu], i)
            if np.linalg.norm(alt - dLL_dxi_dx) > 1e-4:
                breakpoint()
        return dLL_dxi_dx

    # for kernel gradient in stein game
    def dx_du(self, x, u):
        T = self.T
        N = self.N
        n = self.n
        m = self.m
        dim_x = T * N * n
        dim_u = T * N * m
        dxdu = np.zeros((dim_x, dim_u))
        for i in range(N):
            k = 0
            dxdu[k * N * n + i * n:k * N * n + (i + 1) * n,
                 k * N * m + i * m:k * N * m + (i + 1) * m] = self.game.df_du(
                     self.x0[i], u[0, i], i)
            for k in range(1, T):
                dxdu[k * N * n + i * n:k * N * n + (i + 1) * n,
                     k * N * m + i * m:k * N * m + (i + 1) * m] = self.game.df_du(
                         x[k - 1, i], u[k, i], i)

        if self.config.DEBUG:
            dxdu_num = jacobian_numerical(
                fun=lambda uu: self.game.rollout(self.x0, uu.reshape((T, N, m))),
                x=u.flatten(),
                dim=dim_x)
            logger.debug(f'dxdu_num err {np.linalg.norm(dxdu_num - dxdu)}')
            if np.linalg.norm(dxdu_num - dxdu) > 1e-4:
                breakpoint()
        return dxdu

    def dF_dx(self, x, u, i, k):
        ''' F(x,u) = f(x_k_i,u_k_i)-x_k+1_i, find dF_dx, note x here is of dim(T*N*n) '''
        if self.config.USE_CPP:
            return self.cpp.dF_dx([xx for xx in x], [uu for uu in u], i, k)
        T = self.T
        N = self.N
        n = self.n
        dim_x = T * N * n
        dFdx = np.zeros((n, dim_x))
        dFdx[:, (k - 1) * N * n + i * n:(k - 1) * N * n +
             (i + 1) * n] = self.game.df_dx(x[k - 1, i], u[k, i], i)
        dFdx[:, k * N * n + i * n:k * N * n + (i + 1) * n] = -np.eye(n)
        if self.config.CPP_DEBUG:
            alt = self.cpp.dF_dx([xx for xx in x], [uu for uu in u], i, k)
            if np.linalg.norm(alt - dFdx) > 1e-4:
                breakpoint()
        return dFdx

    def dF0_dx(self, x, u, i):
        ''' F0(x,u) = f(x_0_i,u_0_i)-x_1_i, find dF_dx note x here is of dim(T*N*n)
            A specialization for dF_dx when k=0, since we need x0
        '''
        if self.config.USE_CPP:
            return self.cpp.dF0_dx([xx for xx in x], [uu for uu in u], i)
        T = self.T
        N = self.N
        n = self.n
        dim_x = T * N * n
        dFdx = np.zeros((n, dim_x))
        dFdx[:, i * n:(i + 1) * n] = -np.eye(n)
        if self.config.CPP_DEBUG:
            alt = self.cpp.dF0_dx([xx for xx in x], [uu for uu in u], i)
            if np.linalg.norm(alt - dFdx) > 1e-4:
                breakpoint()
        return dFdx

    def dh_dx(self, x, k, i, j):
        ''' find d h(x_i,x_j)/ d x note x here is of dim(T*N*n) '''
        if self.config.USE_CPP:
            return self.cpp.dh_dx([xx for xx in x], k, i, j)
        T = self.T
        N = self.N
        n = self.n
        dim_x = T * N * n
        dhdx = np.zeros((1, dim_x))
        dhdx[:, (k - 1) * N * n + i * n:(k - 1) * N * n +
             (i + 1) * n] = self.game.dh_dxi(x[k - 1, i], x[k - 1, j])
        dhdx[:, (k - 1) * N * n + j * n:(k - 1) * N * n +
             (j + 1) * n] = self.game.dh_dxj(x[k - 1, i], x[k - 1, j])
        if self.config.CPP_DEBUG:
            alt = self.cpp.dh_dx([xx for xx in x], k, i, j)
            if np.linalg.norm(alt - dhdx) > 1e-4:
                breakpoint()
        return dhdx

    def dr_dx(self, x, u, lamda, mu, h_plus_mask):
        ''' return: dim(r)*dim(x) '''
        if self.config.USE_CPP:
            return self.cpp.dr_dx([xx for xx in x], [uu for uu in u],
                                  [ll for ll in lamda], [mmm for mmm in mu],
                                  [hh for hh in h_plus_mask])
        T = self.T
        N = self.N
        n = self.n
        m = self.m
        dim_x = T * N * n
        dim_r = N * (T * n + T * m + T * n) + np.sum(h_plus_mask)
        drdx = np.zeros((dim_r, dim_x))
        index = 0
        for i in range(self.N):
            dLL_dxi_dx = self.dLLi_dxi_dx(x, u, h_plus_mask, lamda, mu, i)
            # this item is identically zero
            # dLL_dudx = np.zeros((dim_u,dim_x))
            dF0dx = self.dF0_dx(x, u, i)
            # dynamics for f(x0,u0) = x1
            drdx[index:index + T * n, :] = dLL_dxi_dx
            index += T * n + T * m
            drdx[index:index + n, :] = self.config.dynamics_residual_weight * dF0dx
            for k in range(1, self.T):
                dFdx = self.dF_dx(x, u, i, k)
                drdx[index + k * n:index +
                     (k + 1) * n, :] = self.config.dynamics_residual_weight * dFdx
            index += n * T
            for k in range(1, self.T + 1):
                indices = np.nonzero(h_plus_mask[k - 1, i])[0]
                if len(indices) == 0:
                    continue
                dhdx = np.vstack(
                    [self.dh_dx(x, k, i, j.item()) for j in indices])
                drdx[index:index + dhdx.shape[0], :] = dhdx
                index += len(indices)

        if self.config.DEBUG:
            drdx_num = jacobian_numerical(lambda xx: self.r(
                xx.reshape(x.shape), u, lamda, mu, h_plus_mask),
                x.flatten(),
                dim=dim_r)
            logger.debug(f'drdx err {np.linalg.norm(drdx-drdx_num)}')
            assert np.linalg.norm(drdx - drdx_num) < 1e-4
        if self.config.CPP_DEBUG:
            alt = self.cpp.dr_dx([xx for xx in x], [uu for uu in u],
                                 [ll for ll in lamda], [mmm for mmm in mu],
                                 [hh for hh in h_plus_mask])
            if np.linalg.norm(alt - drdx) > 1e-4:
                breakpoint()
        return drdx

    def dr_du(self, x, u, lamda, mu, h_plus_mask):
        ''' return: dim(r)*dim(u) '''
        if self.config.USE_CPP:
            return self.cpp.dr_du([xx for xx in x], [uu for uu in u],
                                  [ll for ll in lamda], [mmm for mmm in mu],
                                  [hh for hh in h_plus_mask])
        T = self.T
        N = self.N
        n = self.n
        m = self.m
        dim_u = T * N * m
        dim_r = N * (T * n + T * m + T * n) + np.sum(h_plus_mask)

        drdu = np.zeros((dim_r, dim_u))
        index = 0
        for i in range(self.N):
            index += T * n
            for k in range(self.T):
                dLL_duik_duik = self.game.dJi_dudu(x[k - 1], u[k, i], i)
                drdu[index + k * m:index + (k + 1) * m,
                     k * N * m + i * m:k * N * m + (i + 1) * m] = dLL_duik_duik
            index += T * m
            k = 0
            drdu[index + k * n:index + (k + 1) * n,
                 k * N * m + i * m:k * N * m +
                 (i + 1) * m] = self.config.dynamics_residual_weight * self.game.df_du(
                     self.x0[i], u[k, i], i)
            for k in range(1, self.T):
                drdu[index + k * n:index + (k + 1) * n,
                     k * N * m + i * m:k * N * m +
                     (i + 1) * m] = self.config.dynamics_residual_weight * self.game.df_du(
                         x[k - 1, i], u[k, i], i)
            index += n * T + np.sum(h_plus_mask[:,
                                                i])  # skip  f(x,u)-x+,  h(x,x)

        if self.config.DEBUG:
            drdu_num = jacobian_numerical(lambda uu: self.r(
                x, uu.reshape(u.shape), lamda, mu, h_plus_mask),
                u.flatten(),
                dim=dim_r)
            if np.linalg.norm(drdu - drdu_num) > 1e-4:
                for i in range(N):
                    for k in range(0, T):
                        val = drdu[:, k * N * m + i * m:k * N * m + i * m + m]
                        val_num = drdu_num[:, k * N * m + i * m:k * N * m +
                                           i * m + m]
                        if np.linalg.norm(val - val_num) > 1e-4:
                            logger.debug(
                                f'k={k}, i={i},{np.nonzero(val-val_num)}')
                breakpoint()
            assert np.linalg.norm(drdu - drdu_num) < 1e-4
        if self.config.CPP_DEBUG:
            alt = self.cpp.dr_du([xx for xx in x], [uu for uu in u],
                                 [ll for ll in lamda], [mmm for mmm in mu],
                                 [hh for hh in h_plus_mask])
            if np.linalg.norm(alt - drdu) > 1e-4:
                breakpoint()
        return drdu

    def dr_dlamda(self, x, u, lamda, mu, h_plus_mask):
        ''' return: dim(r)*dim(lamda) '''
        if self.config.USE_CPP:
            return self.cpp.dr_dlamda([xx for xx in x], [uu for uu in u],
                                      [ll
                                       for ll in lamda], [mmm for mmm in mu],
                                      [hh for hh in h_plus_mask])
        T = self.T
        N = self.N
        n = self.n
        m = self.m
        dim_lamda = T * N * n
        dim_r = N * (T * n + T * m + T * n) + np.sum(h_plus_mask)
        dr_dlamda = np.zeros((dim_r, dim_lamda))
        index = 0
        for i in range(N):
            for k in range(1, T):
                # dLLi_dxki_dlamda_ki
                dr_dlamda[index + (k - 1) * n:index + k * n,
                          k * N * n + i * n:k * N * n +
                          (i + 1) * n] = self.game.df_dx(x[k - 1, i], u[k, i], i).T
                # dLLi_dxki_dlamda_k-1,i
                dr_dlamda[index + (k - 1) * n:index + k * n, (k - 1) * N * n +
                          i * n:(k - 1) * N * n + (i + 1) * n] = -np.eye(n)
            k = T
            dr_dlamda[index + (k - 1) * n:index + k * n, (k - 1) * N * n +
                      i * n:(k - 1) * N * n + (i + 1) * n] = -np.eye(n)
            index += T * n  # skip dLL_dxi, index now points at dLLi_dui
            k = 0
            dr_dlamda[index + k * m:index + (k + 1) * m,
                      k * N * n + i * n:k * N * n + (i + 1) * n] = self.game.df_du(
                          self.x0[i], u[k, i], i).T
            for k in range(1, T):
                dr_dlamda[index + k * m:index + (k + 1) * m,
                          k * N * n + i * n:k * N * n +
                          (i + 1) * n] = self.game.df_du(x[k - 1, i], u[k, i], i).T

            index += T * m + n * T + np.sum(
                h_plus_mask[:, i])  # skip  dLL_dui, f(x,u)-x+,  h(x,x)

        if self.config.DEBUG:
            dr_dlamda_num = jacobian_numerical(lambda ll: self.r(
                x, u, ll.reshape(lamda.shape), mu, h_plus_mask),
                lamda.flatten(),
                dim=dim_r)
            logger.debug(
                f'dr_dlamda err {np.linalg.norm(dr_dlamda-dr_dlamda_num)}')
            assert np.linalg.norm(dr_dlamda - dr_dlamda_num) < 1e-4
        if self.config.CPP_DEBUG:
            alt = self.cpp.dr_dlamda([xx for xx in x], [uu for uu in u],
                                     [ll for ll in lamda], [mmm for mmm in mu],
                                     [hh for hh in h_plus_mask])
            if np.linalg.norm(alt - dr_dlamda) > 1e-4:
                breakpoint()
        return dr_dlamda

    def dr_dmu(self, x, u, lamda, mu, h_plus_mask):
        ''' return: dim(r)*dim(mu) '''
        if self.config.USE_CPP:
            return self.cpp.dr_dmu([xx for xx in x], [uu for uu in u],
                                   [ll for ll in lamda], [mmm for mmm in mu],
                                   [hh for hh in h_plus_mask])
        T = self.T
        N = self.N
        n = self.n
        m = self.m
        dim_r = N * (T * n + T * m + T * n) + np.sum(h_plus_mask)
        dim_mu = T * N * N

        dr_dmu = np.zeros((dim_r, dim_mu))
        index = 0
        for i in range(self.N):
            # dmu i,j,k
            dLL_dxi_dmu = self.dLLi_dxi_dmu(x, u, h_plus_mask, lamda, mu, i)
            dr_dmu[index:index + T * n, :] = dLL_dxi_dmu
            if self.config.DEBUG:
                dLL_dxi_dmu_num = jacobian_numerical(lambda mm: self.dLLi_dxi(
                    # pylint: disable-next=cell-var-from-loop
                    x, u, h_plus_mask, lamda, mm.reshape(mu.shape), i),
                    mu.flatten(),
                    dim=self.T * self.n)
                assert np.linalg.norm(dLL_dxi_dmu - dLL_dxi_dmu_num) < 1e-4

            index += T * n + T * m + n * T + np.sum(h_plus_mask[:, i])

        if self.config.DEBUG:
            dr_dmu_num = jacobian_numerical(lambda mm: self.r(
                x, u, lamda, mm.reshape(mu.shape), h_plus_mask),
                mu.flatten(),
                dim=dim_r)
            logger.debug(f'drdx err {np.linalg.norm(dr_dmu-dr_dmu_num)}')
            assert np.linalg.norm(dr_dmu - dr_dmu_num) < 1e-4
        if self.config.CPP_DEBUG:
            alt = self.cpp.dr_dmu([xx for xx in x], [uu for uu in u],
                                  [ll for ll in lamda], [mmm for mmm in mu],
                                  [hh for hh in h_plus_mask])
            if np.linalg.norm(alt - dr_dmu) > 1e-4:
                breakpoint()
        return dr_dmu
