""" Residual Descent Differential Dynamic Game Solver (RD3G) with CasADi """
# pylint: disable=invalid-name, forgotten-debug-statement
# NOTE since casadi use column-major memory layout, consider changing order of indexing
# so most frequent slicing is on columns

import logging
from time import time
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import casadi as cas

from rd3g.utilities.time_util import TimeUtil
from rd3g.core.base_solver import BaseSolver, BaseSolverConfig, Solution
from rd3g.core.base_game import BaseGame

logger = logging.getLogger('RD3G_CasADi')
logger.setLevel(logging.INFO)


# NOTE: changing config requires re-run codegen, since configs are constants
@dataclass(frozen=True)
class RD3GCasadiConfig(BaseSolverConfig):
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


class RD3GCasadi(BaseSolver):
    """Residual Descent Differential Dynamic Game Solver (RD3G) with CasADi

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

    def __init__(self, config: RD3GCasadiConfig, game: BaseGame):
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
        self.profiler = TimeUtil(False)
        # logger.debug_enable()
        self.residual_vec = []
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

    # ----- derivatives and other generic math functions ----
    # NOTE revised for casadi

    def L(self, x_k, u_k_i, x_k1_i, lamda_k, mu_k, i):
        ''' Lagrangian for agent i
        Args:
            x_k: (N,n) state vector at step k
            u_k_i: (m,1) control vector for agent i at step k
            x_k1_i: (n,1) state vector for agent i at step k+1
            lamda_k: (N,n) Multiplier for dynamics constraint
            mu_k: (N,N), symmetric,  multiplier for positive h 
            i: agent index i
        Return:
            val: scalar value of lagrangian
        '''
        N = self.N
        n = self.n
        m = self.m
        assert x_k.shape == (N, n)
        assert u_k_i.shape == (m, 1)
        assert x_k1_i.shape == (n, 1)
        assert lamda_k.shape == (N, n)
        assert mu_k.shape == (N, N)
        # feasibility for h>0
        h_vals = cas.vertcat(*[self.game.h(x_k[i, :].T, x_k[j, :].T) for j in range(self.N)])
        # NOTE add fmax here for safety, it may not be needed if
        # mu_k has structural 00 where h_vals < 0
        # we only want to sum h_val > 0
        mu_h_plus_vals = cas.dot(mu_k[i, :].T, cas.fmax(h_vals, 0))

        # barrier for h < 0
        h_neg_barrier_vals = -1.0/self.rho * cas.log(-cas.fmin(h_vals, -1e-100))

        i_onehot = cas.SX.eye(self.N)[:, i]
        # NOTE it may be better to store lamda_k_T to take advantage of col-major storage
        dynamics_val = lamda_k[i, :] @ (self.game.f(x_k[i, :].T, u_k_i, i_onehot) - x_k1_i)
        return self.game.J(x_k, u_k_i, i_onehot) + mu_h_plus_vals + h_neg_barrier_vals + dynamics_val

    # NOTE wip revised for casadi

    def LLi(self, x, u, lamda, mu, i):
        ''' Lagrangian for agent i across all time steps
        Args:
            x: (N*n,T) Agent states
            u: (N*m,T) Agent control
            lamda: (N*n, T) Multiplier for dynamics constraint
            mu: (N*N, T) Multiplier for positive h 
            i: agent index
        Return:
            val: scalar value of Lagrangian
        '''
        T = self.T
        N = self.N
        m = self.m
        n = self.n
        LLi_val = sum([self.L(cas.reshape(x[:, k - 1], N, n),
                              cas.reshape(u[:, k], N, m)[i, :].T,
                              cas.reshape(x[:, k], N, n)[i, :].T,
                              cas.reshape(lamda[:, k], N, n),
                              cas.reshape(mu[:, k - 1], N, N),
                              i)
                       for k in range(1, T)])
        x0 = self.game.config.get_param('x0')
        # x0 related terms
        i_onehot = cas.SX.eye(self.N)[:, i]
        u0_i = cas.reshape(u[:, 0], N, m)[i, :].T
        LLi_val += self.game.J(x0, u0_i, i_onehot) + \
            cas.reshape(lamda[:, 0], N, n)[i, :] @ (self.game.f(x0[i, :].T, u0_i, i_onehot) -
                                                    cas.reshape(x[:, 0], N, n)[i, :].T)
        # x_T related terms
        x_T = cas.reshape(x[:, T-1], N, n)
        LLi_val += self.game.Jfi(x_T, i_onehot)
        h_vals = cas.vertcat(*[self.game.h(x_T[i, :].T, x_T[j, :].T) for j in range(self.N)])
        mu_h_plus = cas.dot(cas.reshape(mu[:, T-1], N, N)[i, :].T, cas.fmax(h_vals, 0))
        h_neg = -1.0/self.rho * cas.log(-cas.fmin(h_vals, -1e-100))

        LLi_val += mu_h_plus + h_neg
        return LLi_val

    def r(self, x, u, lamda, mu, h_plus_mask):
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
