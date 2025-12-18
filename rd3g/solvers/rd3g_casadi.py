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
        self.construct_gradient_fun()

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

    def construct_gradient_fun(self):
        """ Construct functions based on CasADi autodiff"""

    def init(self):
        """Setup solver parameters that changes between iterations, call this
        funtion to reset the solver."""
        raise NotImplementedError

    """
    def solve(self):

        N = self.N
        T = self.T
        n = self.n
        m = self.m
        # y: x(T*N*n) ,u(T*N*m), lambda(T,N,n),mu(T,N,N)
        logger.debug(
            f'primal variables:{(T*N*n) +(T*N*m)} dual variables:{(N*T*n)+(T*N*N)}'
        )
        t0 = time()


        t_solve = time() - t0
        logger.info(f'Total solve time: {t_solve}s')
        if i == self.config.iterations - 1:
            logger.warning(' algorithm did not reach stopping criterion ')
        full_x_ref = np.vstack([self.x0[np.newaxis, :, :], x_ref])

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
    """

    def final(self):
        self.profiler.summary()
        plt.plot(self.residual_vec, '*-')
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Residual (exp)')
        plt.show()

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
        h_neg_barrier_vals = -1.0/self.rho * cas.sum(cas.log(-cas.fmin(h_vals, -1e-100)))

        i_onehot = cas.SX.eye(self.N)[:, i]
        # NOTE it may be better to store lamda_k_T to take advantage of col-major storage
        dynamics_val = lamda_k[i, :] @ (self.game.f(x_k[i, :].T, u_k_i, i_onehot) - x_k1_i)
        val = self.game.J(x_k, u_k_i, i_onehot) + mu_h_plus_vals + h_neg_barrier_vals + dynamics_val
        assert val.shape == (1, 1)
        return val

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
        h_neg = -1.0/self.rho * cas.sum(cas.log(-cas.fmin(h_vals, -1e-100)))

        LLi_val += mu_h_plus + h_neg
        assert LLi_val.shape == (1, 1)
        return LLi_val

    def r(self, x, u, lamda, mu):
        ''' Residual for the game
        Args:
            x: (N*n,T) Agent states
            u: (N*m,T) Agent control
            lamda: (N*n, T) Multiplier for dynamics constraint
            mu: (N*N, T) Multiplier for positive h 
        Return:
            val: (N*(T*(n+m) + T*n + T*N) column vector of residual r
        '''
        T = self.T
        N = self.N
        n = self.n
        m = self.m
        # elements are column vectors
        r_vec = []
        for i in range(N):
            i_onehot = cas.SX.eye(self.N)[:, i]
            xi_vec = []
            ui_vec = []
            for k in range(T):
                xik = cas.reshape(x[:, k], N, n)[i, :]
                xi_vec.append(xik.T)
                uik = cas.reshape(u[:, k], N, m)[i, :]
                ui_vec.append(uik.T)
            xi = cas.vertcat(*xi_vec)
            ui = cas.vertcat(*ui_vec)
            dLLi_dxi = cas.jacobian(self.LLi(x, u, lamda, mu, i), xi).T
            dLLi_dui = cas.jacobian(self.LLi(x, u, lamda, mu, i), ui).T
            r_vec.append(dLLi_dxi)
            r_vec.append(dLLi_dui)
            assert dLLi_dxi.size2() == 1
            assert dLLi_dui.size2() == 1

            # dynamics residual for f(x0,u0) = x1
            x0 = self.game.config.get_param('x0')
            u0 = cas.reshape(u[:, 0], N, m)
            f0 = self.game.f(x0[i, :].T, u0[i, :].T, i_onehot) - cas.reshape(x[:, 0], N, n)[i, :].T
            assert f0.size2() == 1
            r_vec.append(f0)

            # dynamics residual for f(xk,uk) = x_{k+1}
            for k in range(1, self.T):
                xk = cas.reshape(x[:, k-1], N, n)
                xk1 = cas.reshape(x[:, k], N, n)
                uk = cas.reshape(u[:, k], N, m)
                fk = self.game.f(xk[i, :].T, uk[i, :].T, i_onehot) - xk1[i, :].T
                assert fk.size2() == 1
                r_vec.append(fk)

            # collision residual for h > 0
            for k in range(1, self.T+1):
                xk = cas.reshape(x[:, k-1], N, n)  # x[k] -> x_{k+1} due to index alignment
                h_vals = [self.game.h(xk[i, :].T, xk[j, :].T) for j in range(self.N)]
                h_vals_pos = cas.fmax(cas.vertcat(*h_vals), 0)
                assert h_vals_pos.size2() == 1
                r_vec.append(h_vals_pos)

        return cas.vertcat(*r_vec)
