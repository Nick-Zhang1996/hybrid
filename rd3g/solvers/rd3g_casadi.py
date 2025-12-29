""" Residual Descent Differential Dynamic Game Solver (RD3G) with CasADi """
# pylint: disable=invalid-name, forgotten-debug-statement
# NOTE since casadi use column-major memory layout, consider changing order of indexing
# so most frequent slicing is on columns

import logging
from dataclasses import dataclass
from itertools import accumulate
from time import time

import numpy as np
import scipy.sparse  # sparse matrix operations
import scipy.sparse.linalg
import scipy.linalg
import matplotlib.pyplot as plt
import casadi as cas

from rd3g.utilities.util import dm_to_csc
from rd3g.utilities.time_util import TimeUtil
from rd3g.core.base_solver import BaseSolver, BaseSolverConfig, Solution
from rd3g.core.base_casadi_game import CasadiGameConfig

logger = logging.getLogger('RD3G_CasADi')
logger.setLevel(logging.DEBUG)


def check_inertia(D):
    """ Check the inertia of block-diagonal matrix D.
    Args:
        D: block-diagonal matrix consisting of 1*1 and 2*2 blocks
    Returns:
        positive_eigenvalue_count
        negative_eigenvalue_count
        zero_eigenvalue_count
    """
    n = D.shape[0]
    tol = 1e-5
    i = 0
    pos = 0
    neg = 0
    zero = 0

    def count(pos, neg, zero, val):
        if val > tol:
            pos += 1
        elif val < -tol:
            neg += 1
        else:
            zero += 1
        return pos, neg, zero

    while i < n:
        if i < n-1 and abs(D[i, i+1]) > tol:
            # 2 by 2 block
            eigvals = np.linalg.eigvalsh(D[i:i+2, i:i+2])
            for val in eigvals:
                pos, neg, zero = count(pos, neg, zero, val)
            i += 2
        else:
            # 1 by 1 block
            pos, neg, zero = count(pos, neg, zero, D[i, i])
            i += 1
    assert pos + neg + zero == n
    return pos, neg, zero


# NOTE: changing config requires re-run codegen, since configs are constants
@dataclass(frozen=True)
class RD3GCasadiConfig(BaseSolverConfig):
    """Configs for Residual Game."""
    tolerance: float = 5e-4
    iterations: int = 30
    # backtracking line search param
    bc_a: float = 1e-4  # alpha
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

    def __init__(self, config: RD3GCasadiConfig, game):
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

        # CasADi objects
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
        N = self.N
        n = self.n
        m = self.m
        T = self.T
        gc = self.game.config

        x = cas.SX.sym('x', N*n, T)
        u = cas.SX.sym('u', N*m, T)
        lamda = cas.SX.sym('lamda', N*n, T)
        mu = cas.SX.sym('mu', N*N, T)
        x0 = cas.SX.sym('x0', N, n)
        config_params = [gc.get_int_param_sx(), gc.get_double_param_sx()]

        args = [x, u, lamda, mu]
        y = cas.vertcat(*[cas.vec(val) for val in args])

        # get_n_fun = cas.Function('get_n', [], [self.n])
        # get_m_fun = cas.Function('get_m', [], [self.m])

        r = self.r(*args)
        self.r_fun_casadi = cas.Function('r', args+config_params, [r])

        dr_dy = cas.jacobian(r, y)
        self.dr_dy_fun_casadi = cas.Function('dr_dy', args+config_params, [dr_dy])

        X = self.game.rollout(x0, u)
        self.rollout_fun_casadi = cas.Function('rollout', [x0, u]+config_params, [X])

        H_val = self.get_h_val(x)
        self.H_val_casadi = cas.Function('H', [x]+config_params, [H_val])

        xki = cas.SX.sym('xki', n, 1)
        xkj = cas.SX.sym('xkj', n, 1)
        h_val = self.game.h(xki, xkj)
        self.h_val_casadi = cas.Function('h', [xki, xkj]+config_params, [h_val])

    def init(self):
        """Setup solver parameters that changes between iterations, call this
        funtion to reset the solver."""
        raise NotImplementedError

    def solve(self):
        N = self.N
        T = self.T
        n = self.n
        m = self.m
        # y: x(T*N*n) ,u(T*N*m), lambda(T,N,n),mu(T,N,N)
        logger.debug(
            f'primal variables:{(T*N*n) +(T*N*m)} dual variables:{(N*T*n)+(T*N*N)}'
        )

        u_ref = np.zeros((N*m, T), order='F')
        # x_ref = x_1 .. x_T, NOTE the array index is offset from the math notation
        # return: (N*n, T)
        gc = self.game.config
        int_param_dm = cas.DM(gc.get_int_param_np())
        double_param_dm = cas.DM(gc.get_double_param_np())
        x_ref = self.rollout_fun_casadi(self.x0, u_ref, int_param_dm, double_param_dm)
        # NOTE to convert to np array
        # np.array(x_ref, order='F'),reshape(N,n,T, order='F') -> (N, n, T)
        lambda_ref = cas.DM.zeros((N*n, T))
        # defined for all h_k_i_j, but all values may not be used
        mu_ref = cas.DM.zeros((N*N, T))

        for _ in range(self.config.iterations):
            x_ref, u_ref, lambda_ref, mu_ref = self.step(x_ref, u_ref, lambda_ref, mu_ref)

    def step(self, x_ref, u_ref, lambda_ref, mu_ref):
        """ Solver step function
        Args:
            x_ref: N*n,T, casadi.DM
            u_ref: N*m,T
            lamda: N*n,T
            mu: N*N,T
        Return:
            x_ref, u_ref, lamda, mu, updated
        """
        N = self.N
        T = self.T
        n = self.n
        m = self.m
        gc = self.game.config
        x = x_ref
        u = u_ref
        lamda = lambda_ref
        mu = mu_ref
        int_param_dm = cas.DM(gc.get_int_param_np())
        double_param_dm = cas.DM(gc.get_double_param_np())
        params_dm = [int_param_dm, double_param_dm]

        r0_val = self.r_fun_casadi(x, u, lamda, mu, *params_dm)
        dr_dy_val = self.dr_dy_fun_casadi(x, u, lamda, mu, *params_dm)
        r0_np = np.array(r0_val)
        r0_norm = np.linalg.norm(r0_np)
        dr_dy_csc = dm_to_csc(dr_dy_val)

        # Naive method: solve sparse system directly
        # r0 + dr_dy @ dy = 0
        # solve for dy directly
        t0 = time()
        dy_np, istop, itn, normr = scipy.sparse.linalg.lsqr(dr_dy_csc, -r0_np)[:4]
        dt = time() - t0
        residual = np.linalg.norm(normr)
        istop_lut = ['', 'Direct Sol', "Least Square Sol"]
        logger.info(f'Full stop:{istop_lut[istop]}, {dt=}s {itn=}, {residual=}')

        # study the LDL decomposition
        lu, d, perm = scipy.linalg.ldl(dr_dy_csc.toarray())
        pos, neg, zero = check_inertia(d)
        logger.info(f'Full-system Inertia: {pos, neg, zero}')

        # Check, does dy improve residual? do a line search --- Yes!
        dy = cas.DM(dy_np)
        # size of x, u, lamda, mu
        sizes = [0, N*n*T, N*m*T, N*n*T, N*N*T]
        offsets = list(accumulate(sizes))
        dx, du, dlamda, dmu = cas.vertsplit(dy, offsets)
        step_size = 1.0
        step_size_vec = []
        stepped_r_vec = []
        for i in range(self.config.backtracking_max_iter):
            r_val = self.r_fun_casadi(x+step_size*cas.reshape(dx, N*n, T),
                                      u+step_size*cas.reshape(du, N*m, T),
                                      lamda+step_size*cas.reshape(dlamda, N*n, T),
                                      mu+step_size*cas.reshape(dmu, N*N, T),
                                      int_param_dm, double_param_dm
                                      )
            r_norm = np.linalg.norm(r_val)
            step_size_vec.append(step_size)
            stepped_r_vec.append(r_norm)
            if r_norm > (1 - self.config.bc_a * step_size) * r0_norm:
                step_size *= self.config.bc_b
            else:
                break
        # plt.plot(step_size_vec, stepped_r_vec, '*-')
        # plt.plot(0, r0_norm, 'o')
        # plt.title('Full descent')
        # plt.show()

        # Remove inactive constraints and their multiplier
        # h < 0 -> inactive cosntraint, remove rows for h, also remove columns for mu
        # r consists of all agent's residual concatenated, EACH agent has n_ri = 2nT + mT + NT rows
        # If h(i,j,k) < 0, then r[i*n_ri + 2nT + k*N + j] = 0 and can be removed
        # Also, the corresponding multiplier mu dim(NN,T) row [N*i+j, k] can be removed
        # NOTE that the same applies for flipped i,j
        # TODO handle symmetry and keep sparseness
        n_ri = 2*n*T + m*T + N*T
        mu_in_y_offset = N*n*T + N*m*T + N*n*T

        H_val = np.array(self.H_val_casadi(x, *params_dm), order='F').reshape((N, N, T), order='F')
        neg_h_mask = np.array(H_val < 0).nonzero()
        inactive_r_rows = []
        inactive_y_rows = []
        for i, j, k in zip(*neg_h_mask):
            inactive_r_rows.append(i*n_ri + 2*n*T + m*T + k*N + j)
            inactive_y_rows.append(mu_in_y_offset + k*N*N + N*i+j)
            """
            # Check h_val is indeed negative
            print(f'adding {i,j,k}, -> {inactive_r_rows[-1]}')
            xk = np.array(x_ref[:, k], order='F').reshape((N, n), order='F')
            h_val = self.h_val_casadi(cas.DM(xk[i, :]), cas.DM(xk[j, :]), *params_dm)
            if (h_val > 0):
                print(i, j, k)
                breakpoint()
            assert h_val < 0
            """

        # set all self-collision to be inactive
        for k in range(T):
            for i in range(N):
                j = i
                inactive_r_rows.append(i*n_ri + 2*n*T + m*T + k*N + j)
                inactive_y_rows.append(mu_in_y_offset + k*N*N + N*i+j)
                """
                # Verify that h(xi,xi) > 0, because an agent always collide with itself
                print(f'adding {i,j,k}, -> {inactive_r_rows[-1]}')
                xk = np.array(x_ref[:, k], order='F').reshape(N, n)
                h_val = self.h_val_casadi(cas.DM(xk[i, :]), cas.DM(xk[j, :]), *params_dm)
                assert h_val >= 0
                """
        # Remove zero rows & columns in the linear system
        all_r_indices = np.arange(dr_dy_csc.shape[0])
        all_y_indices = np.arange(dr_dy_csc.shape[1])
        # Get the "complement" (the indices you want to keep)
        active_r_rows = np.setdiff1d(all_r_indices, inactive_r_rows)
        active_y_rows = np.setdiff1d(all_y_indices, inactive_y_rows)

        # Check that the rows/cols removed are indeed useless
        """
        for row_idx in inactive_r_rows:
            if np.sum(np.abs(np.array(dr_dy_csc[row_idx, :]))) > 1e-5:
                print(row_idx)
                breakpoint()
        """
        # TODO set the relevant mu to 0
        reduced_r0 = r0_np[active_r_rows, :]
        reduced_dr_dy_csc = dr_dy_csc[active_r_rows, :][:, active_y_rows]

        # Reduced method: solve sparse system directly
        # r0 + dr_dy @ dy = 0
        # Solve for reduced_dy
        t0 = time()
        reduced_dy, istop, itn, normr = scipy.sparse.linalg.lsqr(reduced_dr_dy_csc, -reduced_r0)[:4]
        dt = time() - t0
        residual = np.linalg.norm(normr)
        logger.info(f'Reduced stop:{istop_lut[istop]},{dt=}s {itn=}, {residual=}')
        # Study the LDL decomposition
        lu, d, perm = scipy.linalg.ldl(reduced_dr_dy_csc.toarray())
        pos, neg, zero = check_inertia(d)
        logger.info(f'Reduced-system Inertia: {pos, neg, zero}')
        # Inertia checking for SOSC -> not enough pos, too many neg and zeros
        # Primal variables - inactive lagrange multipliers (mu)
        in_n = T*N*n + T*N*m
        in_m = T*N*(n+N) - len(inactive_r_rows)  # f, h - inactive collision constraints
        logger.info(f'Expected SOSC inertia {in_n,in_m,0}')

        # Do we have linearly dependent constraints?

        breakpoint()

        # Verify residual reduction with a line search
        # Recover full dy
        dy = np.zeros(dr_dy_csc.shape[1])
        dy[active_y_rows] = reduced_dy

        dx, du, dlamda, dmu = cas.vertsplit(cas.DM(dy), offsets)
        step_size = 1.0
        step_size_vec = []
        stepped_r_vec = []
        for i in range(self.config.backtracking_max_iter):
            r_val = self.r_fun_casadi(x+step_size*cas.reshape(dx, N*n, T),
                                      u+step_size*cas.reshape(du, N*m, T),
                                      lamda+step_size*cas.reshape(dlamda, N*n, T),
                                      mu+step_size*cas.reshape(dmu, N*N, T),
                                      int_param_dm, double_param_dm
                                      )
            r_norm = np.linalg.norm(r_val)
            step_size_vec.append(step_size)
            stepped_r_vec.append(r_norm)
            if r_norm > (1 - self.config.bc_a * step_size) * r0_norm:
                step_size *= self.config.bc_b
            else:
                break
        # plt.plot(step_size_vec, stepped_r_vec, '*-')
        # plt.plot(0, r0_norm, 'o')
        # plt.title('Reduced descent')
        # plt.show()
        # TODO try just keep the lowest r value, no need to reducing step

        new_x = x+step_size*cas.reshape(dx, N*n, T)
        new_u = u+step_size*cas.reshape(du, N*m, T)
        new_lamda = lamda+step_size*cas.reshape(dlamda, N*n, T)
        new_mu = mu+step_size*cas.reshape(dmu, N*N, T)
        logger.info(f'{r0_norm=}, {step_size=}, {r_norm=}')

        # Where does the norm come from? -> mostly collision (h)
        r_val_np = r_val.toarray()
        r_Lx, r_Lu, r_f, r_h = self.residual_components(r_val)
        logger.info(f'Residual breakdown {r_Lx=}, {r_Lu=}, {r_f=}, {r_h=}')

        # Did we remove active constraints? - > yes
        inactive_residual = np.linalg.norm(r_val_np[inactive_r_rows, 0])
        logger.info(f'{inactive_residual=}')

        # Example? Check ORIGINAL x (active constraint is evaluated on original x)
        # agent 0
        i = 0
        offset = n*T+m*T+n*T
        h_agent_0 = r_val_np[offset:offset+T*N, 0]
        new_x_np = np.array(x.toarray(), order='F').reshape((N, n, T), order='F')
        h_vals = []
        for j in range(self.N):
            h_val_ij = [
                self.h_val_casadi(cas.DM(new_x_np[i, :, k]), cas.DM(new_x_np[j, :, k]), *params_dm)
                for k in range(self.T)
            ]
            h_vals.append(h_val_ij)
        h_vals = np.array(h_vals)

        breakpoint()

        return new_x, new_u, new_lamda, new_mu

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
        LLi_val += (self.game.J(x0, u0_i, i_onehot) +
                    cas.reshape(lamda[:, 0], N, n)[i, :] @ (self.game.f(x0[i, :].T, u0_i, i_onehot) -
                                                            cas.reshape(x[:, 0], N, n)[i, :].T))
        # x_T related terms
        x_T = cas.reshape(x[:, T-1], N, n)
        LLi_val += self.game.Jfi(x_T, i_onehot)
        h_vals = cas.vertcat(*[self.game.h(x_T[i, :].T, x_T[j, :].T) for j in range(self.N)])
        mu_h_plus = cas.dot(cas.reshape(mu[:, T-1], N, N)[i, :].T, cas.fmax(h_vals, 0))
        h_neg = -1.0/self.rho * cas.sum(cas.log(-cas.fmin(h_vals, -1e-100)))

        LLi_val += mu_h_plus + h_neg
        assert LLi_val.shape == (1, 1)
        return LLi_val

    def get_h_val(self, x):
        """
        Args:
            x: (N*n,T) Agent states
        Return:
            h_val: (N*N, T)
        """
        # TODO this is a symmetric matrix
        h_k_vals = []
        for k in range(self.T):
            x_k = cas.reshape(x[:, k], self.N, self.n)
            h_k_i_vals = []
            for i in range(self.N):
                h_k_i = cas.vertcat(*[self.game.h(x_k[i, :].T, x_k[j, :].T) for j in range(self.N)])
                h_k_i_vals.append(h_k_i)
            h_k_vals.append(cas.vertcat(*h_k_i_vals).T)
        h_val = cas.vertcat(*h_k_vals).T
        assert h_val.shape == (self.N*self.N, self.T)
        return h_val

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
            r_vec.append(dLLi_dxi)  # n*T
            r_vec.append(dLLi_dui)  # m*T
            assert dLLi_dxi.size2() == 1
            assert dLLi_dui.size2() == 1

            # dynamics residual for f(x0,u0) = x1
            x0 = self.game.config.get_param('x0')
            u0 = cas.reshape(u[:, 0], N, m)
            f0 = self.game.f(x0[i, :].T, u0[i, :].T, i_onehot) - cas.reshape(x[:, 0], N, n)[i, :].T
            assert f0.size2() == 1
            r_vec.append(f0)  # n

            # dynamics residual for f(xk,uk) = x_{k+1}
            for k in range(1, self.T):
                xk = cas.reshape(x[:, k-1], N, n)
                xk1 = cas.reshape(x[:, k], N, n)
                uk = cas.reshape(u[:, k], N, m)
                fk = self.game.f(xk[i, :].T, uk[i, :].T, i_onehot) - xk1[i, :].T
                assert fk.size2() == 1
                r_vec.append(fk)  # (T-1)*n (entire loop)

            # collision residual for h > 0
            for k in range(1, self.T+1):
                xk = cas.reshape(x[:, k-1], N, n)  # x[k] -> x_{k+1} due to index alignment
                h_vals = [self.game.h(xk[i, :].T, xk[j, :].T) for j in range(self.N)]
                h_vals[i] = 0  # ignore self-collision, keep this dummy entry to simplify indices
                h_vals_pos = cas.fmax(cas.vertcat(*h_vals), 0)
                assert h_vals_pos.size2() == 1
                r_vec.append(h_vals_pos)  # T*N

        return cas.vertcat(*r_vec)

    def constraint_components(self, dr_dy):
        """ Re-organize the dr_dy hessian matrix to the following format
        [H A.T
         A 0 ]
        Returns:
            H, A
        """

    def residual_components(self, r):
        """ Check the residual for each subcomponents
        Args:
            r: (dim, 1) residual vector, CasADi DM matrix
        Returns:
            each component in r
        """
        r = r.toarray()
        T = self.T
        N = self.N
        n = self.n
        m = self.m
        offset = 0
        dLLi_dxi = []
        dLLi_dui = []
        f = []
        h = []
        for i in range(N):
            dLLi_dxi += list(r[offset:offset + n*T, 0])
            offset += n*T

            dLLi_dui += list(r[offset:offset + m*T, 0])
            offset += m*T

            f += list(r[offset:offset + n*T, 0])
            offset += n*T

            h += list(r[offset:offset + N*T, 0])
            offset += N*T
        assert offset == r.shape[0]
        r_Lx = np.linalg.norm(dLLi_dxi)
        r_Lu = np.linalg.norm(dLLi_dui)
        r_f = np.linalg.norm(f)
        r_h = np.linalg.norm(h)
        return r_Lx, r_Lu, r_f, r_h
