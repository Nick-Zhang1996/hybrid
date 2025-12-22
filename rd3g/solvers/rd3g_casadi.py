""" Residual Descent Differential Dynamic Game Solver (RD3G) with CasADi """
# pylint: disable=invalid-name, forgotten-debug-statement
# NOTE since casadi use column-major memory layout, consider changing order of indexing
# so most frequent slicing is on columns

import logging
from dataclasses import dataclass
from itertools import accumulate

import numpy as np
import scipy.sparse  # sparse matrix operations
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import casadi as cas

from rd3g.utilities.util import dm_to_csc
from rd3g.utilities.time_util import TimeUtil
from rd3g.core.base_solver import BaseSolver, BaseSolverConfig, Solution
from rd3g.core.base_casadi_game import CasadiGameConfig

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

        h_val = self.get_h_val(x)
        self.h_val_casadi = cas.Function('H', [x]+config_params, [h_val])

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

        lambda_ref = np.zeros((N*n, T), order='F')

        # defined for all h_k_i_j, but all values may not be used
        mu_ref = np.zeros((N*N, T), order='F')
        self.step(x_ref, u_ref, lambda_ref, mu_ref)

    def step(self, x_ref, u_ref, lambda_ref, mu_ref):
        """ Solver step function
        Args:
            x_ref: N*n,T,
            u_ref: N*m,T
            lamda: N*n,T
            mu: N*N,T
        Return:
            TBD
        """
        N = self.N
        T = self.T
        n = self.n
        m = self.m
        gc = self.game.config
        x = cas.DM(x_ref.reshape((N*n, T)))
        u = cas.DM(u_ref.reshape((N*m, T)))
        lamda = cas.DM(lambda_ref.reshape((N*n, T)))
        mu = cas.DM(mu_ref.reshape((N*N, T)))
        int_param_dm = cas.DM(gc.get_int_param_np())
        double_param_dm = cas.DM(gc.get_double_param_np())

        r0_val = self.r_fun_casadi(x, u, lamda, mu, int_param_dm, double_param_dm)
        dr_dy_val = self.dr_dy_fun_casadi(x, u, lamda, mu, int_param_dm, double_param_dm)
        r0_np = np.array(r0_val)
        dr_dy_csc = dm_to_csc(dr_dy_val)

        # Naive method: solve sparse system directly
        # r0 + dr_dy @ dy = 0
        # solve for dy directly
        dy_np, istop, itn, normr = scipy.sparse.linalg.lsqr(dr_dy_csc, -r0_np)[:4]
        # Check, does dy improve residual? do a line search --- Yes!
        dy = cas.DM(dy_np)
        # size of x, u, lamda, mu
        sizes = [0, N*n*T, N*m*T, N*n*T, N*N*T]
        offsets = list(accumulate(sizes))
        dx, du, dlamda, dmu = cas.vertsplit(dy, offsets)
        step_size = 1e-5
        step_size_vec = []
        stepped_r_vec = []
        while step_size < 1:
            r_val = self.r_fun_casadi(x+step_size*cas.reshape(dx, N*n, T),
                                      u+step_size*cas.reshape(du, N*m, T),
                                      lamda+step_size*cas.reshape(dlamda, N*n, T),
                                      mu+step_size*cas.reshape(dmu, N*N, T),
                                      int_param_dm, double_param_dm
                                      )
            step_size_vec.append(step_size)
            stepped_r_vec.append(np.linalg.norm(r_val))
            step_size *= 2
            if step_size > 1:
                step_size = 1
        plt.plot(step_size_vec, stepped_r_vec, '*-')
        plt.show()

        # Remove inactive constraints and their multiplier
        # h < 0 -> inactive cosntraint, remove rows for h, also remove columns for mu
        # r consists of all agent's residual concatenated, EACH agent has n_ri = 2nT + mT + NT rows
        # If h(i,j,k) < 0, then r[i*n_ri + 2nT + k*N + j] = 0 and can be removed
        # Also, the corresponding multiplier mu dim(NN,T) row [N*i+j, k] can be removed
        # NOTE that the same applies for flipped i,j

        # Remove zero rows & columns

        # TODO reduce size by removing empty rows and columns

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
        h_val = cas.vertcat(*h_k_vals)
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
                h_vals_pos = cas.fmax(cas.vertcat(*h_vals), 0)
                assert h_vals_pos.size2() == 1
                r_vec.append(h_vals_pos)  # T*N

        return cas.vertcat(*r_vec)
