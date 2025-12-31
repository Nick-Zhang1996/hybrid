""" Residual Descent Differential Dynamic Game Solver (RD3G) with CasADi """
# pylint: disable=invalid-name, forgotten-debug-statement
# NOTE since casadi use column-major memory layout, consider changing order of indexing
# so most frequent slicing is on columns

import sys
import logging
from dataclasses import dataclass
from itertools import accumulate
from time import time

import numpy as np
import casadi as cas
import qdldl
import scipy.sparse  # sparse matrix operations
from scipy.sparse.linalg import lsqr
import scipy.linalg
from scipy.linalg import norm
import matplotlib.pyplot as plt

from rd3g.core.base_solver import BaseSolver, BaseSolverConfig, Solution
from rd3g.utilities.time_util import TimeUtil
from rd3g.utilities.util import dm_to_csc


logger = logging.getLogger('RD3G_CasADi')
logger.setLevel(logging.DEBUG)
DEBUG = False


def create_partial_identity(n, k):
    """
    Creates an n x n CSC matrix with the first k diagonal elements set to 1.
    """
    if k > n:
        raise ValueError("k cannot be larger than n")
    data = np.ones(k)
    rows = np.arange(k)
    cols = np.arange(k)
    mat = scipy.sparse.csc_matrix((data, (rows, cols)), shape=(n, n))

    return mat


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


def solve_linear(A, b, method):
    """ Solve for x in Ax = b, give x, residual, and inertia of A
    Args:
        A: (n,n) csc_matrix
        b: (N,1) csc_matrix
    Return:
        x: such that Ax-b is minimized
        res: residual, |Ax-b|_2
        inertia: (pos, neg, zero) number of positive, negative, zero eigenvals from LDL
    """
    if method == 'lsqr':
        # Solve for reduced_dy
        t0 = time()
        x, istop, itn, normr = lsqr(A, b)[:4]
        dt = time() - t0
        residual = norm(normr)
        istop_lut = {1: 'Direct Sol', 2: 'Least Square Sol', 7: 'Iter limit'}
        logger.info(f'Reduced stop:{istop_lut[istop]},{dt=}s {itn=}, {residual=}')

        # Get inertia from LDL decomposition
        A_np = A.toarray()
        lu, d, perm = scipy.linalg.ldl(A_np)
        kkt_inertia = check_inertia(d)
        logger.info(f'Reduced-system Inertia: {kkt_inertia}')
        return x, residual, kkt_inertia
    elif method == 'qdldl':
        t0 = time()
        A_upper = scipy.sparse.triu(A, format='csc')
        A_upper.eliminate_zeros()
        A_upper.sort_indices()
        A_upper.sum_duplicates()
        solver = qdldl.Solver(A_upper, upper=True)
        #  C = P @ A @ P.T, C = L @ D @ L.T
        L_zero_diag, D_diag, P_vec = solver.factors()
        qdldl_x = solver.solve(b)
        pos = np.sum(D_diag > 0)
        neg = np.sum(D_diag < 0)
        qdldl_inertia = (pos, neg, len(D_diag) - pos - neg)
        residual = norm(A @ qdldl_x - b)
        dt = time() - t0
        logger.info(f'Reduced LDL ,{dt=:.6f}s {residual=:.6f}')
        return qdldl_x, residual, qdldl_inertia

    if DEBUG:
        # check qdldl dy against lsqr result
        diff_norm = norm(qdldl_x - x)
        x_norm = norm(x)
        qdldl_x_norm = norm(qdldl_x)
        logger.info(f'{diff_norm=}, {x_norm=}, {qdldl_x_norm=}')

# NOTE: changing config requires re-run codegen, since configs are constants


@dataclass(frozen=True)
class RD3GCasadiConfig(BaseSolverConfig):
    """Configs for Residual Game."""
    tolerance: float = 5e-4
    iterations: int = 100
    # backtracking line search param
    bc_a: float = 1e-4  # alpha
    bc_b: float = 0.5  # beta
    backtracking_max_iter: int = 10
    # NOTE this is not implemented in cpp
    dynamics_residual_weight: float = 1.0
    # barrier function scaling schedule
    rho_0: float = 20.0
    # scaling rate for rho, rho+ = rho * rho_b
    rho_b: float = 1.0
    # Apply Levenberg-Marquardt Regularization
    reg: float = 1e-4


class LineSearchMaxIter(Exception):
    """ Max iteration in line search is reached without a good step size"""


class LineSearchSuccess(Exception):
    """ Line search found a step size that gives adequate residual reduction """


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

        self.guess = np.zeros((self.m, self.N, self.T))
        self.violations = None

        self.rho = self.config.rho_0
        # Levenberg-Marquardt Regularization coeff
        # When line search is stuck, larger coeff is used. Re-sets when line search succeeds.
        self.reg = self.config.reg
        self.profiler = TimeUtil(True)

        # logger.debug_enable()
        self.residual_vec = []
        self.validate()

        # CasADi objects
        self.construct_gradient_fun()

    def validate(self):
        """Check the dimension of initial state x0, guess for control."""
        assert self.guess.shape == (self.m, self.N, self.T), (
            'Incorrect self.guess dimension, '
            f'should be {(self.m, self.N, self.T)}, but got {self.guess.shape}'
        )
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

        # NOTE: casadi only works with 2D matrices, we combine n,N into first dimension
        # Slicing columns is more efficient in casadi, therefore we put T in the column dimension
        # Conceptually, the three dimensions are stacked in order of n, N, T
        # To obtain x_i_k, do this: cas.reshape(x[:,k], n, N)[:,i], note the column major storage
        # This results in a slice that's continuous in memory
        x = cas.SX.sym('x', n*N, T)
        u = cas.SX.sym('u', m*N, T)
        lamda = cas.SX.sym('lamda', n*N, T)
        # For agent i's collision against j, cas.reshape(mu[:,k], N,N)[j,i]
        mu = cas.SX.sym('mu', N*N, T)
        x0 = cas.SX.sym('x0', n, N)
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
        # y: x(n*N*T) ,u(m*N*T), lambda(n,N,T),mu(N,N,T)
        logger.debug(
            f'primal variables:{(T*N*n) +(T*N*m)} dual variables:{(N*T*n)+(T*N*N)}'
        )

        u_ref = np.zeros((m*N, T), order='F')
        # x_ref = x_1 .. x_T, NOTE the array index is offset from the math notation
        # return: (n*N, T)
        gc = self.game.config
        int_param_dm = cas.DM(gc.get_int_param_np())
        double_param_dm = cas.DM(gc.get_double_param_np())
        x_ref = self.rollout_fun_casadi(self.x0, u_ref, int_param_dm, double_param_dm)
        # NOTE to convert to np array
        # np.array(x_ref, order='F'),reshape(n,N,T, order='F') -> (n, N, T)
        lambda_ref = cas.DM.zeros((n*N, T))
        # defined for all h_k_i_j, but all values may not be used
        mu_ref = cas.DM.zeros((N*N, T))

        i = 0
        has_converged = False
        is_optimal = False
        t0 = time()
        for i in range(self.config.iterations):
            logger.info(f'--- iter {i} ---')
            x_ref, u_ref, lambda_ref, mu_ref, res, has_converged, is_optimal = self.step(
                x_ref, u_ref, lambda_ref, mu_ref)
            if has_converged:
                break
        dt = time() - t0
        msg = ''
        if has_converged and is_optimal:
            msg = 'Converged to NE'
        elif has_converged and not is_optimal:
            msg = 'Converged to saddle point'
        else:
            msg = 'Max Iteration reached'

        logger.info(f'Stop after {i} iteration because {msg}')

        return Solution(elapsed_time=dt,
                        iterations=i,
                        u=u_ref,
                        x=x_ref,
                        residual=res,
                        has_converged=has_converged,
                        is_optimal=is_optimal)

    def step(self, x_ref, u_ref, lambda_ref, mu_ref):
        """ Solver step function
        Args:
            x_ref: n*N,T, casadi.DM
            u_ref: m*N,T
            lamda: n*N,T
            mu: N*N,T
        Return:
            x_ref, u_ref, lamda, mu: updated
            res: residual
        """
        p = self.profiler
        p.s()
        p.s('prep')
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
        p.e('prep')

        p.s('Form KKT')
        r0_val = self.r_fun_casadi(x, u, lamda, mu, *params_dm)
        dr_dy_val = self.dr_dy_fun_casadi(x, u, lamda, mu, *params_dm)
        r0_np = np.array(r0_val)
        r0_norm = norm(r0_np)
        dr_dy_csc = dm_to_csc(dr_dy_val)
        p.e('Form KKT')

        # size of x, u, lamda, mu
        sizes = [0, n*N*T, m*N*T, n*N*T, N*N*T]
        offsets = list(accumulate(sizes))
        naive_method = False
        if naive_method:
            istop_lut = {1: 'Direct Sol', 2: 'Least Square Sol', 7: 'Iter limit'}
            # Naive method: solve sparse system directly
            # r0 + dr_dy @ dy = 0
            # solve for dy directly
            t0 = time()
            dy_np, istop, itn, normr = lsqr(dr_dy_csc, -r0_np)[:4]
            dt = time() - t0
            residual = norm(normr)
            logger.info(f'Full stop:{istop_lut[istop]}, {dt=}s {itn=}, {residual=}')

            # Study the LDL decomposition
            lu, d, perm = scipy.linalg.ldl(dr_dy_csc.toarray())
            del lu
            del perm
            pos, neg, zero = check_inertia(d)
            logger.info(f'Full-system Inertia: {pos, neg, zero}')

            # Check, does dy improve residual? do a line search --- Yes!
            dy = cas.DM(dy_np)
            dx, du, dlamda, dmu = cas.vertsplit(dy, offsets)
            step_size = 1.0
            step_size_vec = []
            stepped_r_vec = []
            for i in range(self.config.backtracking_max_iter):
                r_val = self.r_fun_casadi(x+step_size*cas.reshape(dx, n*N, T),
                                          u+step_size*cas.reshape(du, m*N, T),
                                          lamda+step_size*cas.reshape(dlamda, n*N, T),
                                          mu+step_size*cas.reshape(dmu, N*N, T),
                                          int_param_dm, double_param_dm
                                          )
                r_norm = norm(r_val)
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
        dLL_f_offset = N*T*(2*n+m)
        mu_in_y_offset = n*N*T + m*N*T + n*N*T
        p.s('Reduce KKT')

        H_val = np.array(self.H_val_casadi(x, *params_dm), order='F').reshape((N, N, T), order='F')
        neg_h_mask = np.array(H_val < 0).nonzero()
        inactive_r_rows = []
        inactive_y_rows = []
        for i, j, k in zip(*neg_h_mask):
            inactive_r_rows.append(dLL_f_offset + k*N*N + i*N + j)
            inactive_y_rows.append(mu_in_y_offset + k*N*N + i*N + j)
            # Check h_val is indeed negative
            # print(f'adding {i,j,k}, -> {inactive_r_rows[-1]}')
            # xk = np.array(x_ref[:, k], order='F').reshape((n, N), order='F')
            # h_val = self.h_val_casadi(cas.DM(xk[:, i]), cas.DM(xk[:, j]), *params_dm)
            # if (h_val > 0):
            #     print(i, j, k)
            #     breakpoint()
            # assert h_val < 0

        # set all self-collision to be inactive
        for k in range(T):
            for i in range(N):
                j = i
                inactive_r_rows.append(dLL_f_offset + k*N*N + i*N + j)
                inactive_y_rows.append(mu_in_y_offset + k*N*N + i*N + j)
                # Verify that h(xi,xi) > 0, because an agent always collide with itself
                # print(f'adding {i,j,k}, -> {inactive_r_rows[-1]}')
                # xk = np.array(x_ref[:, k], order='F').reshape(n, N)
                # h_val = self.h_val_casadi(cas.DM(xk[:, i]), cas.DM(xk[:, j]), *params_dm)
                # assert h_val >= 0
        # Remove zero rows & columns in the linear system
        all_r_indices = np.arange(dr_dy_csc.shape[0])
        all_y_indices = np.arange(dr_dy_csc.shape[1])
        # Get the "complement" (the indices you want to keep)
        active_r_rows = np.setdiff1d(all_r_indices, inactive_r_rows)
        active_y_rows = np.setdiff1d(all_y_indices, inactive_y_rows)

        # TODO set the relevant mu to 0
        KKT_residual = reduced_r0 = r0_np[active_r_rows, :]
        KKT = reduced_dr_dy_csc = dr_dy_csc[active_r_rows, :][:, active_y_rows]
        p.e('Reduce KKT')

        # Reduced method: solve sparse system directly
        # r0 + dr_dy @ dy = 0
        # Apply Levenberg-Marquardt Regularization
        # H = H + reg * I
        primal_var_count = (n+m)*N*T
        ind = np.arange(primal_var_count)
        reg_matrix = scipy.sparse.eye(KKT.shape[0], format="csc")
        reg_matrix[ind, ind] = self.reg
        # Apply constraint relaxation to allow AMD permutation in LDL
        ind = np.arange(primal_var_count, KKT.shape[0])
        reg_matrix[ind, ind] = -self.reg
        KKT += reg_matrix

        p.s('Solve Linear')
        reduced_dy, residual, kkt_inertia = solve_linear(KKT, -KKT_residual, method='qdldl')
        p.e('Solve Linear')

        # Inertia checking for SOSC
        # Primal variables - inactive lagrange multipliers (mu)
        in_n = n*N*T + m*N*T  # Dimension of primal vars
        in_m = T*N*(n+N) - len(inactive_r_rows)  # f, h - inactive collision constraints
        expected_inertia = (in_n, in_m, 0)
        # logger.info(f'Expected SOSC inertia {expected_inertia}')
        if expected_inertia == kkt_inertia:
            is_optimal = True
        else:
            is_optimal = False
            logger.info(f'Bad inertia: Expected {expected_inertia}, actual {kkt_inertia}')

        if DEBUG:
            p.s('Debug checking')
            KKT_np = KKT.toarray()
            # Do we have linearly dependent constraints?
            H, A = self.hessian_components(KKT)
            H_pos, H_neg, H_zero = check_inertia(H.toarray())
            # Lots of zero eigenvals
            logger.info(f'H inertia {H_pos, H_neg, H_zero}')

            A_dense = A.toarray()
            rank = np.linalg.matrix_rank(A_dense, tol=1e-10)
            logger.info(f'{A.shape=}, {rank=}')

            cond_num = np.linalg.cond(KKT_np)
            print(f"Condition Number: {cond_num}")
            p.e('Debug checking')

        # Verify residual reduction with a line search
        # Recover full dy
        dy = np.zeros(dr_dy_csc.shape[1])
        dy[active_y_rows] = reduced_dy

        p.s('Line Search')
        dx, du, dlamda, dmu = cas.vertsplit(cas.DM(dy), offsets)
        step_size = 1.0
        step_size_vec = []
        stepped_r_vec = []
        try:
            for i in range(self.config.backtracking_max_iter):
                r_val = self.r_fun_casadi(x+step_size*cas.reshape(dx, n*N, T),
                                          u+step_size*cas.reshape(du, m*N, T),
                                          lamda+step_size*cas.reshape(dlamda, n*N, T),
                                          mu+step_size*cas.reshape(dmu, N*N, T),
                                          int_param_dm, double_param_dm
                                          )
                r_norm = norm(r_val)
                step_size_vec.append(step_size)
                stepped_r_vec.append(r_norm)
                if r_norm > (1 - self.config.bc_a * step_size) * r0_norm:
                    step_size *= self.config.bc_b
                else:
                    raise LineSearchSuccess
            raise LineSearchMaxIter
        except LineSearchMaxIter:
            self.reg *= 10
        except LineSearchSuccess:
            self.reg = self.config.reg
        p.e('Line Search')

        p.s('Cleanup')
        new_x = x+step_size*cas.reshape(dx, n*N, T)
        new_u = u+step_size*cas.reshape(du, m*N, T)
        new_lamda = lamda+step_size*cas.reshape(dlamda, n*N, T)
        new_mu = mu+step_size*cas.reshape(dmu, N*N, T)
        p.e('Cleanup')
        logger.info(f'{r0_norm=:.6f}, {self.reg=}, {step_size=}, {r_norm=:.6f}')

        p.s('More debug checking')
        if DEBUG:
            # Where does the residual come from?
            r_val_np = r_val.toarray()
            r_Lx, r_Lu, r_f, r_h = self.residual_components(r_val)
            logger.info(f'Residual breakdown {r_Lx=}, {r_Lu=}, {r_f=}, {r_h=}')

            # Are inactive residual indeed inactive?
            inactive_residual = norm(r_val_np[inactive_r_rows, 0])
            # logger.debug(f'{inactive_residual=}')
            assert inactive_residual < 1e-10

            # Check the linearization is valid
            r0_np = np.array(r0_val)
            r_np = np.array(r_val)
            expected_r = r0_val * (1-step_size)
            diff = norm(expected_r - r_np)
            logger.info(f'Diff in expected r {diff=}')

            # plt.plot(step_size_vec, stepped_r_vec, '*-')
            # plt.plot(0, r0_norm, 'o')
            # plt.title('Reduced descent')
            # plt.show()
            # TODO Use a merit function of form r_val + C * h_residual
            p.e('More debug checking')
        self.residual_vec.append(r0_norm)
        p.e()

        has_converged = r_norm < self.config.tolerance

        return new_x, new_u, new_lamda, new_mu, r_norm, has_converged, is_optimal

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
            x_k: (n,N) state vector at step k
            u_k_i: (m,1) control vector for agent i at step k
            x_k1_i: (n,1) state vector for agent i at step k+1
            lamda_k: (n,N) Multiplier for dynamics constraint
            mu_k: (N,N), symmetric?,  multiplier for positive h
            i: agent index i
        Return:
            val: scalar value of lagrangian
        '''
        N = self.N
        n = self.n
        m = self.m
        assert x_k.shape == (n, N)
        assert u_k_i.shape == (m, 1)
        assert x_k1_i.shape == (n, 1)
        assert lamda_k.shape == (n, N)
        assert mu_k.shape == (N, N)
        # feasibility for h>0
        h_vals = cas.vertcat(*[self.game.h(x_k[:, i], x_k[:, j]) for j in range(self.N)])
        # NOTE add fmax here for safety, it may not be needed if
        # mu_k has structural 00 where h_vals < 0
        # we only want to sum h_val > 0
        mu_h_plus_vals = cas.dot(mu_k[:, i], cas.fmax(h_vals, 0))

        # barrier for h < 0
        h_neg_barrier_vals = -1.0/self.rho * cas.sum(cas.log(-cas.fmin(h_vals, -1e-100)))

        i_onehot = cas.SX.eye(self.N)[:, i]
        # NOTE it may be better to store lamda_k_T to take advantage of col-major storage
        dynamics_val = lamda_k[:, i].T @ (self.game.f(x_k[:, i], u_k_i, i_onehot) - x_k1_i)
        val = self.game.J(x_k, u_k_i, i_onehot) + mu_h_plus_vals + h_neg_barrier_vals + dynamics_val
        assert val.shape == (1, 1)
        return val

    def LLi(self, x, u, lamda, mu, i):
        ''' Lagrangian for agent i across all time steps
        Args:
            x: (n*N,T) Agent states
            u: (m*N,T) Agent control
            lamda: (n*N, T) Multiplier for dynamics constraint
            mu: (N*N, T) Multiplier for positive h
            i: agent index
        Return:
            val: scalar value of Lagrangian
        '''
        T = self.T
        N = self.N
        m = self.m
        n = self.n
        LLi_val = sum([self.L(cas.reshape(x[:, k - 1], n, N),
                              cas.reshape(u[:, k], m, N)[:, i],
                              cas.reshape(x[:, k], n, N)[:, i],
                              cas.reshape(lamda[:, k], n, N),
                              cas.reshape(mu[:, k - 1], N, N),
                              i)
                      for k in range(1, T)])
        x0 = self.game.config.get_param('x0')
        # x0 related terms
        i_onehot = cas.SX.eye(self.N)[:, i]
        u0_i = cas.reshape(u[:, 0], m, N)[:, i]
        LLi_val += (self.game.J(x0, u0_i, i_onehot) +
                    cas.reshape(lamda[:, 0], n, N)[:, i].T
                    @ (self.game.f(x0[:, i], u0_i, i_onehot)
                    - cas.reshape(x[:, 0], n, N)[:, i]))
        # x_T related terms
        x_T = cas.reshape(x[:, T-1], n, N)
        LLi_val += self.game.Jfi(x_T, i_onehot)
        h_vals = cas.vertcat(*[self.game.h(x_T[:, i], x_T[:, j]) for j in range(self.N)])
        mu_h_plus = cas.dot(cas.reshape(mu[:, T-1], N, N)[:, i], cas.fmax(h_vals, 0))
        h_neg = -1.0/self.rho * cas.sum(cas.log(-cas.fmin(h_vals, -1e-100)))

        LLi_val += mu_h_plus + h_neg
        assert LLi_val.shape == (1, 1)
        return LLi_val

    def get_h_val(self, x):
        """
        Args:
            x: (n*N,T) Agent states
        Return:
            h_val: (N*N, T)
        """
        # TODO this is a symmetric matrix
        h_k_vals = []
        for k in range(self.T):
            x_k = cas.reshape(x[:, k], self.n, self.N)
            h_k_i_vals = []
            for i in range(self.N):
                h_k_i = cas.vertcat(*[self.game.h(x_k[:, i], x_k[:, j]) for j in range(self.N)])
                h_k_i_vals.append(h_k_i)
            h_k_vals.append(cas.vertcat(*h_k_i_vals).T)
        h_val = cas.vertcat(*h_k_vals).T
        assert h_val.shape == (self.N*self.N, self.T)
        return h_val

    def r(self, x, u, lamda, mu):
        ''' Residual for the game
        Args:
            x: (n*N,T) Agent states
            u: (m*N,T) Agent control
            lamda: (n*N, T) Multiplier for dynamics constraint
            mu: (N*N, T) Multiplier for positive h
        Return:
            val: (nNT+mNT+nNT+NNT, 1) column vector of residual r
        '''
        T = self.T
        N = self.N
        n = self.n
        m = self.m
        # elements are column vectors
        r_vec = []
        eye = cas.SX.eye(self.N)

        # dLLi_dx n*N*T
        for k in range(T):
            for i in range(N):
                xki = cas.reshape(x[:, k], n, N)[:, i]
                dLLi_dxki = cas.jacobian(self.LLi(x, u, lamda, mu, i), xki).T
                r_vec.append(dLLi_dxki)  # n
                assert dLLi_dxki.shape == (n, 1)

        # dLLi_du m*N*T
        for k in range(T):
            for i in range(N):
                uki = cas.reshape(u[:, k], m, N)[:, i]
                dLLi_duki = cas.jacobian(self.LLi(x, u, lamda, mu, i), uki).T
                r_vec.append(dLLi_duki)  # m
                assert dLLi_duki.shape == (m, 1)

        # Dynamics residual for f(x0,u0) = x1 n*N
        for i in range(N):
            i_onehot = eye[:, i]
            x0 = self.game.config.get_param('x0')
            u0 = cas.reshape(u[:, 0], m, N)
            f0 = self.game.f(x0[:, i], u0[:, i], i_onehot) - cas.reshape(x[:, 0], n, N)[:, i]
            assert f0.shape == (n, 1)
            r_vec.append(f0)  # n

        # Dynamics residual for f(xk,uk) = x_{k+1} n*N*(T-1)
        for k in range(1, self.T):
            for i in range(N):
                i_onehot = eye[:, i]
                xk = cas.reshape(x[:, k-1], n, N)
                xk1 = cas.reshape(x[:, k], n, N)
                uk = cas.reshape(u[:, k], m, N)
                fk = self.game.f(xk[:, i], uk[:, i], i_onehot) - xk1[:, i]
                assert fk.shape == (n, 1)
                r_vec.append(fk)  # n

        # Constraint residual for h(xi, xj) N*N*T
        for k in range(1, self.T+1):
            for i in range(N):
                # collision residual for h > 0
                xk = cas.reshape(x[:, k-1], n, N)  # x[k] -> x_{k+1} due to index alignment
                h_vals = [self.game.h(xk[:, i], xk[:, j]) for j in range(self.N)]
                h_vals[i] = 0  # ignore self-collision, keep this dummy entry to simplify indices
                h_vals_pos = cas.fmax(cas.vertcat(*h_vals), 0)
                assert h_vals_pos.shape == (N, 1)
                r_vec.append(h_vals_pos)  # N

        return cas.vertcat(*r_vec)

    def hessian_components(self, dr_dy):
        """ Re-organize the dr_dy hessian matrix to the following format
        [H A.T
         A 0 ]
        Args:
            dr_dy: scipy.sparse.csc_matrix
        Returns:
            H, A
        """
        T = self.T
        N = self.N
        n = self.n
        m = self.m
        primal_n = N*T*(n+m)  # primal variables
        H = dr_dy[:primal_n, :primal_n]
        A = dr_dy[primal_n:, :primal_n]
        AT = dr_dy[:primal_n, primal_n:]
        empty = dr_dy[primal_n:, primal_n:]
        # diff = (H.T - H).toarray()
        # assert norm(diff) < 1e-10
        # diff = (AT.T - A).toarray()
        # assert norm(diff) < 1e-10

        assert np.sum(np.abs((A-AT.T).data)) < 1e-10
        assert np.sum(np.abs((H-H.T).data)) < 1e-10
        # Due to regularization, this is not empty
        # assert empty.nnz == 0
        return H, A

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
        r_Lx = norm(r[n*N*T, 0])
        offset += n*N*T
        r_Lu = norm(r[offset:offset+m*N*T])
        offset += m*N*T
        r_f = norm(r[offset:offset+n*N*T])
        offset += n*N*T
        r_h = norm(r[offset:offset+N*N*T])
        offset += N*N*T
        assert r.shape == (offset, 1)
        return r_Lx, r_Lu, r_f, r_h
