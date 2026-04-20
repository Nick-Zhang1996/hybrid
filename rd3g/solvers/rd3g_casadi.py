""" Residual Descent Differential Dynamic Game Solver (RD3G) with CasADi """
# pylint: disable=invalid-name, forgotten-debug-statement

import logging
from dataclasses import dataclass
from itertools import accumulate
from time import time
from deprecated import deprecated

import numpy as np
import casadi as cas
import qdldl
from scipy.sparse import csc_matrix
import scipy.sparse  # sparse matrix operations
from scipy.sparse.linalg import lsqr, spsolve
import scipy.linalg
from scipy.linalg import norm
import matplotlib.pyplot as plt

from rd3g.core.base_solver import BaseSolver, BaseSolverConfig, Solution
from rd3g.utilities.time_util import TimeUtil
from rd3g.utilities.util import BASEDIR
from rd3g.utilities.casadi_util import dm_to_csc


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
# change repr() of numpy floats to look like "1.23" instead of "np.float64(1.23)" for conciseness
np.set_printoptions(legacy="1.25")


def as_numpy_array(value):
    """Convert array-like solver state to a dense numpy array."""
    if hasattr(value, 'toarray'):
        return value.toarray()
    return np.asarray(value)


@dataclass
class BrGameResult:
    """ Result of a Best Response game (per agent game) """
    is_optimal: bool
    Ki: csc_matrix
    Ki_reg: csc_matrix
    dy_i: np.ndarray
    L_zero_diag: csc_matrix
    D_diag: np.ndarray
    P_vec: np.ndarray
    reg: float

    def verify(self):
        """ Verify the LDL decomposition of Ki_reg. """
        # C = P @ Ki_reg @ P.T, C = L @ D @ L.T
        # P is the permutation matrix, P.T = inv(P)
        P = self.get_P()
        L = self.get_L()
        C_upper = scipy.sparse.triu(csc_matrix(P @ self.Ki_reg @ P.T), format='csc')
        D = scipy.sparse.diags(self.D_diag)
        C = scipy.sparse.triu(L @ D @ L.T)
        assert scipy.sparse.linalg.norm(C_upper - C) < 1e-10

    def get_inv_D(self):
        inv_D_diag = 1.0/self.D_diag
        invD = scipy.sparse.diags(inv_D_diag)
        return invD

    def get_L(self):
        L = self.L_zero_diag + scipy.sparse.eye(self.L_zero_diag.shape[0])
        return L

    def get_P(self):
        l = len(self.P_vec)
        rows = np.arange(l)
        cols = self.P_vec
        data = np.ones(l)
        P = csc_matrix((data, (rows, cols)), shape=(l, l))
        return P


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


def solve_linear(A, b, method, profiler):
    """ Solve for x in Ax = b, give x, residual, and inertia of A
    Args:
        A: (n,n) csc_matrix
        b: (N,1) np.array
    Return:
        x: such that Ax-b is minimized
        res: residual, |Ax-b|_2
        inertia: (pos, neg, zero) number of positive, negative, zero eigenvals from LDL
    """
    p = profiler
    if method == 'lsqr':
        # Solve for reduced_dy
        t0 = time()
        x, istop, itn, normr = lsqr(A, b)[:4]
        dt = time() - t0
        residual = norm(normr)
        istop_lut = {1: 'Direct Sol', 2: 'Least Square Sol', 7: 'Iter limit'}
        logger.info(f'Reduced stop:{istop_lut[istop]},{dt=}s {itn=}, {residual=}')

        # Get inertia from LDL decomposition
        # NOTE this only works for symmetric matrices
        A_np = A.toarray()
        lu, d, perm = scipy.linalg.ldl(A_np)
        del lu
        del perm
        kkt_inertia = check_inertia(d)
        logger.info(f'Reduced-system Inertia: {kkt_inertia}')
        return x, residual, kkt_inertia
    elif method == 'qdldl':
        # NOTE only works for symmetric A
        t0 = time()
        p.s('pre-process')
        A_upper = scipy.sparse.triu(A, format='csc')
        A_upper.eliminate_zeros()
        A_upper.sort_indices()
        A_upper.sum_duplicates()
        p.e('pre-process')
        p.s('structural factorization')
        # pylint:disable-next=c-extension-no-member
        solver = qdldl.Solver(A_upper, upper=True)
        p.e('structural factorization')
        #  C = P @ A @ P.T, C = L @ D @ L.T
        p.s('retrieve factorization')
        L_zero_diag, D_diag, P_vec = solver.factors()
        del L_zero_diag
        del P_vec
        p.e('retrieve factorization')
        p.s('numerical solution')
        qdldl_x = solver.solve(b)
        p.e('numerical solution')
        pos = np.sum(D_diag > 0)
        neg = np.sum(D_diag < 0)
        qdldl_inertia = (pos, neg, len(D_diag) - pos - neg)
        p.s('post-processing')
        # residual = norm(A @ qdldl_x - b)  # 30 % of total time!, use csc_matrix
        b_csc = scipy.sparse.csc_matrix(b.reshape(-1, 1))
        qdldl_x_csc = scipy.sparse.csc_matrix(qdldl_x.reshape(-1, 1))
        residual = scipy.sparse.linalg.norm(A @ qdldl_x_csc - b_csc)
        p.e('post-processing')
        dt = time() - t0
        # logger.info(f'Reduced LDL ,{dt=:.6f}s {residual=:.6f}')
        return qdldl_x, residual, qdldl_inertia
    elif method == 'spsolve':
        spsolve_x = spsolve(A, b)
        b_csc = scipy.sparse.csc_matrix(b.reshape(-1, 1))
        spsolve_x_csc = scipy.sparse.csc_matrix(spsolve_x.reshape(-1, 1))
        residual = scipy.sparse.linalg.norm(A @ spsolve_x_csc - b_csc)
        # inertia is not supported
        return spsolve_x, residual, None


def ldl_solve(A, b, profiler, solver_info=None):
    """ Solve a symmetric problem with LDL.
    If solver_info is provided, assume A is the same, and skip the symbolic factorization to save time."""
    p = profiler
    if solver_info is None:
        p.s('pre-process')
        A_upper = scipy.sparse.triu(A, format='csc')
        # A_upper.eliminate_zeros()
        # A_upper.sort_indices()
        # A_upper.sum_duplicates()
        p.e('pre-process')
        p.s('structural factorization')
        # pylint:disable-next=c-extension-no-member
        solver = qdldl.Solver(A_upper, upper=True)
        p.e('structural factorization')
    else:
        solver = solver_info['solver']
        A_upper = solver_info['A_upper']
    #  C = P @ A @ P.T, C = L @ D @ L.T
    p.s('Solve linear sys')
    qdldl_x = solver.solve(b)
    p.e('Solve linear sys')
    # residual = norm(A @ qdldl_x - b)  # 30 % of total time!, use csc_matrix below
    # p.s('Calc residual')
    # b_csc = scipy.sparse.csc_matrix(b.reshape(-1, 1))
    # qdldl_x_csc = scipy.sparse.csc_matrix(qdldl_x.reshape(-1, 1))
    # residual = scipy.sparse.linalg.norm(A @ qdldl_x_csc - b_csc)
    # p.e('Calc residual')
    res = None  # Expensvie to calculate and always machine precision
    return qdldl_x, res, {'solver': solver, 'A_upper': A_upper}


@dataclass(frozen=True)
class RD3GCasadiConfig(BaseSolverConfig):
    """Configs for Residual Game.
     NOTE Some changes here require rerun codegen and recompiling the cpp program """
    tolerance: float = 5e-4  # 1e-5  # 5e-4 in benchmark
    """ The residual threhold for stopping solver iterations. """
    iterations: int = 20  # 20 in benchmark
    """ Maximum solver iterations before giving up. """
    # Line search params
    bc_a: float = 1e-4
    """ alpha, minimal acceptable residual improvement in line search """
    bc_b: float = 0.5
    """ beta, shrink coefficient for step size after a failed step """
    line_search_max_iter: int = 10
    """ Max line search iterations. """
    reg: float = 0
    """ Initial Levenberg-Marquardt Regularization Coefficient for main KKT"""
    reg_inertia: float = 1e-5
    """ Initial Levenberg-Marquardt Regularization Coefficient for agent KKT. 
    Necessary for AMD pivoting to work. """
    inertia_correction: bool = False
    """ Apply inertia correction to each agent KKT """
    reduce_kkt_system: bool = False
    """ Eliminate equality-constrained variables before solving the main KKT system """
    rollout_each_step: bool = False
    """ Rollout control to get new state trajectory at the start of each solver iter """
    precondition_with_potential: bool = True
    """ Precondition the game KKT with a potential KKT to speed up computing"""
    max_in_reg_iter: int = 10
    """ Inertia correction max iterations """
    max_in_reg_val: float = 1.0
    """ Maximum inertia regularization """
    linear_solver_method: str = 'sparselu'  # lscg, ldl, lsqr, sparselu, superlu, umfpack
    """ Sparse linear solver used by the C++ backend """
    max_failed_line_search: int = 3
    """ Number of solver steps with line search failure before solver stops trying """
    tau_decay: float = 0.2
    """ Coefficient to shrink complementary slackness"""


class LineSearchMaxIter(Exception):
    """ Max iteration in line search is reached without a good step size"""


class LineSearchSuccess(Exception):
    """ Line search found a step size that gives adequate residual reduction """


class RD3GCasadi(BaseSolver):
    """Residual Descent Differential Dynamic Game Solver (RD3G) with CasADi

    Attributes:
        config (RD3GCasadiConfig): Configuration nametuple to
            specify solver iterations, cpp binary, tolerance etc.
        N (int): Number of agents
        T (int): Horizon length
        dt (float): Time step length
        n (int): Dimension of state for a single agent
        m (int): Dimension of control for a single agent
        x0 (np.ndarray): [N, n] Initial State for all agents
        # guess (np.ndarray): [T,N,m] Initial guess for control traj

    """

    def __init__(self, config: RD3GCasadiConfig, game, cpp_only=False):
        """ cpp_only: if True, skip constructing casadi function construction """
        BaseSolver.__init__(self, config, game)

        gc = self.game.config
        self.N = gc.N
        self.T = gc.T
        self.dt = gc.dt
        self.n = gc.n
        self.m = gc.m
        self.n_hi = gc.n_hi
        self.n_c = gc.n_c

        # Levenberg-Marquardt Regularization
        # When line search is stuck, coeff is increased. Re-sets when line search succeeds.
        self.reg = self.config.reg
        self.line_search_fail_count = 0
        self.profiler = TimeUtil(True)

        self.residual_vec = []
        self.validate()

        # Construct CasADi objects, ~3 seconds
        # Leave this for users, since if cpp backend is loaded this won't be needed
        if not cpp_only:
            self.construct_casadi_fun()
        self.cpp_solver = None

    def validate(self):
        assert isinstance(self.n, int) and self.n > 0
        assert isinstance(self.m, int) and self.m > 0
        assert isinstance(self.T, int) and self.T > 0
        assert isinstance(self.N, int) and self.N > 0
        gc = self.game.config
        assert gc.x0.shape == (self.n, self.N)
        return

    def construct_casadi_fun(self):
        """ Construct functions based on CasADi autodiff"""
        gc = self.game.config
        N = gc.N
        n = gc.n
        m = gc.m
        T = gc.T
        n_hi = gc.n_hi
        # context: (n_c*N, T) Game context, changes between iteration, but constant within iteration.
        #     This contains variables too expensive to AD.
        #     e.g. path curvature at each player position.
        context = self.game.context.get_context_sx()
        logger.info('Constructing CasADi functions... ')

        # NOTE: casadi only works with 2D matrices, we combine n,N into first dimension
        # Slicing columns is more efficient in casadi, therefore we put T in the column dimension
        # Conceptually, the three dimensions are stacked in order of n, N, T
        # To obtain x_i_k, do this: cas.reshape(x[:,k], n, N)[:,i], note the column major storage
        # This results in a slice that's continuous in memory
        x = cas.SX.sym('x', n*N, T)
        u = cas.SX.sym('u', m*N, T)
        # Lagrang multiplier for equality constraints, we only have dynamics constraint
        lamda = cas.SX.sym('lamda', n*N, T)
        # Lagrange multiplier for inequality constraints
        # h() dim: n_hi, N is organized by agents, collision constraints first
        # For agent i's collision against j, cas.reshape(mu, n_hi,N)[j,i]
        # For agent i's k-th non-collision constraint, cas.reshape(mu, n_hi,N)[N+k,i]
        mu = cas.SX.sym('mu', n_hi*N, 1)
        x0 = cas.SX.sym('x0', n, N)
        config_params = [gc.get_int_param_sx(), gc.get_double_param_sx()]

        args = [x, u, lamda, mu, context]

        self.get_n_fun = cas.Function('get_n', [], [self.n])
        self.get_m_fun = cas.Function('get_m', [], [self.m])

        # Check for nans
        # opts = {'regularity_check': True}
        opts = {}
        h_val = self.game.h(x, u, context)
        self.h_casadi = cas.Function('h', [x, u, context]+config_params, [h_val], opts)

        r_val, h_val = self.r(*args)
        self.r_casadi = cas.Function('r', args+config_params, [r_val, h_val], opts)

        y = cas.vertcat(*[cas.vec(val) for val in [x, u, lamda, mu]])
        dr_dy = cas.jacobian(r_val, y)
        self.dr_dy_casadi = cas.Function('dr_dy', args+config_params, [dr_dy], opts)

        X = self.game.rollout(x0, u)
        self.rollout_casadi = cas.Function('rollout', [x0, u]+config_params, [X])

        xki = cas.SX.sym('xki', n, 1)
        xkj = cas.SX.sym('xkj', n, 1)
        col_h_val = self.game.collision_h(xki, xkj)
        self.collision_h_casadi = cas.Function('collision_h', [xki, xkj]+config_params, [col_h_val])

        # Construct KKT matrix for K_i
        # To obtain x_i_k, do this: , note the column major storage
        self.ri_casadi_vec = []
        self.Ki_casadi_vec = []
        for i in range(N):
            xik_vec = [cas.reshape(x[:, k], n, N)[:, i] for k in range(T)]
            xi = cas.vertcat(*xik_vec)
            uik_vec = [cas.reshape(u[:, k], m, N)[:, i] for k in range(T)]
            ui = cas.vertcat(*uik_vec)
            lamdaik_vec = [cas.reshape(lamda[:, k], n, N)[:, i] for k in range(T)]
            lamdai = cas.vertcat(*lamdaik_vec)
            mui = cas.reshape(mu, n_hi, N)[:, i]
            args_i = [xi, ui, lamdai, mui]
            hi_vals = h_val[:, i]
            yi = cas.vertcat(*[cas.vec(val) for val in args_i])
            L = self.LLi(x, u, lamda, mu, i, hi_vals, context)
            ri = cas.jacobian(L, yi)
            self.ri_casadi_vec.append(cas.Function(f'r_{i}', args+config_params, [ri]))
            Ki = cas.jacobian(ri, yi)
            self.Ki_casadi_vec.append(cas.Function(f'K_{i}', args+config_params, [Ki]))

        context_val = self.game.get_context(xki)
        # Get game context for one state (n,1) -> (n_c,1)
        self.get_context_casadi = cas.Function('get_context', [xki], [context_val])

        # (n, N) -> (n_c, N)
        get_context_n_N = self.get_context_casadi.map(N)

        def get_context_nN(xnN):
            x_n_N = cas.reshape(xnN, n, N)
            return cas.reshape(get_context_n_N(x_n_N), self.n_c*self.N, 1)
        xnN = cas.SX.sym('xnN', n*N, 1)
        get_context_nN_casadi = cas.Function('get_context_nN', [xnN], [get_context_nN(xnN)])
        # (nN, T) -> (N, T)
        get_context_nN_T = get_context_nN_casadi.map(T)
        full_context_val = get_context_nN_T(x)

        # Get game context for whole state (nN, T) -> (n_c*N, T)
        self.get_full_context_casadi = cas.Function('get_full_context', [x], [full_context_val])

        logger.info('Constructing CasADi functions... Done')

    def init_cpp_backend(self):
        """ Load CPP solver"""
        # logger.error("Build directory %s not found. Did you compile?", build_dir)
        # pylint:disable-next=import-outside-toplevel, no-name-in-module
        from build.lib import rd3g_casadi
        # The source code need to be generated and compiled before the following code can be run

        # Load compiled solver, compare results
        # Example for sending matrices to/from casadi
        game = self.game
        module_name = game.__module__.rsplit('.', maxsplit=1)[-1]
        solver_config = self.config

        try:
            # pylint:disable-next=c-extension-no-member
            self.cpp_solver = rd3g_casadi.Rd3gCasadi(
                game.config.N,
                game.config.T,
                game.config.n_hi,
                game.config.dt,
                solver_config.bc_a,
                solver_config.bc_b,
                solver_config.reg,
                solver_config.reg_inertia,
                solver_config.inertia_correction,
                solver_config.rollout_each_step,
                solver_config.tolerance,
                solver_config.tau_decay,
                solver_config.line_search_max_iter,
                solver_config.max_failed_line_search,
                solver_config.iterations,
                solver_config.max_in_reg_iter,
                solver_config.max_in_reg_val,
                solver_config.linear_solver_method,
                0,  # 0:error, 1:warning, 2:info, 3:debug
                BASEDIR,
                module_name
            )
        except RuntimeError as e:
            if 'Cannot load shared library' in str(e):
                logger.error('CasADi Failed to load dll, make sure its compiled')
            raise

    def solve_cpp_backend(self, u_ref=None):
        if self.cpp_solver is None:
            logger.error('Call init_cpp_backend() first')
            raise RuntimeError
        x0 = self.game.config.x0

        gc = self.game.config
        params_np = [gc.get_int_param_np(), gc.get_double_param_np()]

        if u_ref is None:
            u_ref = np.zeros((self.m*self.N, self.T), order='F')
        assert np.isfortran(x0)
        assert np.isfortran(u_ref)
        # x0 is still passed explicitly for rollout generation, while the config parameters
        # provide the symbolic x0 used inside the generated CasADi functions.

        t0 = time()
        # reduced_dy, res, inertia = self.cpp_solver.solve(x0, u_ref, *params_np)
        retval = self.cpp_solver.solve(x0, u_ref, *params_np)
        x, u, lamda, mu, residual, has_converged, is_optimal, i, msg = retval
        dt = time()-t0
        del lamda
        del mu
        del msg
        logger.debug(f"cpp solve() took: {dt:.6f}s")

        return Solution(elapsed_time=dt,
                        iterations=i,
                        u=u,
                        x=x,
                        residual=float(residual),
                        has_converged=has_converged,
                        is_optimal=is_optimal)

    @deprecated
    def solve_cpp_backend_rand_restart(self, restarts=10):
        """ Naively run solve() multiple times with random initial guess"""
        if self.cpp_solver is None:
            logger.error('Call init_cpp_backend() first')
            raise RuntimeError

        gc = self.game.config
        params_np = [gc.get_int_param_np(), gc.get_double_param_np()]
        x0 = gc.x0
        # NOTE this is specific to car merge game
        logger.info('Solving game with 10 random restarts')
        logger.warning("Using sample u specific to car merging game")
        u_mean = np.array([-0.02230492, -0.00410712])
        u_cov = np.array([[0.10592138, 0.00403225], [0.00403225, 0.00832558]])

        t0 = time()
        for i in range(restarts):
            raw_samples = np.random.multivariate_normal(u_mean, u_cov, size=(self.N, self.T))
            # raw_sampled: (N, T, m) -> [Agent, Time, Control_Dim]
            # reshaped_samples:  (m, N, T) -> [Agent, Control_Dim, Time]
            reshaped_samples = raw_samples.transpose(2, 0, 1).reshape(self.m * self.N, self.T)
            u_ref = np.array(reshaped_samples, order='F')
            assert np.isfortran(x0)
            assert np.isfortran(u_ref)

            # reduced_dy, res, inertia = self.cpp_solver.solve(x0, u_ref, *params_np)
            retval = self.cpp_solver.solve(x0, u_ref, *params_np)
            x, u, lamda, mu, residual, has_converged, is_optimal, i, msg = retval
            del lamda
            del mu
            del msg
            if is_optimal and has_converged:
                break
        dt = time()-t0
        logger.debug(f"cpp solver took : {dt:.6f}, restarts={i}")

        return Solution(elapsed_time=dt,
                        iterations=i,
                        u=u,
                        x=x,
                        residual=residual,
                        has_converged=has_converged,
                        is_optimal=is_optimal)

    def solve(self, u_ref=None):
        gc = self.game.config
        N = gc.N
        T = gc.T
        n = gc.n
        m = gc.m
        n_hi = gc.n_hi
        x0 = gc.x0
        # y: x(n*N*T) ,u(m*N*T), lambda(n,N,T),mu(n_hi*N)
        logger.debug(
            f'Primal variables:{(T*N*n) + (T*N*m)} Dual variables:{(N*T*n)+n_hi*N}'
        )
        params_np = [gc.get_int_param_np(), gc.get_double_param_np()]
        tau = 0.1  # Perturbed complementary slackness mu * s = tau > 0, homotopy param -> 0

        if u_ref is None:
            u = np.zeros((m*N, T), order='F')
        else:
            u = u_ref
        # x = x_1 .. x_T, NOTE the array index is offset from the math notation
        # (n*N, T)
        x = self.rollout_casadi(x0, u, *params_np)
        # x = self.cpp_solver.rollout(x0, u, *params_np)
        # To convert to np array
        # np.asarray(x_ref, order='F'),reshape(n,N,T, order='F') -> (n, N, T)
        context = self.get_full_context_casadi(x)

        # self.h_casadi = cas.Function('h', [x, u, context]+config_params, [h_val], opts)
        h_val = self.h_casadi(x, u, context, *params_np)
        s = cas.fmax(1e-2, -cas.reshape(h_val, n_hi*N, 1))  # Slack variable h(x,u) + s = 0, s>=0
        mu = tau / s  # Multiplier for h(x,u) + s (n_hi*N, 1)
        lamda = cas.DM.zeros((n*N, T))  # Multiplier for dynamics constraints
        filter_state = []

        i = 0
        converged = False
        optimal = False
        self.line_search_fail_count = 0
        msg = 'Max iteration has been reached'
        t0 = time()
        for i in range(self.config.iterations):
            logger.info(f'--- iter {i} ---')
            if self.config.rollout_each_step:
                x = self.rollout_casadi(x0, u, *params_np)
            x, u, lamda, mu, s, res, converged, optimal, filter_state, _ = self.step(
                x, u, lamda, mu, s, filter_state, tau)

            current_gap = np.sum(np.asarray(s) * np.asarray(mu)) / (n_hi * N)
            tau = max(1e-8, self.config.tau_decay * current_gap)
            logger.debug('Perturbed Complementary Slackness: tau=%.8f', tau)
            if converged:
                break
            if self.line_search_fail_count >= self.config.max_failed_line_search:
                msg = 'Line search no progress'
                break
        dt = time() - t0
        if converged and optimal:
            msg = 'Converged to NE'
        elif converged and not optimal:
            msg = 'Converged to saddle point'

        logger.info(f'Stop after {i} iteration because {msg}')

        # u_ref: m*N, T
        # sol.u: T,N,m
        # xn*N, T
        # sol.x: T,N,n
        return Solution(elapsed_time=dt,
                        iterations=i,
                        u=as_numpy_array(u).reshape((m, N, T), order='F'),
                        x=as_numpy_array(x).reshape((n, N, T), order='F'),
                        residual=res,
                        has_converged=converged,
                        is_optimal=optimal)

    def step(self, x, u, lamda, mu, s, filter_state, tau):
        """ Solver step function
        Args:
            x: n*N,T, casadi.DM
            u: m*N,T
            lamda: n*N,T
            mu: n_hi*N, 1
            s: (n_hi*N, 1) slack variable
            filter_state: list[(infeasibility, residual)] Previously rejected states
            tau: float, Perturbed complementary slackness
        Return:
            x:
            u:
            lamda:
            mu:
            s: updated primal/dual variables
            res: residual
            converged: Bool for algorithm converged, residual < threshold
            is_optimal: Bool for algorithm has correct inertia for PD projected hessian
            filter_state: Updated filter for line search
            debug_dict: Dict of process variables
        """
        p = self.profiler
        p.s()
        p.s('prep')
        gc = self.game.config
        N = gc.N
        T = gc.T
        n = gc.n
        m = gc.m
        n_hi = gc.n_hi
        nNT = n*N*T
        mNT = m*N*T

        int_param_dm = cas.DM(gc.get_int_param_np())
        double_param_dm = cas.DM(gc.get_double_param_np())
        params_dm = [int_param_dm, double_param_dm]
        context_dm = self.get_full_context_casadi(x)
        args = [x, u, lamda, mu, context_dm, *params_dm]
        p.e('prep')

        p.s('Form KKT')
        r0_val, h0_val = self.r_casadi(*args)
        dr_dy_val = self.dr_dy_casadi(*args)
        # Add eliminated slack variable as a function of mu
        LHS = dr_dy_val
        mu_offset = nNT + mNT + nNT
        LHS[mu_offset:, mu_offset:] += - cas.diag(s / mu)
        RHS = -r0_val
        RHS[mu_offset:, 0] += - tau / mu

        RHS_np = np.asarray(RHS)
        p.e('Form KKT')

        p.s('Inertia Checking')
        # Formulate the KKT matrix for each agent's best response game, verify inertia, and solve
        is_optimal = True
        saddle_agent_idx = []
        br_game_vec = []
        for i in range(N):
            # Construct KKT matrix for agent i K_i
            Ki = self.Ki_casadi_vec[i](*args)
            Ki = dm_to_csc(Ki)
            ri = self.ri_casadi_vec[i](*args)
            # self.check_Ki(Ki)

            # Apply Levenberg-Marquardt Regularization
            # H = H + reg * I
            primal_var_count = (n+m)*T
            ind = np.arange(primal_var_count)
            reg_matrix = scipy.sparse.eye(Ki.shape[0], format="csc")
            reg_matrix[ind, ind] = self.config.reg_inertia
            # Apply constraint relaxation to allow AMD permutation in LDL
            ind = np.arange(primal_var_count, Ki.shape[0])
            reg_matrix[ind, ind] = -self.config.reg_inertia
            Ki_reg = Ki + reg_matrix

            in_Ki, solver = self.get_inertia(Ki_reg)
            dy_i = solver.solve(-ri.toarray())

            exp_in_Ki = (primal_var_count, Ki.shape[0]-primal_var_count, 0)
            # logger.debug(f'K{i} inertia {in_Ki}, optimal {exp_in_Ki}')
            L_zero_diag, D_diag, P_vec = solver.factors()

            reg = 0
            if self.config.inertia_correction:
                if in_Ki != exp_in_Ki:
                    is_optimal = False
                    saddle_agent_idx.append(i)
                    data = np.full(primal_var_count, 1)
                    indices = np.arange(primal_var_count)
                    I_H = scipy.sparse.csc_matrix((data, (indices, indices)), shape=Ki_reg.shape)
                    # Binary search to find minimal reg to correct inertia
                    reg_upper = 2.0  # Gives good inertia
                    reg_lower = 1.0  # Gives bad inertia
                    reg_upper_in, _ = self.get_inertia(Ki_reg+reg_upper*I_H)
                    reg_lower_in, _ = self.get_inertia(Ki_reg+reg_lower*I_H)
                    reg_iter = 0
                    for reg_iter in range(self.config.max_in_reg_iter):
                        if reg_upper_in != exp_in_Ki:  # Inc upper bound
                            reg_lower = reg_upper
                            reg_lower_in = reg_upper_in
                            reg_upper *= 2
                            reg_upper_in, _ = self.get_inertia(Ki_reg+reg_upper*I_H)
                        elif reg_lower_in == exp_in_Ki:  # Dec lower bound
                            reg_upper = reg_lower
                            reg_upper_in = reg_lower_in
                            reg_lower /= 2
                            reg_lower_in, _ = self.get_inertia(Ki_reg+reg_lower*I_H)
                        else:
                            reg_middle = (reg_upper + reg_lower) / 2
                            reg_middle_in, _ = self.get_inertia(Ki_reg+reg_middle*I_H)
                            if reg_middle_in == exp_in_Ki:
                                reg_upper = reg_middle
                                reg_upper_in = reg_middle_in
                            else:
                                reg_lower = reg_middle
                                reg_lower_in = reg_middle_in
                        if (reg_upper - reg_lower)/reg_upper < 0.1:
                            break
                        if reg_upper > self.config.max_in_reg_val:
                            reg_upper = self.config.max_in_reg_val
                            break
                    reg = reg_upper
                    if reg_upper_in != exp_in_Ki:
                        logger.warning(
                            f'Fail to correct K{i} inertia: {reg_upper_in}, {reg_iter=}, {reg=}')
                    else:
                        logger.debug(f'corrected K{i} {reg_iter=}, {reg=}')

            br_game_vec.append(BrGameResult(is_optimal, Ki, Ki_reg,
                               dy_i, L_zero_diag, D_diag, P_vec, reg))

        p.e('Inertia Checking')
        reg_vec = [val.reg for val in br_game_vec]
        logger.info(f'Saddle agents: {saddle_agent_idx}, reg: {reg_vec}')

        if self.config.inertia_correction:
            # Inertia correcting regularization
            in_reg_mtx = self.make_full_KKT_reg(reg_vec)
            LHS += in_reg_mtx

        primal_var_count = (n+m)*N*T
        # size of x, u, lamda, mu
        sizes = [0, n*N*T, m*N*T, n*N*T, n_hi*N]
        offsets = list(accumulate(sizes))

        if self.config.precondition_with_potential:
            self.reg = self.config.reg_inertia
        # Apply Levenberg-Marquardt Regularization
        # H = H + reg * I
        ind = np.arange(primal_var_count)
        reg_matrix = scipy.sparse.eye(LHS.shape[0], format="csc")
        reg_matrix[ind, ind] = self.reg
        # Apply constraint relaxation to allow AMD permutation in LDL
        ind = np.arange(primal_var_count, LHS.shape[0])
        reg_matrix[ind, ind] = -self.reg
        LHS += reg_matrix
        LHS_csc = dm_to_csc(LHS)
        RHS_np = np.asarray(RHS)

        if self.config.precondition_with_potential:
            p.s('Precondition')
            p.s('prep')
            LHS_csc_T = LHS_csc.T
            S_csc = (LHS_csc + LHS_csc_T) * 0.5
            A_csc = (LHS_csc - LHS_csc_T) * 0.5
            p.e('prep')
            # DEBUG: find spectral radius of inv(S) @ A
            dy = np.zeros((2*nNT+mNT+n_hi*N, 1))
            t0 = time()
            solver_info = None
            last_residual = np.inf
            for i in range(10):
                iter_LHS = S_csc
                iter_RHS = RHS_np - A_csc @ dy
                new_dy, _, solver_info = ldl_solve(iter_LHS, iter_RHS, p, solver_info)
                p.s('total_res')
                total_res = np.linalg.norm(LHS @ new_dy - RHS)
                p.e('total_res')
                if total_res > last_residual:
                    break
                dy = new_dy.reshape(-1, 1)
                logger.debug(f'Preconditioned iter {total_res=}')
                if total_res < 1e-2 or total_res > 0.9 * last_residual:
                    break
                last_residual = total_res
            dt = time() - t0
            logger.info(f'Preconditioned KKT: {dt=:.4f}')
            p.e('Precondition')
        else:
            p.s('Solve Linear (full KKT)')
            t0 = time()
            # NOTE change back to dy. we only run this segment for timing comparison
            dy, residual, _ = solve_linear(LHS_csc, RHS_np, method='spsolve', profiler=p)
            dt = time() - t0
            r0_norm = np.linalg.norm(RHS_np)
            logger.info(f'Full KKT :{dt=:.4f}, {r0_norm=:.4f}, {residual=:.4f}')
            p.e('Solve Linear (full KKT)')

        p.s('Line Search')
        dx, du, dlamda, dmu = cas.vertsplit(cas.DM(dy), offsets)
        ds = -s + (tau - s * dmu)/mu
        # Fraction to full step, must < 1.0
        # Step size is upper bounded by s + step_size*ds > 0.05 s, also mu

        # Primal Step Size (for x, u, s)
        # Fraction-to-boundary rule
        raw_s = -0.995 * np.asarray(s) / np.asarray(ds)
        max_ss_s = raw_s[raw_s > 0]
        alpha_p = np.min(np.hstack([max_ss_s, [1.0]])).item()

        # Dual Step Size (for lamda, mu)
        raw_mu = -0.995 * np.asarray(mu) / np.asarray(dmu)
        max_ss_mu = raw_mu[raw_mu > 0]
        alpha_d = np.min(np.hstack([max_ss_mu, [1.0]])).item()

        logger.debug('Max step size primal: %.8f, dual: %.8f, smaller one is used', alpha_p, alpha_d)
        alpha_p = alpha_d = np.min([alpha_d, alpha_p]).item()  # Use same step size for primal dual
        step_size = 1.0

        # Filter line search
        # To simplify the linear problem, we eliminated the complementary slackness
        # For the line search, we must restore the full residual from r
        # r: [dL/dx, dL/du, f(x,u)-x, h(x)]
        # full_r: [dL/dx, dL/du, f(x,u)-x, h(x)+s, \mu s-tau]
        full_r = np.vstack([r0_val, mu * s - tau])
        full_r[mu_offset:mu_offset+n_hi*N] += s
        # Perturbed complementary slackness residual
        comp_res = np.linalg.norm(full_r[-n_hi*N:, 0], 1).item()
        # Infeasibility residual
        primal_res = np.linalg.norm(full_r[nNT+mNT:-n_hi*N, 0], 1).item()
        # Optimality residual
        dual_res = comp_res + np.linalg.norm(full_r[:nNT+mNT, 0], 1).item()
        logger.debug(f'step_size=0, {primal_res=:.5f}(feas), {dual_res=:.5f}(opt)')

        # Initialize filter, set a large upper bound for residual
        if len(filter_state) == 0:
            filter_state.append((max(primal_res*1.2, 1e4), -np.inf))
        try:
            for _ in range(self.config.line_search_max_iter):
                # Evaluate primal/dual residual at trial point
                new_x = x+step_size*alpha_p*cas.reshape(dx, n*N, T)
                new_u = u+step_size*alpha_p*cas.reshape(du, m*N, T)
                new_lamda = lamda+step_size*alpha_d*cas.reshape(dlamda, n*N, T)
                new_mu = mu+step_size*alpha_d*cas.reshape(dmu, n_hi*N, 1)
                new_s = s + step_size * alpha_p * ds
                new_context = self.get_full_context_casadi(new_x)
                # logger.debug(f"Min new_mu at step {step_size}: {cas.mmin(new_mu)}")
                r_val, h_val = self.r_casadi(new_x,
                                             new_u,
                                             new_lamda,
                                             new_mu,
                                             new_context,
                                             int_param_dm, double_param_dm
                                             )
                full_r = np.vstack([r_val, new_mu * new_s - tau])
                full_r[mu_offset:mu_offset+n_hi*N] += new_s
                trial_comp_res = np.linalg.norm(full_r[-n_hi*N:, 0], 1).item()
                # Infeasibility residual
                trial_primal_res = np.linalg.norm(full_r[nNT+mNT:-n_hi*N, 0], 1).item()
                # Optimality residual
                trial_dual_res = trial_comp_res + np.linalg.norm(full_r[:nNT+mNT, 0], 1).item()
                logger.debug(f'{step_size=:.8f}, {trial_primal_res=:.5f}, {trial_dual_res=:.5f}')
                improve_optimality = trial_dual_res < dual_res - self.config.bc_a * primal_res
                improve_feasibility = trial_primal_res < (1-self.config.bc_a)*primal_res
                # Dominated by starting point?
                if (not improve_optimality) and (not improve_feasibility):
                    # logger.debug('No improvement over initial state')
                    step_size *= self.config.bc_b
                    continue

                # Dominated by filter history?
                is_dominated = False
                for _primal_res, _dual_res in filter_state:
                    _improve_optimality = trial_dual_res < _dual_res - self.config.bc_a * _primal_res
                    _improve_feasibility = trial_primal_res < (1-self.config.bc_a)*_primal_res
                    if (not _improve_optimality) and (not _improve_feasibility):
                        # logger.debug('Dominated by filter')
                        is_dominated = True
                        break
                if is_dominated:
                    step_size *= self.config.bc_b
                    continue

                # Successful line search
                if (not improve_optimality) and improve_feasibility:
                    # Improved feasibility at the expense of optimality.
                    # Add to filter so we don't regress in future
                    filter_state.append((max(1e-4, primal_res), dual_res))
                    # logger.debug('Adding to filter')
                raise LineSearchSuccess
            # NOTE: future enhance What if no improvement at all? add infeasibility correction step
            raise LineSearchMaxIter
        except LineSearchMaxIter:
            self.reg = np.clip(self.reg * 10, a_min=1e-10, a_max=0.1)
            self.line_search_fail_count += 1
            step_size = 0.0
            new_x = x
            new_u = u
            new_lamda = lamda
            new_mu = mu
            new_s = s
            logger.debug(f"Max Iter reached, Inflating KKT regularization to {self.reg}")
        except LineSearchSuccess:
            logger.debug("Line search success")
            self.reg = self.config.reg
            self.line_search_fail_count = 0
        p.e('Line Search')

        h_val_np = np.asarray(h_val, order='F').flatten(order='F')
        h_pos = np.linalg.norm(h_val_np[h_val_np > 0], 1)
        logger.info(
            f'{primal_res=:.6f}, {dual_res=:.6f},'
            f'{trial_primal_res=:.6f}, {trial_dual_res=:.6f},'
            f'{h_pos=:.5f},'
            f'{self.reg=}, {step_size=:.6f}')

        res = trial_primal_res + trial_dual_res
        self.residual_vec.append(res)
        p.e()

        converged = res < self.config.tolerance
        debug_dict = {}

        return new_x, new_u, new_lamda, new_mu, new_s, res, converged, is_optimal, filter_state, debug_dict

    @deprecated
    def debug(self, u):
        """Compare selected Python and C++ intermediate quantities for debugging."""
        self.init_cpp_backend()
        self.solve_cpp_backend(u)
        N = self.N
        T = self.T
        n = self.n
        m = self.m
        n_hi = self.n_hi
        gc = self.game.config
        params_np = [gc.get_int_param_np(), gc.get_double_param_np()]
        # return: (n*N, T)
        x = self.rollout_casadi(self.game.config.x0, u, *params_np)
        lamda = cas.DM.zeros((n*N, T))
        mu = cas.DM.zeros((n_hi*N, 1))
        # new_x, new_u, _, _, res, has_converged, is_optimal, debug_dict = self.step(x, u, lamda, mu)

        # Check each intermediate variable
        context_py = self.get_full_context_casadi(x)
        context_cpp = self.cpp_solver.debug_get_context()
        context_cpp = np.array(context_cpp, order='F').reshape((12, 20), order='F')
        context_diff = np.linalg.norm(context_py - context_py)
        print(f'{context_diff=}')
        # Context identical -- verified
        # r0 = r(x,u, lamda, mu, context)
        # r0 diff = 60
        x_cpp = self.cpp_solver.debug_get_x()
        u_cpp = self.cpp_solver.debug_get_u()
        print(f'x_diff = {np.linalg.norm(x_cpp-x)}')
        print(f'u_diff = {np.linalg.norm(u_cpp-u)}')
        r0_py = debug_dict['full_r0']
        res = self.cpp_solver.debug_get_full_r0()
        r0_cpp = csc_matrix(
            (res.data, res.row, res.colind),
            shape=res.shape
        )
        r0_diff = np.linalg.norm(r0_py - r0_cpp)
        print(f'{r0_diff=}')

        KKT_py = debug_dict['full_KKT']
        res = self.cpp_solver.debug_get_full_KKT()  # SparseMatrixResult
        KKT_cpp = csc_matrix(
            (res.data, res.row, res.colind),
            shape=res.shape
        )
        KKT_diff = scipy.sparse.linalg.norm(KKT_py - KKT_cpp)
        print(f'{KKT_diff=}')

        dy_py = debug_dict['full_dy']
        dy_cpp = self.cpp_solver.debug_get_full_dy()
        dy_diff = np.sum(np.abs(dy_py - dy_cpp))
        print(f'{dy_diff=}')
        res_py = scipy.linalg.norm(KKT_py @ dy_py + r0_py)
        res_cpp = scipy.linalg.norm(KKT_cpp @ dy_cpp + r0_cpp)
        print(f'{res_py=}')
        print(f'{res_cpp=}')
        breakpoint()

    def _rollout_full_x(self, u_ref, x_ref=None):
        """ Rollout from u_ref, prepend x0 to beginning. If x_ref is provided, then just prepend x0
        Args:
            u_ref: (m, N, T)
            x_ref: Optional, (n, N, T)
        Output:
            full_x: (n, N, T+1)
        """
        n = self.n
        m = self.m
        T = self.T
        N = self.N
        assert u_ref.shape == (m, N, T)
        x0 = self.game.config.x0

        # n*N, T
        if x_ref is None:
            u_cat = u_ref.reshape((m*N, T), order='F')
            gc = self.game.config
            params_np = [gc.get_int_param_np(), gc.get_double_param_np()]
            if self.cpp_solver is None:
                # Use rollout_casadi, pure python
                try:
                    x_ref = self.rollout_casadi(x0, u_cat, *params_np).full()
                except AttributeError:
                    logger.error('cpp_only must be False to populate rollout_casadi()')
                    raise
            else:
                # Use cpp rollout
                x_ref = self.cpp_solver.rollout(x0, u_cat, *params_np)

        full_x = np.dstack([x0[:, :, np.newaxis], x_ref.reshape((n, N, T), order='F')])
        assert full_x.shape == (n, N, T+1)
        return full_x

    def visualize(self, u_ref, x_ref=None, save=False, show=True):
        """ Visualize the game with given and control (u) in a single frame.
        Args:
            u_ref: (m, N, T, order='F')
        """
        u_ref = u_ref.reshape((self.m, self.N, self.T), order='F')
        x_ref = self._rollout_full_x(u_ref, x_ref)
        return self.game.visualize(u_ref, x_ref, show=show, save=save)

    def animate(self, u_ref, x_ref=None, save_gif=False, save_snapshots=False):
        """ Animate the game with given and control (u).
        Args:
            u_ref: (m, N, T, order='F')
        """
        u_ref = u_ref.reshape((self.m, self.N, self.T), order='F')
        x_ref = self._rollout_full_x(u_ref, x_ref)
        self.game.animate(u_ref, x_ref, show=True, save_gif=save_gif, save_snapshots=save_snapshots)

    def final(self):
        self.profiler.summary()
        plt.plot(self.residual_vec, '*-')
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Residual (exp)')
        plt.show()

    # ----- derivatives and other generic math functions ----
    # NOTE revised for casadi

    def L(self, x_k, u_k_i, x_k1_i, lamda_k, mu_i, i, context_k):
        ''' Lagrangian for agent i at time k, excluding inequality constraints
        Args:
            x_k: (n, N) state vector at step k
            u_k_i: (m, 1) control vector for agent i at step k
            x_k1_i: (n, 1) state vector for agent i at step k+1
            lamda_k: (n, N) Multiplier for dynamics constraint
            mu_i: (n_hi, 1),  (unused) multiplier for inequality constraint h()
            i: agent index i
            context_k: (n_c, N) context at step k
        Return:
            val: scalar value of lagrangian
        '''
        N = self.N
        n = self.n
        m = self.m
        n_hi = self.n_hi
        assert x_k.shape == (n, N)
        assert u_k_i.shape == (m, 1)
        assert x_k1_i.shape == (n, 1)
        assert lamda_k.shape == (n, N)
        assert mu_i.shape == (n_hi, 1)

        i_onehot = cas.SX.eye(self.N)[:, i]
        # NOTE it may be better to store lamda_k_T to take advantage of col-major storage
        dynamics_val = lamda_k[:, i].T @ (self.game.f(x_k[:, i],
                                                      u_k_i, i_onehot, context_k[:, i]) - x_k1_i)
        val = self.game.J(x_k, u_k_i, i_onehot) + dynamics_val
        assert val.shape == (1, 1)
        return val

    def LLi(self, x, u, lamda, mu, i, hi_vals, context):
        ''' Lagrangian for agent i across all time steps
        Args:
            x: (n*N, T) Agent states
            u: (m*N, T) Agent control
            lamda: (n*N, T) Multiplier for dynamics constraint
            mu: (n_hi*N, 1) Multiplier for positive h
            i: agent index
            hi_vals: (n_hi, 1) h() values for agent i
            context: (n_c*N, T) context variable of game
        Return:
            val: scalar value of Lagrangian
        '''
        T = self.T
        N = self.N
        m = self.m
        n = self.n
        n_hi = self.n_hi
        n_c = self.n_c
        LLi_val = sum([self.L(cas.reshape(x[:, k - 1], n, N),
                              cas.reshape(u[:, k], m, N)[:, i],
                              cas.reshape(x[:, k], n, N)[:, i],
                              cas.reshape(lamda[:, k], n, N),
                              cas.reshape(mu, n_hi, N)[:, i],
                              i,
                              cas.reshape(context[:, k], n_c, N))
                       for k in range(1, T)])

        # feasibility for h>0
        # NOTE add fmax here for safety,
        # we only want to sum h_val > 0
        mu_i = cas.reshape(mu, n_hi, N)[:, i]
        LLi_val += cas.dot(mu_i, hi_vals)

        x0 = self.game.config.get_param('x0')
        # x0 related terms
        lamda_0_i = cas.reshape(lamda[:, 0], n, N)[:, i]  # k=0
        i_onehot = cas.SX.eye(self.N)[:, i]
        u0_i = cas.reshape(u[:, 0], m, N)[:, i]
        context_i_0 = cas.reshape(context[:, 0], n_c, N)[:, i]
        LLi_val += (self.game.J(x0, u0_i, i_onehot) +
                    cas.dot(lamda_0_i, self.game.f(x0[:, i], u0_i, i_onehot, context_i_0)
                    - cas.reshape(x[:, 0], n, N)[:, i]))
        # x_T related terms
        x_T = cas.reshape(x[:, T-1], n, N)
        LLi_val += self.game.Jfi(x_T, i_onehot)

        assert LLi_val.shape == (1, 1)
        return LLi_val

    def r(self, x, u, lamda, mu, context):
        ''' Residual for the game
        Args:
            x: (n*N, T) Agent states
            u: (m*N, T) Agent control
            lamda: (n*N, T) Multiplier for dynamics constraint
            mu: (n_hi*N, 1) Multiplier for positive h
            context: (n_c*N, T) Game context, changes between iteration, but constant within iteration.
                This contains variables too expensive to AD.
                e.g. path curvature at each player position.
        Return:
            r_val: (nNT+mNT+nNT+n_hi*N, 1) column vector of residual r
            h_val: (n_hi, N) Result of h(x, u), which is needed for active set on constraints
        '''
        T = self.T
        N = self.N
        n = self.n
        m = self.m
        n_hi = self.n_hi
        n_c = self.n_c
        # elements are column vectors
        r_vec = []
        eye = cas.SX.eye(self.N)
        h_val = self.game.h(x, u, context)

        # dLLi_dx for all i, n*N*N*T
        # for LL_idx in range(N):  # dLL[i]
        #     # dLLi_dx{-i} is a part of the residual, so two agent indices
        #     hi_val = h_val[:, LL_idx]
        #     # dLLi_dx n*N*T
        #     for k in range(T):
        #         for i in range(N):  # dLLi_dx[k][i]
        #             xki = cas.reshape(x[:, k], n, N)[:, i]
        #             dLLi_dxki = cas.jacobian(self.LLi(x, u, lamda, mu, LL_idx, hi_val), xki).T
        #             r_vec.append(dLLi_dxki)  # n
        #             assert dLLi_dxki.shape == (n, 1)

        x_vec = cas.vec(x)
        u_vec = cas.vec(u)

        # dLLi_dx n*N*T
        dLLi_dx_vec = []
        dLLi_du_vec = []
        for i in range(N):
            hi_val = h_val[:, i]
            lagrangian_i = self.LLi(x, u, lamda, mu, i, hi_val, context)
            dLLi_dx_vec.append(cas.jacobian(lagrangian_i, x_vec).T)
            dLLi_du_vec.append(cas.jacobian(lagrangian_i, u_vec).T)

        for k in range(T):
            xk_offset = k * n * N
            for i in range(N):  # dLLi_dx[k][i]
                dLLi_dxki = dLLi_dx_vec[i][xk_offset + i*n:xk_offset + (i+1)*n]
                r_vec.append(dLLi_dxki)  # n
                assert dLLi_dxki.shape == (n, 1)

        # dLLi_dui for all i, m*N*T
        # Unlike dLLi_dx, dLL[i]_du[i] only applies to the same index set, dLLi_du{-i} != 0
        for k in range(T):
            uk_offset = k * m * N
            for i in range(N):
                dLLi_duki = dLLi_du_vec[i][uk_offset + i*m:uk_offset + (i+1)*m]
                r_vec.append(dLLi_duki)  # m
                assert dLLi_duki.shape == (m, 1)

        # Dynamics residual for f(x0,u0) - x1  (n*N)
        for i in range(N):
            i_onehot = eye[:, i]
            x0 = self.game.config.get_param('x0')
            u0 = cas.reshape(u[:, 0], m, N)
            context_i_0 = cas.reshape(context[:, 0], n_c, N)
            f0 = self.game.f(x0[:, i], u0[:, i], i_onehot, context_i_0[:, i]) - \
                cas.reshape(x[:, 0], n, N)[:, i]
            assert f0.shape == (n, 1)
            r_vec.append(f0)  # n

        # Dynamics residual for f(xk,uk) - x_{k+1}  (n*N*(T-1))
        for k in range(1, self.T):
            context_i_k = cas.reshape(context[:, k], n_c, N)
            for i in range(N):
                i_onehot = eye[:, i]
                xk = cas.reshape(x[:, k-1], n, N)
                xk1 = cas.reshape(x[:, k], n, N)
                uk = cas.reshape(u[:, k], m, N)
                fk = self.game.f(xk[:, i], uk[:, i], i_onehot, context_i_k[:, i]) - xk1[:, i]
                assert fk.shape == (n, 1)
                r_vec.append(fk)  # n

        r_vec.append(cas.vec(h_val))  # n_hi * N
        r_val = cas.vertcat(*r_vec)
        assert r_val.shape == (n*N*T+m*N*T+n*N*T+n_hi*N, 1)
        assert h_val.shape == (n_hi, N)
        return r_val, h_val

    def check_Ki(self, Ki):
        """ Verify agent KKT matrix is in following format, return component H, A
        [H A.T
         A 0 ]
        Args:
            Ki: scipy.sparse.csc_matrix
        Returns:
            H, A
        """
        tol = 1e-2
        T = self.T
        n = self.n
        m = self.m
        primal_n = (n+m)*T  # primal variables for agent i
        H = Ki[:primal_n, :primal_n]
        A = Ki[primal_n:, :primal_n]
        AT = Ki[:primal_n, primal_n:]

        empty = Ki[primal_n:, primal_n:]
        assert scipy.sparse.linalg.norm(empty) < tol
        assert empty.nnz == 0

        assert np.sum(np.abs((H-H.T).data)) < tol
        dyn_diff = (A-AT.T).toarray()[:n*T, :]
        assert np.sum(np.abs(dyn_diff)) < tol

        h_diff = (A-AT.T).toarray()[n*T:, :]
        assert np.sum(np.abs(h_diff)) < tol

        return H, A

    def check_KKT(self, Ki):
        """ Verify KKT matrix is in following format, return component H, A
        [H A.T
         A 0 ]
        Args:
            Ki: scipy.sparse.csc_matrix
        Returns:
            H, A
        NOTE does not apply to current formulation
        """
        T = self.T
        N = self.N
        n = self.n
        m = self.m
        primal_n = (n+m)*N*T  # primal variables for the game
        H = Ki[:primal_n, :primal_n]
        A = Ki[primal_n:, :primal_n]
        AT = Ki[:primal_n, primal_n:]

        empty = Ki[primal_n:, primal_n:]
        assert scipy.sparse.linalg.norm(empty) < 1e-10
        assert empty.nnz == 0

        assert np.sum(np.abs((H-H.T).data)) < 1e-10
        dyn_diff = (A-AT.T).toarray()[:n*N*T, :]
        assert np.sum(np.abs(dyn_diff)) < 1e-10

        # won't pass by construction
        # e.g. dLLi/dxi does not depend on mu_j, but mu_j * h(xj,xi) depends on xi
        # h_diff = (A-AT.T).toarray()[n*N*T:, :]
        # assert np.sum(np.abs(h_diff)) < 1e-10

        # assert np.sum(np.abs((A-AT.T).data)) < 1e-10
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
        n_hi = self.n_hi
        m = self.m
        offset = 0
        r_Lx = norm(r[n*N*T, 0])
        offset += n*N*T
        r_Lu = norm(r[offset:offset+m*N*T])
        offset += m*N*T
        r_f = norm(r[offset:offset+n*N*T])
        offset += n*N*T
        r_h = norm(r[offset:offset+n_hi*N])
        offset += n_hi*N
        assert r.shape == (offset, 1)
        return r_Lx, r_Lu, r_f, r_h

    def dr_to_dri(self, dr):
        """ Slice dr to a vector of dri. Also applies to dy -> dyi
        Args:
            dr: (dim) column vector
        Returns:
            dri_vec: list[ (dim)]"""
        N = self.N
        T = self.T
        n = self.n
        m = self.m
        n_hi = self.n_hi
        dri_vec = []
        xi_size = n*T
        ui_size = m*T
        li_size = n*T
        mi_size = n_hi
        dri_vec = []
        dri_size = n*T + m*T + n*T + n_hi
        for i in range(N):
            dri = np.zeros(dri_size) + np.nan
            offset = 0
            offset_i = 0
            # x
            dri[:xi_size] = dr[i*xi_size: (i+1)*xi_size]
            offset += xi_size*N
            offset_i += xi_size
            # u
            dri[offset_i:offset_i+ui_size] = dr[offset + i*ui_size: offset + (i+1)*ui_size]
            offset += ui_size*N
            offset_i += ui_size
            # lamda
            dri[offset_i:offset_i+li_size] = dr[offset + i*li_size: offset + (i+1)*li_size]
            offset += li_size*N
            offset_i += li_size
            # mu
            dri[offset_i:offset_i+mi_size] = dr[offset + i*mi_size: offset + (i+1)*mi_size]
            offset += mi_size*N
            offset_i += mi_size
            assert offset == dr.shape[0]
            assert offset_i == dri.shape[0]
            assert not np.any(np.isnan(dri))
            dri_vec.append(dri)
        return dri_vec

    def dri_to_dr(self, dri_vec):
        """ Construct dr from dri, also works on dy -> dy_i
        Args:
            dr: (dim) column vector
        Returns:
            dri_vec: list[ (dim)]"""
        N = self.N
        T = self.T
        n = self.n
        m = self.m
        n_hi = self.n_hi
        dr_size = n*N*T + m*N*T + n*N*T + n_hi*N
        dr = np.zeros(dr_size) + np.nan
        for i in range(N):
            dri = dri_vec[i]
            offset = 0
            offset_i = 0
            # x
            xi_size = n*T
            dr[i*xi_size: (i+1)*xi_size] = dri[:xi_size]
            offset += xi_size*N
            offset_i += xi_size
            # u
            ui_size = m*T
            dr[offset + i*ui_size: offset +
                (i+1)*ui_size] = dri[offset_i:offset_i+ui_size]
            offset += ui_size*N
            offset_i += ui_size
            # lamda
            li_size = n*T
            dr[offset + i*li_size: offset +
                (i+1)*li_size] = dri[offset_i:offset_i+li_size]
            offset += li_size*N
            offset_i += li_size
            # mu
            mi_size = n_hi
            dr[offset + i*mi_size: offset +
                (i+1)*mi_size] = dri[offset_i:offset_i+mi_size]
            offset += mi_size*N
            offset_i += mi_size
            assert offset == dr_size
            assert offset_i == dri.shape[0]
        assert not np.any(np.isnan(dr))
        return dr

    def r_idx_str(self, idx):
        """ Given an r index, print its name"""
        N = self.N
        n = self.n
        m = self.m
        T = self.T
        n_hi = self.n_hi
        if idx < n*N*T:
            Ti = idx // (n*N)
            Ni = (idx - Ti*n*N) // n
            ni = idx - Ti*n*N - Ni*n
            return f'dLLi/dxi {Ti=}, {Ni=}, {ni=}'
        idx -= n*N*T

        if idx < m*N*T:
            Ti = idx // (m*N)
            Ni = (idx - Ti*m*N) // m
            mi = idx - Ti*m*N - Ni*m
            return f'dLLi/dui {Ti=}, {Ni=}, {mi=}'
        idx -= m*N*T

        if idx < n*N*T:
            Ti = idx // (n*N)
            Ni = (idx - Ti*n*N) // n
            ni = idx - Ti*n*N - Ni*n
            return f'f(x,u) {Ti=}, {Ni=}, {ni=}'
        idx -= n*N*T

        if idx < n_hi*N:  # n_hi = N*T (in that order)
            Ni = idx // n_hi
            Ti = (idx - n_hi*Ni) // N
            Nj = idx - n_hi*Ni - N*Ti
            return f'h(x,u) {Ni=}, {Nj=}, {Ti=}'

    def y_idx_str(self, idx):
        """ Given an y index, print its name"""
        N = self.N
        n = self.n
        m = self.m
        T = self.T
        n_hi = self.n_hi
        if idx < n*N*T:
            Ti = idx // (n*N)
            Ni = (idx - Ti*n*N) // n
            ni = idx - Ti*n*N - Ni*n
            return f'x {Ti=}, {Ni=}, {ni=}'
        idx -= n*N*T

        if idx < m*N*T:
            Ti = idx // (m*N)
            Ni = (idx - Ti*m*N) // m
            mi = idx - Ti*m*N - Ni*m
            return f'u {Ti=}, {Ni=}, {mi=}'
        idx -= m*N*T

        if idx < n*N*T:
            Ti = idx // (n*N)
            Ni = (idx - Ti*n*N) // n
            ni = idx - Ti*n*N - Ni*n
            return f'lambda for f(x,u) {idx=}, {Ti=}, {Ni=}, {ni=}'
        idx -= n*N*T

        if idx < n_hi*N:  # n_hi = N*T (in that order)
            Ni = idx // n_hi
            Ti = (idx - n_hi*Ni) // N
            Nj = idx - n_hi*Ni - N*Ti
            return f'mu for h(x,u) {idx=} {Ni=}, {Nj=}, {Ti=}'

    def make_full_KKT_reg(self, reg_vec):
        """ Given a list of regularization value for each agent,
        create a regularizaiton matrix for full game KKT"""
        N = self.N
        n = self.n
        m = self.m
        T = self.T
        n_hi = self.n_hi
        l = n*N*T + m*N*T + n*N*T + n_hi*N
        data = []
        row = []
        for i in range(self.N):
            for k in range(self.T):
                for idx in range(self.n):
                    data.append(reg_vec[i])
                    row.append(k*(n*N) + i*n + idx)
                for idx in range(self.m):
                    data.append(reg_vec[i])
                    row.append(n*N*T + k*(m*N) + i*m + idx)
        reg_mtx = csc_matrix((data, (row, row)), shape=(l, l))
        return reg_mtx

    def get_inertia(self, mtx):
        """ Given matrix mtx, give inertia (pos,neg,zero), and qdldl solver instance"""
        upper = scipy.sparse.triu(mtx, format='csc')
        upper.eliminate_zeros()
        upper.sort_indices()
        upper.sum_duplicates()
        # pylint:disable-next=c-extension-no-member
        solver = qdldl.Solver(upper, upper=True)

        #  C = P @ A @ P.T, C = L @ D @ L.T
        # L_zero_diag, D_diag, P_vec = solver.factors()
        _, D_diag, _ = solver.factors()
        pos = np.sum(D_diag > 0)
        neg = np.sum(D_diag < 0)
        zero = len(D_diag) - pos - neg
        inertia = (pos, neg, zero)
        return inertia, solver
