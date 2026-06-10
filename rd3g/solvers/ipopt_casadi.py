"""IPOPT-based residual minimization solver for generalized Nash games."""

import logging
from dataclasses import dataclass, field
from itertools import accumulate
from time import time

import casadi as cas
import matplotlib.pyplot as plt
import numpy as np

from rd3g.core.base_solver import Solution
from rd3g.solvers.interior_point_game import (
    InteriorPointGame,
    InteriorPointGameConfig,
    as_numpy_array,
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

__all__ = [
    'IpoptCasadi',
    'IpoptCasadiConfig',
    'IPOPTCasadi',
    'IPOPTCasadiConfig',
]


@dataclass(frozen=True)
class IpoptCasadiConfig(InteriorPointGameConfig):
    """Configuration for the IPOPT residual-minimization baseline."""
    iterations: int = 200
    """Maximum IPOPT iterations before giving up."""
    initial_mu: float = 1.0
    """Initial multiplier value for inequality constraints."""
    mu_lower_bound: float = 0.0
    """Lower bound for inequality multipliers."""
    h_upper_bound: float = 0.0
    """Upper bound for game inequalities h(x,u) <= h_upper_bound."""
    constraint_tolerance: float = 1e-6
    """Tolerance used when checking returned constraint feasibility."""
    print_time: bool = False
    """Whether CasADi should print solver timing."""
    ipopt_print_level: int = 0
    """IPOPT print_level option."""
    ipopt_options: dict = field(default_factory=dict)
    """Additional options forwarded under the IPOPT option namespace."""


class IpoptCasadi(InteriorPointGame):
    """Solve the GNE residual least-squares problem with IPOPT.

    The NLP is

        minimize_y ||r(y)||_2^2
        subject to mu >= 0, h(x,u) <= 0,

    where ``r`` is inherited from :class:`InteriorPointGame`.
    """

    def __init__(self, config: IpoptCasadiConfig, game, cpp_only=False):
        if cpp_only:
            raise NotImplementedError('IpoptCasadi does not support a C++ backend')
        super().__init__(config, game, cpp_only=cpp_only)
        self.ipopt_iteration_stats = {}

    def construct_casadi_fun(self):
        """Construct CasADi functions and the IPOPT NLP."""
        gc = self.game.config
        N = gc.N
        n = gc.n
        m = gc.m
        T = gc.T

        logger.info('Constructing IPOPT CasADi NLP... ')

        x = cas.SX.sym('x', n * N, T)
        u = cas.SX.sym('u', m * N, T)
        lamda = cas.SX.sym('lamda', n * N, T)
        mu = cas.SX.sym('mu', self.dual_h_dim, 1)
        x0 = cas.SX.sym('x0', n, N)
        config_params = [gc.get_int_param_sx(), gc.get_double_param_sx()]

        context = self.game.context.get_context_sx()
        args = [x, u, lamda, mu, context]

        self.get_n_fun = cas.Function('get_n', [], [self.n])
        self.get_m_fun = cas.Function('get_m', [], [self.m])

        h_val = self.game.h(x, u, context)
        self.h_casadi = cas.Function('h', [x, u, context] + config_params, [h_val])

        r_val, h_val = self.r(*args)
        self.r_casadi = cas.Function('r', args + config_params, [r_val, h_val])

        X = self.game.rollout(x0, u)
        self.rollout_casadi = cas.Function('rollout', [x0, u] + config_params, [X])

        xki = cas.SX.sym('xki', n, 1)
        xkj = cas.SX.sym('xkj', n, 1)
        col_h_val = self.game.collision_h(xki, xkj)
        self.collision_h_casadi = cas.Function(
            'collision_h', [xki, xkj] + config_params, [col_h_val])

        self._construct_context_function(x)

        nlp_context = self._full_context_expr(x)
        nlp_r, nlp_h = self.r(x, u, lamda, mu, nlp_context)
        objective = cas.dot(nlp_r, nlp_r)
        constraints = cas.vertcat(mu, cas.vec(nlp_h))
        y = cas.vertcat(*[cas.vec(val) for val in [x, u, lamda, mu]])
        params = cas.vertcat(*config_params)

        self.ipopt_residual_casadi = cas.Function(
            'ipopt_residual',
            [x, u, lamda, mu] + config_params,
            [nlp_r, nlp_h, objective],
        )
        self.nlp_variables = y
        self.nlp_constraints = constraints
        self.nlp_solver = cas.nlpsol(
            'ipopt_casadi_solver',
            'ipopt',
            {'x': y, 'f': objective, 'g': constraints, 'p': params},
            self._ipopt_options(),
        )

        logger.info('Constructing IPOPT CasADi NLP... Done')

    def _construct_context_function(self, x):
        """Build get_full_context_casadi for a trajectory x."""
        gc = self.game.config
        n = gc.n
        N = gc.N
        T = gc.T

        xki = cas.SX.sym('xki', n, 1)
        context_val = self.game.get_context(xki)
        self.get_context_casadi = cas.Function('get_context', [xki], [context_val])

        get_context_n_N = self.get_context_casadi.map(N)

        def get_context_nN(xnN):
            x_n_N = cas.reshape(xnN, n, N)
            return cas.reshape(get_context_n_N(x_n_N), self.n_c * self.N, 1)

        xnN = cas.SX.sym('xnN', n * N, 1)
        get_context_nN_casadi = cas.Function('get_context_nN', [xnN], [get_context_nN(xnN)])
        get_context_nN_T = get_context_nN_casadi.map(T)
        full_context_val = get_context_nN_T(x)
        self.get_full_context_casadi = cas.Function('get_full_context', [x], [full_context_val])

    def _full_context_expr(self, x):
        """Return the symbolic full-context expression for x."""
        return self.get_full_context_casadi(x)

    def _ipopt_options(self):
        ipopt_options = {
            'tol': self.config.tolerance,
            'acceptable_tol': self.config.tolerance,
            'constr_viol_tol': self.config.constraint_tolerance,
            'max_iter': self.config.iterations,
            'print_level': self.config.ipopt_print_level,
        }
        ipopt_options.update(self.config.ipopt_options)
        return {
            'print_time': self.config.print_time,
            'ipopt': ipopt_options,
        }

    def _constraint_bounds(self):
        mu_dim = self.dual_h_dim
        h_dim = self.dual_h_dim
        lbg = np.concatenate([
            np.full(mu_dim, self.config.mu_lower_bound),
            np.full(h_dim, -np.inf),
        ])
        ubg = np.concatenate([
            np.full(mu_dim, np.inf),
            np.full(h_dim, self.config.h_upper_bound),
        ])
        return lbg, ubg

    def _variable_bounds(self):
        var_count = self.nlp_variables.shape[0]
        return np.full(var_count, -np.inf), np.full(var_count, np.inf)

    def _initial_guess(self, u_ref):
        gc = self.game.config
        N = gc.N
        T = gc.T
        n = gc.n
        m = gc.m

        if u_ref is None:
            u = np.zeros((m * N, T), order='F')
        else:
            u = np.asarray(u_ref, order='F')
            if u.shape == (m, N, T):
                u = u.reshape((m * N, T), order='F')
            assert u.shape == (m * N, T)

        params_np = [gc.get_int_param_np(), gc.get_double_param_np()]
        x = self.rollout_casadi(gc.x0, u, *params_np)
        lamda = cas.DM.zeros((n * N, T))
        mu = cas.DM.ones((self.dual_h_dim, 1)) * self.config.initial_mu
        y0 = np.concatenate([
            np.asarray(x).reshape(-1, order='F'),
            np.asarray(u).reshape(-1, order='F'),
            np.asarray(lamda).reshape(-1, order='F'),
            np.asarray(mu).reshape(-1, order='F'),
        ])
        return y0

    def _unpack_solution(self, y):
        gc = self.game.config
        N = gc.N
        T = gc.T
        n = gc.n
        m = gc.m
        nNT = n * N * T
        mNT = m * N * T
        sizes = [0, nNT, mNT, nNT, self.dual_h_dim]
        offsets = list(accumulate(sizes))

        y = np.asarray(y).reshape(-1, order='F')
        x = y[offsets[0]:offsets[1]].reshape((n * N, T), order='F')
        u = y[offsets[1]:offsets[2]].reshape((m * N, T), order='F')
        lamda = y[offsets[2]:offsets[3]].reshape((n * N, T), order='F')
        mu = y[offsets[3]:offsets[4]].reshape((self.dual_h_dim, 1), order='F')
        return x, u, lamda, mu

    def _residual_history_from_stats(self, fallback_residual=None):
        """Return ||r||_2 per IPOPT iteration from the squared-residual objective."""
        obj_history = self.ipopt_iteration_stats.get('obj', [])
        if len(obj_history) == 0:
            return [] if fallback_residual is None else [fallback_residual]
        obj_history = np.asarray(obj_history, dtype=float)
        return np.sqrt(np.clip(obj_history, a_min=0.0, a_max=None)).tolist()

    def solve(self, u_ref=None):
        """Solve the residual least-squares NLP with IPOPT."""
        gc = self.game.config
        params_np = [gc.get_int_param_np(), gc.get_double_param_np()]
        params = np.concatenate(params_np)
        lbg, ubg = self._constraint_bounds()
        lbx, ubx = self._variable_bounds()
        y0 = self._initial_guess(u_ref)

        t0 = time()
        result = self.nlp_solver(
            x0=y0,
            p=params,
            lbx=lbx,
            ubx=ubx,
            lbg=lbg,
            ubg=ubg,
        )
        dt = time() - t0

        x, u, lamda, mu = self._unpack_solution(result['x'])
        r_val, h_val, objective = self.ipopt_residual_casadi(
            x, u, lamda, mu, *params_np)
        residual = float(np.linalg.norm(np.asarray(r_val)))
        objective = float(objective)
        self.residual_vec.append(residual)

        stats = self.nlp_solver.stats()
        self.ipopt_iteration_stats = stats.get('iterations', {})
        self.residual_vec = self._residual_history_from_stats(fallback_residual=residual)
        h_np = np.asarray(h_val).reshape(-1)
        mu_np = np.asarray(mu).reshape(-1)
        h_feasible = h_np.size == 0 or np.max(h_np) <= self.config.constraint_tolerance
        mu_feasible = (
            mu_np.size == 0 or
            np.min(mu_np) >= self.config.mu_lower_bound - self.config.constraint_tolerance
        )
        has_converged = residual <= self.config.tolerance and h_feasible and mu_feasible
        is_optimal = bool(stats.get('success', False))

        logger.info(
            'IPOPT status=%s, objective=%.6g, residual=%.6g, h_feasible=%s, mu_feasible=%s',
            stats.get('return_status', ''),
            objective,
            residual,
            h_feasible,
            mu_feasible,
        )

        return Solution(
            elapsed_time=dt,
            iterations=stats.get('iter_count', 0),
            u=as_numpy_array(u).reshape((gc.m, gc.N, gc.T), order='F'),
            x=as_numpy_array(x).reshape((gc.n, gc.N, gc.T), order='F'),
            residual=residual,
            has_converged=has_converged,
            is_optimal=is_optimal,
        )

    def final(self):
        """Plot IPOPT residual history from the last solve."""
        self.profiler.summary()
        residual_history = self._residual_history_from_stats()
        if len(residual_history) == 0:
            residual_history = self.residual_vec
        if len(residual_history) == 0:
            logger.warning('No IPOPT residual history available to plot')
            return

        _, ax = plt.subplots()
        ax.plot(residual_history, '*-', label='||r||_2')
        ax.set_yscale('log')
        ax.set_xlabel('IPOPT iteration')
        ax.set_ylabel('Residual')
        ax.legend()
        plt.show()


IPOPTCasadiConfig = IpoptCasadiConfig
IPOPTCasadi = IpoptCasadi
