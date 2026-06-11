"""CasADi Newton rootfinder for the game KKT residual."""

import logging
from dataclasses import dataclass, field
from itertools import accumulate
from time import time
from typing import Optional

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
    'NewtonCasadi',
    'NewtonCasadiConfig',
]


@dataclass(frozen=True)
class NewtonCasadiConfig(InteriorPointGameConfig):
    """Configuration for the CasADi Newton rootfinder baseline."""
    initial_tau: float = 0.1
    """Initial perturbed-complementarity value."""
    tau_min: float = 1e-8
    """Smallest perturbed-complementarity value used by homotopy."""
    homotopy_iterations: int = 5
    """Number of Newton rootfinder calls while shrinking tau."""
    min_slack: float = 1e-2
    """Minimum initial slack for inactive inequalities."""
    initial_mu: Optional[float] = None
    """Initial multiplier value; if None, use initial_tau / slack."""
    abstol_step: float = 1e-10
    """CasADi Newton stopping tolerance for the rootfinder step."""
    line_search: bool = True
    """Whether CasADi Newton should use its internal line search."""
    print_iteration: bool = False
    """Whether CasADi Newton should print one line per rootfinder iteration."""
    print_time: bool = False
    """Whether CasADi should print rootfinder timing."""
    error_on_fail: bool = False
    """Whether CasADi should raise when the Newton rootfinder fails."""
    rootfinder_options: dict = field(default_factory=dict)
    """Additional options forwarded to CasADi rootfinder."""


class NewtonCasadi(InteriorPointGame):
    """Solve the inherited game KKT residual with CasADi's Newton rootfinder.

    The rootfinder solves an interior-point residual for
    ``z = vec(x, u, lambda, mu, s)``:

        stationarity = 0, dynamics = 0, h + s = 0, mu * s - tau = 0.

    CasADi forms and factors the residual Jacobian inside the ``newton``
    plugin.
    """

    def __init__(self, config: NewtonCasadiConfig, game, cpp_only=False):
        if cpp_only:
            raise NotImplementedError('NewtonCasadi does not support a C++ backend')
        super().__init__(config, game, cpp_only=cpp_only)
        self.newton_stats = {}

    def construct_casadi_fun(self):
        """Construct inherited CasADi functions and the Newton rootfinder."""
        super().construct_casadi_fun()
        self._construct_newton_rootfinder()

    def _construct_newton_rootfinder(self):
        """Build a flattened interior-point residual for CasADi rootfinder."""
        gc = self.game.config
        n = gc.n
        m = gc.m
        N = gc.N
        T = gc.T
        nNT = n * N * T
        mNT = m * N * T
        sizes = [0, nNT, mNT, nNT, self.dual_h_dim, self.dual_h_dim]
        offsets = list(accumulate(sizes))

        z = cas.SX.sym('z', offsets[-1], 1)
        tau = cas.SX.sym('tau')
        config_params = [gc.get_int_param_sx(), gc.get_double_param_sx()]

        x = cas.reshape(z[offsets[0]:offsets[1]], n * N, T)
        u = cas.reshape(z[offsets[1]:offsets[2]], m * N, T)
        lamda = cas.reshape(z[offsets[2]:offsets[3]], n * N, T)
        mu = cas.reshape(z[offsets[3]:offsets[4]], self.dual_h_dim, 1)
        slack = cas.reshape(z[offsets[4]:offsets[5]], self.dual_h_dim, 1)
        context = self.get_full_context_casadi(x)
        r_val, h_val = self.r(x, u, lamda, mu, context)
        h_offset = nNT + mNT + nNT
        h_residual = cas.reshape(h_val, self.dual_h_dim, 1) + slack
        comp_residual = mu * slack - tau
        interior_point_r = cas.vertcat(r_val[:h_offset], h_residual, comp_residual)
        dr_dz = cas.jacobian(interior_point_r, z)

        self.newton_residual_casadi = cas.Function(
            'newton_residual',
            [z, tau] + config_params,
            [interior_point_r],
        )
        self.newton_eval_casadi = cas.Function(
            'newton_eval',
            [z, tau] + config_params,
            [interior_point_r, h_val, comp_residual, dr_dz],
        )
        self.newton_rootfinder = cas.rootfinder(
            'newton_casadi_solver',
            'newton',
            self.newton_residual_casadi,
            self._rootfinder_options(),
        )

    def _rootfinder_options(self):
        options = {
            'abstol': self.config.tolerance,
            'abstolStep': self.config.abstol_step,
            'line_search': self.config.line_search,
            'max_iter': self.config.iterations,
            'print_iteration': self.config.print_iteration,
            'print_time': self.config.print_time,
            'error_on_fail': self.config.error_on_fail,
        }
        options.update(self.config.rootfinder_options)
        return options

    def _offsets(self):
        gc = self.game.config
        nNT = gc.n * gc.N * gc.T
        mNT = gc.m * gc.N * gc.T
        return list(accumulate([0, nNT, mNT, nNT, self.dual_h_dim, self.dual_h_dim]))

    def _initial_guess(self, u_ref, tau):
        gc = self.game.config
        N = gc.N
        T = gc.T
        n = gc.n
        m = gc.m

        if u_ref is None:
            u = np.zeros((m * N, T), order='F')
        else:
            u = np.array(u_ref, dtype=float, order='F', copy=True)
            if u.shape == (m, N, T):
                u = u.reshape((m * N, T), order='F')
            assert u.shape == (m * N, T)

        params_np = [gc.get_int_param_np(), gc.get_double_param_np()]
        x = self.rollout_casadi(gc.x0, u, *params_np)
        context = self.get_full_context_casadi(x)
        h_val = self.h_casadi(x, u, context, *params_np)
        h_np = np.asarray(h_val).reshape(self.dual_h_dim, order='F')
        slack = np.maximum(self.config.min_slack, -h_np).reshape(
            (self.dual_h_dim, 1), order='F')
        lamda = np.zeros((n * N, T), order='F')
        if self.config.initial_mu is None:
            mu = tau / slack
        else:
            mu = np.full((self.dual_h_dim, 1), self.config.initial_mu, order='F')

        return np.concatenate([
            np.asarray(x).reshape(-1, order='F'),
            u.reshape(-1, order='F'),
            lamda.reshape(-1, order='F'),
            mu.reshape(-1, order='F'),
            slack.reshape(-1, order='F'),
        ])

    def _unpack_solution(self, y):
        gc = self.game.config
        offsets = self._offsets()
        y = np.asarray(y).reshape(-1, order='F')

        x = y[offsets[0]:offsets[1]].reshape((gc.n * gc.N, gc.T), order='F')
        u = y[offsets[1]:offsets[2]].reshape((gc.m * gc.N, gc.T), order='F')
        lamda = y[offsets[2]:offsets[3]].reshape((gc.n * gc.N, gc.T), order='F')
        mu = y[offsets[3]:offsets[4]].reshape((self.dual_h_dim, 1), order='F')
        slack = y[offsets[4]:offsets[5]].reshape((self.dual_h_dim, 1), order='F')
        return x, u, lamda, mu, slack

    def solve(self, u_ref=None):
        """Solve the interior-point KKT residual with CasADi Newton."""
        gc = self.game.config
        params_np = [gc.get_int_param_np(), gc.get_double_param_np()]
        tau = self.config.initial_tau
        z0 = self._initial_guess(u_ref, tau)

        r0, _, _, _ = self.newton_eval_casadi(z0, tau, *params_np)
        initial_residual = float(np.linalg.norm(np.asarray(r0)))
        residual = initial_residual
        z_sol = z0
        dr_dz = None
        h_val = None
        comp_residual = None

        t0 = time()
        rootfinder_failed = False
        used_fallback = False
        self.residual_vec = [initial_residual]
        stats = {}
        total_iterations = 0
        max_homotopy_iterations = max(1, self.config.homotopy_iterations)
        solved_tau = tau
        for _ in range(max_homotopy_iterations):
            try:
                raw_z_sol = self.newton_rootfinder(z_sol, tau, *params_np)
            except RuntimeError as exc:
                if self.config.error_on_fail:
                    raise
                logger.warning('Newton rootfinder failed: %s', exc)
                rootfinder_failed = True
                used_fallback = True
                break

            stats = self.newton_rootfinder.stats()
            total_iterations += stats.get('iter_count', 0)
            if not np.all(np.isfinite(np.asarray(raw_z_sol, dtype=float))):
                logger.warning(
                    'Newton returned non-finite values; keeping the previous iterate')
                used_fallback = True
                break

            r_val, h_val, comp_residual, dr_dz = self.newton_eval_casadi(
                raw_z_sol, tau, *params_np)
            stage_residual = float(np.linalg.norm(np.asarray(r_val)))
            if not np.isfinite(stage_residual):
                logger.warning(
                    'Newton residual is non-finite; keeping the previous iterate')
                used_fallback = True
                break

            z_sol = raw_z_sol
            residual = stage_residual
            solved_tau = tau
            self.residual_vec.append(residual)
            if tau <= self.config.tau_min:
                break
            tau = max(self.config.tau_min, self.config.tau_decay * tau)
        dt = time() - t0
        tau = solved_tau

        self.newton_stats = stats
        if h_val is None or comp_residual is None or dr_dz is None:
            r_val, h_val, comp_residual, dr_dz = self.newton_eval_casadi(
                z_sol, tau, *params_np)
            residual = float(np.linalg.norm(np.asarray(r_val)))
            if len(self.residual_vec) == 0 or self.residual_vec[-1] != residual:
                self.residual_vec.append(residual)

        x, u, lamda, mu, slack = self._unpack_solution(z_sol)
        del lamda

        has_converged = residual <= self.config.tolerance
        is_optimal = (
            bool(stats.get('success', False))
            and has_converged
            and not used_fallback
            and not rootfinder_failed
        )

        h_np = np.asarray(h_val).reshape(-1)
        mu_np = np.asarray(mu).reshape(-1)
        slack_np = np.asarray(slack).reshape(-1)
        comp_np = np.asarray(comp_residual).reshape(-1)
        logger.info(
            'Newton status=%s, residual=%.6g, initial_residual=%.6g, '
            'tau=%.6g, max_h=%.6g, min_mu=%.6g, min_s=%.6g, '
            'comp_norm=%.6g, nnz_dr_dz=%d',
            stats.get('return_status', ''),
            residual,
            initial_residual,
            tau,
            np.max(h_np) if h_np.size else 0.0,
            np.min(mu_np) if mu_np.size else 0.0,
            np.min(slack_np) if slack_np.size else 0.0,
            np.linalg.norm(comp_np) if comp_np.size else 0.0,
            dr_dz.nnz(),
        )

        return Solution(
            elapsed_time=dt,
            iterations=total_iterations,
            u=as_numpy_array(u).reshape((gc.m, gc.N, gc.T), order='F'),
            x=as_numpy_array(x).reshape((gc.n, gc.N, gc.T), order='F'),
            residual=residual,
            has_converged=has_converged,
            is_optimal=is_optimal,
        )

    def final(self):
        """Plot the initial and final Newton residuals."""
        self.profiler.summary()
        if len(self.residual_vec) == 0:
            logger.warning('No Newton residual history available to plot')
            return

        _, ax = plt.subplots()
        ax.plot(self.residual_vec, '*-', label='||r||_2')
        ax.set_yscale('log')
        ax.set_xlabel('Newton solve')
        ax.set_ylabel('Residual')
        ax.legend()
        plt.show()
