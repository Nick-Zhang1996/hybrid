"""Iterative linear-quadratic game solver built on CasADi derivatives."""

import logging
from dataclasses import dataclass
from time import time

import casadi as cas
import matplotlib.pyplot as plt
import numpy as np

from rd3g.core.base_solver import BaseSolver, BaseSolverConfig, Solution
from rd3g.utilities.time_util import TimeUtil


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ExecutionTimer = TimeUtil

__all__ = [
    'ILQGame',
    'ILQGameConfig',
    'ILQGameSolver',
    'ILQPolicy',
]


@dataclass(frozen=True)
class ILQGameConfig(BaseSolverConfig):
    """Configuration for :class:`ILQGame`."""
    tolerance: float = 5e-4
    """Stop when feedforward update, control update, and violation are below this."""
    iterations: int = 50
    """Maximum iLQ-game iterations."""
    step_size: float = 0.05
    """How far to move along the new LQ feedforward direction."""
    feedback_scale: float = 1.0
    """Scale applied to the LQ feedback gain during policy rollout."""
    lq_regularization: float = 1e-6
    """Diagonal regularization added to each LQ stationary linear system."""
    state_regularization: float = 1e-8
    """Diagonal regularization added to each state Hessian."""
    control_regularization: float = 1e-6
    """Diagonal regularization added to each player's own control Hessian."""
    barrier_weight: float = 1e-2
    """Initial relaxed log-barrier weight for h(x, u) <= 0 constraints."""
    barrier_growth: float = 5.0
    """Multiplier applied to the barrier weight when constraints are violated."""
    barrier_max: float = 1e6
    """Maximum barrier weight."""
    barrier_relaxation: float = 1e-2
    """Relaxed log-barrier transition distance from the constraint boundary."""
    constraint_tolerance: float = 1e-5
    """Allowed positive h(x, u) before increasing the barrier."""
    variational_gne: bool = True
    """If True, the game exposes one shared canonical constraint column."""
    use_line_search: bool = True
    """Backtrack the feedforward step when rollout quality degrades."""
    line_search_decay: float = 0.5
    """Step shrink factor used by the rollout line search."""
    line_search_max_iter: int = 8
    """Maximum rollout line-search trials."""
    max_control_step_norm: float = 1e3
    """Reject rollout trials with very large control motion."""
    jit: bool = True
    """JIT compile optional CasADi helper functions when possible."""


@dataclass
class ILQPolicy:
    """Time-indexed affine feedback policy in deviation coordinates."""
    P: object
    """Stacked feedback gains with shape (m*N, n*N*T)."""
    v: object
    """Stacked feedforward controls with shape (m*N, T)."""
    feedforward_norm: float
    """Root-mean-square feedforward control norm."""


class ILQGame(BaseSolver):
    """Iterative linear-quadratic approximation for general-sum games.

    At each iteration this solver rolls out a reference trajectory, uses
    CasADi to linearize dynamics and quadraticize each player's local cost,
    solves the resulting finite-horizon LQ game by dynamic programming, and
    rolls out the affine feedback policy

        u_i = u_ref_i - P_i (x - x_ref) + eta v_i.

    Inequality constraints h(x, u) <= 0 are added to each local quadratic cost
    with a relaxed log barrier. The barrier weight grows when the current
    iterate violates constraints.
    """

    def __init__(self, config: ILQGameConfig, game, cpp_only=False):
        if cpp_only:
            raise NotImplementedError('ILQGame does not support a C++ backend')
        BaseSolver.__init__(self, config, game)

        gc = self.game.config
        self.N = gc.N
        self.T = gc.T
        self.dt = gc.dt
        self.n = gc.n
        self.m = gc.m
        self.n_h = getattr(gc, 'n_h', getattr(gc, 'n_hi', 0))
        self.n_c = gc.n_c
        assert gc.variational_gne == config.variational_gne, (
            'Game and Solver must have the same variational_gne setting'
        )
        self.variational_gne = config.variational_gne
        self.h_multiplier_cols = 1 if self.variational_gne else self.N
        self.n_h_stage = self.n_h // self.T if self.T > 0 else 0
        if self.n_h != self.n_h_stage * self.T:
            raise ValueError(
                'ILQGame expects inequality constraints to be grouped by stage, '
                f'but n_h={self.n_h} is not divisible by T={self.T}.'
            )

        self.profiler = ExecutionTimer(True)
        self.residual_vec = []
        self.violation_vec = []
        self.barrier_weight = float(config.barrier_weight)
        self.last_policy = None
        self._casadi_primed = False
        self._casadi_jit_enabled = bool(config.jit)
        self.validate()
        self.construct_casadi_fun()
        self.prime_casadi()

    def validate(self):
        """Check dimensions supplied by the game."""
        assert isinstance(self.n, int) and self.n > 0
        assert isinstance(self.m, int) and self.m > 0
        assert isinstance(self.T, int) and self.T > 0
        assert isinstance(self.N, int) and self.N > 0
        assert self.game.config.x0.shape == (self.n, self.N)

    def construct_casadi_fun(self):
        """Construct CasADi derivative and dense LQ helper functions."""
        gc = self.game.config
        n = self.n
        m = self.m
        N = self.N
        T = self.T
        n_c = self.n_c
        config_params = [gc.get_int_param_sx(), gc.get_double_param_sx()]

        logger.info('Constructing ILQGame CasADi functions... ')

        x0 = cas.SX.sym('x0', n, N)
        u_traj = cas.SX.sym('u', m * N, T)
        X = self.game.rollout(x0, u_traj)
        self.rollout_casadi = cas.Function('ilq_rollout', [x0, u_traj] + config_params, [X])

        x_stage = cas.SX.sym('x_stage', n * N, 1)
        u_stage = cas.SX.sym('u_stage', m * N, 1)
        context_stage = cas.SX.sym('context_stage', n_c * N, 1)

        f_val = self._stage_dynamics_expr(x_stage, u_stage, context_stage)
        A_val = cas.jacobian(f_val, x_stage)
        B_val = cas.jacobian(f_val, u_stage)
        self.stage_dynamics_casadi = cas.Function(
            'ilq_stage_dynamics',
            [x_stage, u_stage, context_stage] + config_params,
            [f_val, A_val, B_val],
        )

        self.get_context_stage_casadi = self._construct_context_stage_function()

        h_x = cas.SX.sym('x', n * N, T)
        h_u = cas.SX.sym('u', m * N, T)
        h_context = cas.SX.sym('context', n_c * N, T)
        h_val = self.game.h(h_x, h_u, h_context)
        self.h_casadi = cas.Function('ilq_h', [h_x, h_u, h_context] + config_params, [h_val])

        xki = cas.SX.sym('xki', n, 1)
        xkj = cas.SX.sym('xkj', n, 1)
        col_h_val = self.game.collision_h(xki, xkj)
        self.collision_h_casadi = cas.Function(
            'ilq_collision_h', [xki, xkj] + config_params, [col_h_val])

        self._construct_lq_stage_casadi(config_params)
        self._construct_policy_rollout_casadi(config_params)

        logger.info('Constructing ILQGame CasADi functions... Done')

    def _casadi_function_options(self):
        """Return options for optional JIT CasADi helper functions."""
        return {'jit': True} if self._casadi_jit_enabled else {}

    def _casadi_jit_function(self, name, inputs, outputs):
        """Create a JIT helper, falling back only if the local compiler rejects it."""
        opts = self._casadi_function_options()
        if not opts:
            return cas.Function(name, inputs, outputs)
        try:
            return cas.Function(name, inputs, outputs, opts)
        except RuntimeError as exc:
            if not self._is_casadi_compile_failure(exc):
                raise
            logger.warning(
                'CasADi JIT compilation failed for %s; retrying without JIT.',
                name,
            )
            return cas.Function(name, inputs, outputs)

    @staticmethod
    def _is_casadi_compile_failure(exc):
        text = str(exc).lower()
        return 'compilation failed' in text or 'shell_compiler' in text

    def _construct_context_stage_function(self):
        """Build a function mapping a stacked stage state to stacked context."""
        n = self.n
        xki = cas.SX.sym('xki', n, 1)
        xnN = cas.SX.sym('xnN', n * self.N, 1)
        context_val = self.game.get_context(xki)
        self.get_context_agent_casadi = cas.Function('ilq_get_context_agent', [xki], [context_val])
        context_flat = self._stage_context_expr(xnN)
        return cas.Function('ilq_get_context_stage', [xnN], [context_flat])

    def _stage_context_expr(self, x_stage):
        """Return symbolic stacked context for one stage."""
        if self.n_c == 0:
            return cas.SX.zeros(0, 1)
        x_mat = cas.reshape(x_stage, self.n, self.N)
        return cas.vertcat(*[
            self.game.get_context(x_mat[:, i])
            for i in range(self.N)
        ])

    def _stage_dynamics_expr(self, x_stage, u_stage, context_stage):
        """Return symbolic stacked next state for one time step."""
        n = self.n
        m = self.m
        N = self.N
        n_c = self.n_c
        eye = cas.SX.eye(N)
        x_mat = cas.reshape(x_stage, n, N)
        u_mat = cas.reshape(u_stage, m, N)
        context_mat = cas.reshape(context_stage, n_c, N)
        x_next = []
        for i in range(N):
            x_next.append(
                self.game.f(x_mat[:, i], u_mat[:, i], eye[:, i], context_mat[:, i])
            )
        return cas.vertcat(*x_next)

    def _stage_cost_expr(self, k, i, x_stage, u_stage, context_stage, barrier_weight):
        """Return symbolic player-i stage cost for LQ approximation."""
        n = self.n
        m = self.m
        N = self.N
        x_mat = cas.reshape(x_stage, n, N)
        u_mat = cas.reshape(u_stage, m, N)
        i_onehot = cas.SX.eye(N)[:, i]
        zero_u_i = cas.SX.zeros(m, 1)

        running_state = self.game.J(x_mat, zero_u_i, i_onehot)
        control_part = self.game.J(x_mat, u_mat[:, i], i_onehot) - running_state
        if k == self.T - 1:
            state_part = self.game.Jfi(x_mat, i_onehot)
        else:
            state_part = running_state

        barrier_part = self._stage_barrier_expr(
            k, i, x_stage, u_stage, context_stage, barrier_weight)
        cost = state_part + control_part + barrier_part
        assert cost.shape == (1, 1)
        return cost

    def _stage_barrier_expr(self, k, i, x_stage, u_stage, context_stage, barrier_weight):
        """Build relaxed log-barrier cost for one stage and player."""
        if self.n_h_stage == 0:
            return cas.SX.zeros(1, 1)

        nN = self.n * self.N
        mN = self.m * self.N
        cN = self.n_c * self.N
        x_cols = [cas.SX.zeros(nN, 1) for _ in range(self.T)]
        u_cols = [cas.SX.zeros(mN, 1) for _ in range(self.T)]
        context_cols = [cas.SX.zeros(cN, 1) for _ in range(self.T)]
        x_cols[k] = x_stage
        u_cols[k] = u_stage
        context_cols[k] = context_stage

        h_full = self.game.h(
            cas.horzcat(*x_cols),
            cas.horzcat(*u_cols),
            cas.horzcat(*context_cols),
        )
        h_col = 0 if self.variational_gne else i
        h_stage = h_full[k * self.n_h_stage:(k + 1) * self.n_h_stage, h_col]

        barrier = cas.SX.zeros(1, 1)
        for row in range(self.n_h_stage):
            barrier += barrier_weight * self._relaxed_log_barrier(h_stage[row])
        return barrier

    def _relaxed_log_barrier(self, h):
        """Relaxed -log(-h) barrier with quadratic extension near h=0."""
        delta = float(self.config.barrier_relaxation)
        phi_log = -cas.log(-h)
        z = h + delta
        phi_quad = -np.log(delta) + z / delta + 0.5 * (z / delta) ** 2
        return cas.if_else(h < -delta, phi_log, phi_quad)

    def _construct_lq_stage_casadi(self, config_params):
        """Build JIT helpers for dense LQ assembly and value recursion."""
        self.lq_stage_assemble_casadi = [
            self._construct_lq_stage_assemble_casadi(k, config_params)
            for k in range(self.T)
        ]
        self.lq_stage_update_casadi = self._construct_lq_stage_update_casadi()

    def _construct_lq_stage_assemble_casadi(self, k, config_params):
        """Build a per-stage helper that returns the dense stationary system."""
        nN = self.n * self.N
        mN = self.m * self.N
        eye_x = cas.SX.eye(nN)
        eye_u = cas.SX.eye(self.m)
        eye_mN = cas.SX.eye(mN)

        x_dyn = cas.SX.sym(f'x_dyn_k{k}', nN, 1)
        x_cost = cas.SX.sym(f'x_cost_k{k}', nN, 1)
        u_stage = cas.SX.sym(f'u_stage_k{k}', mN, 1)
        Z_stack = cas.SX.sym(f'Z_stack_k{k}', nN, nN * self.N)
        zeta_stack = cas.SX.sym(f'zeta_stack_k{k}', nN, self.N)
        barrier_weight = cas.SX.sym(f'barrier_weight_k{k}', 1, 1)

        context_dyn = self._stage_context_expr(x_dyn)
        context_cost = self._stage_context_expr(x_cost)
        f_val = self._stage_dynamics_expr(x_dyn, u_stage, context_dyn)
        A = cas.jacobian(f_val, x_dyn)
        B = cas.jacobian(f_val, u_stage)

        S = cas.SX.zeros(mN, mN)
        Y = cas.SX.zeros(mN, nN)
        y = cas.SX.zeros(mN, 1)
        M_stack = cas.SX.zeros(nN, nN * self.N)
        ell_stack = cas.SX.zeros(nN, self.N)
        R_stack = cas.SX.zeros(self.m, self.m * self.N * self.N)
        r_stack = cas.SX.zeros(self.m, self.N * self.N)

        for i in range(self.N):
            row = slice(i * self.m, (i + 1) * self.m)
            z_col = slice(i * nN, (i + 1) * nN)
            cost = self._stage_cost_expr(k, i, x_cost, u_stage, context_cost, barrier_weight)
            q_i = cas.jacobian(cost, x_cost).T
            Q_i = cas.hessian(cost, x_cost)[0]
            Q_i = 0.5 * (Q_i + Q_i.T) + self.config.state_regularization * eye_x
            M_i = Q_i + Z_stack[:, z_col]
            ell_i = q_i + zeta_stack[:, i]
            Bi = B[:, row]

            M_stack[:, z_col] = M_i
            ell_stack[:, i] = ell_i

            for j in range(self.N):
                col = slice(j * self.m, (j + 1) * self.m)
                r_col = i * self.N + j
                R_col = slice(r_col * self.m, (r_col + 1) * self.m)
                u_j = u_stage[col]
                R_ij = cas.hessian(cost, u_j)[0]
                R_ij = 0.5 * (R_ij + R_ij.T)
                if i == j:
                    R_ij += self.config.control_regularization * eye_u
                r_ij = cas.jacobian(cost, u_j).T

                R_stack[:, R_col] = R_ij
                r_stack[:, r_col] = r_ij

                block = Bi.T @ M_i @ B[:, col]
                if i == j:
                    block += R_ij
                S[row, col] = block

            Y[row, :] = Bi.T @ M_i @ A
            y[row, :] = Bi.T @ ell_i + r_stack[:, i * self.N + i]

        S_reg = S + self.config.lq_regularization * eye_mN
        return self._casadi_jit_function(
            f'ilq_lq_stage_assemble_k{k}',
            [x_dyn, x_cost, u_stage, Z_stack, zeta_stack, barrier_weight] + config_params,
            [A, B, S_reg, Y, y, M_stack, ell_stack, R_stack, r_stack],
        )

    def _construct_lq_stage_update_casadi(self):
        """Build a helper that applies the LQ value recursion after the solve."""
        nN = self.n * self.N
        mN = self.m * self.N

        A = cas.SX.sym('A_lq_update', nN, nN)
        B = cas.SX.sym('B_lq_update', nN, mN)
        M_stack = cas.SX.sym('M_stack_lq_update', nN, nN * self.N)
        ell_stack = cas.SX.sym('ell_stack_lq_update', nN, self.N)
        R_stack = cas.SX.sym('R_stack_lq_update', self.m, self.m * self.N * self.N)
        r_stack = cas.SX.sym('r_stack_lq_update', self.m, self.N * self.N)
        P = cas.SX.sym('P_lq_update', mN, nN)
        v = cas.SX.sym('v_lq_update', mN, 1)

        F = A - B @ P
        beta = B @ v
        next_Z_stack = cas.SX.zeros(nN, nN * self.N)
        next_zeta_stack = cas.SX.zeros(nN, self.N)

        for i in range(self.N):
            z_col = slice(i * nN, (i + 1) * nN)
            M_i = M_stack[:, z_col]
            ell_i = ell_stack[:, i]
            Zi = F.T @ M_i @ F
            zi = F.T @ (M_i @ beta + ell_i)
            for j in range(self.N):
                row = slice(j * self.m, (j + 1) * self.m)
                r_col = i * self.N + j
                R_col = slice(r_col * self.m, (r_col + 1) * self.m)
                Pj = P[row, :]
                vj = v[row, :]
                R_ij = R_stack[:, R_col]
                r_ij = r_stack[:, r_col]
                Zi += Pj.T @ R_ij @ Pj
                zi -= Pj.T @ (R_ij @ vj + r_ij)

            next_Z_stack[:, z_col] = 0.5 * (Zi + Zi.T)
            next_zeta_stack[:, i] = zi

        return self._casadi_jit_function(
            'ilq_lq_stage_update',
            [A, B, M_stack, ell_stack, R_stack, r_stack, P, v],
            [next_Z_stack, next_zeta_stack],
        )

    def _construct_policy_rollout_casadi(self, config_params):
        """Build a JIT-able nonlinear rollout for a stacked affine policy."""
        nN = self.n * self.N
        mN = self.m * self.N
        gc = self.game.config

        u_ref = cas.SX.sym('u_ref_policy_rollout', mN, self.T)
        x_ref = cas.SX.sym('x_ref_policy_rollout', nN, self.T)
        P_stack = cas.SX.sym('P_policy_rollout', mN, nN * self.T)
        v_stack = cas.SX.sym('v_policy_rollout', mN, self.T)
        step_size = cas.SX.sym('step_size_policy_rollout', 1, 1)

        xk = cas.reshape(gc.get_param('x0'), nN, 1)
        x_cols = []
        u_cols = []
        context_cols = []
        for k in range(self.T):
            x_ref_k = cas.reshape(gc.get_param('x0'), nN, 1) if k == 0 else x_ref[:, k - 1]
            dx = xk - x_ref_k
            Pk = P_stack[:, k * nN:(k + 1) * nN]
            uk = u_ref[:, k] - self.config.feedback_scale * (Pk @ dx) + step_size * v_stack[:, k]
            context = self._stage_context_expr(xk)
            x_next, _, _ = self.stage_dynamics_casadi(xk, uk, context, *config_params)
            x_cols.append(x_next)
            u_cols.append(uk)
            context_cols.append(self._stage_context_expr(x_next))
            xk = x_next

        X = cas.horzcat(*x_cols)
        U = cas.horzcat(*u_cols)
        context_traj = (
            cas.horzcat(*context_cols)
            if context_cols
            else cas.SX.zeros(self.n_c * self.N, 0)
        )
        h_vec = cas.vec(self.game.h(X, U, context_traj))

        self.policy_rollout_casadi = cas.Function(
            'ilq_policy_rollout',
            [u_ref, x_ref, P_stack, v_stack, step_size] + config_params,
            [X, U, h_vec],
            self._casadi_function_options(),
        )

    def _params_np(self):
        gc = self.game.config
        return [gc.get_int_param_np(), gc.get_double_param_np()]

    def _initial_control(self, u_ref):
        if u_ref is None:
            return np.zeros((self.m * self.N, self.T), order='F')
        u = np.asarray(u_ref, dtype=float, order='F')
        if u.shape == (self.m, self.N, self.T):
            u = u.reshape((self.m * self.N, self.T), order='F')
        assert u.shape == (self.m * self.N, self.T)
        return np.array(u, dtype=float, order='F', copy=True)

    def prime_casadi(self):
        """Run CasADi functions once so JIT compilation is outside solve timing."""
        if self._casadi_primed:
            return
        logger.info('Priming ILQGame CasADi functions... ')
        u = np.zeros((self.m * self.N, self.T), order='F')
        params = self._params_np()
        x = self.rollout_casadi(self.game.config.x0, u, *params)
        x_np = np.asarray(x, dtype=float, order='F')
        x0_flat = self.game.config.x0.reshape((self.n * self.N, 1), order='F')
        u0 = u[:, [0]]
        context0 = self._stage_context_np(x0_flat)
        self.stage_dynamics_casadi(x0_flat, u0, context0, *params)
        self.h_casadi(
            x_np,
            u,
            np.zeros((self.n_c * self.N, self.T), order='F'),
            *params,
        )
        full_x = self._full_x(x_np)
        Z_stack = cas.DM.zeros(self.n * self.N, self.n * self.N * self.N)
        zeta_stack = cas.DM.zeros(self.n * self.N, self.N)
        P_zero_stage = np.zeros((self.m * self.N, self.n * self.N), order='F')
        v_zero_stage = np.zeros((self.m * self.N, 1), order='F')
        for k in range(self.T):
            x_dyn = full_x[:, [k]]
            x_cost = full_x[:, [k + 1]]
            u_stage = u[:, [k]]
            outputs = self.lq_stage_assemble_casadi[k](
                x_dyn, x_cost, u_stage, Z_stack, zeta_stack, self.barrier_weight, *params)
            self.lq_stage_update_casadi(
                outputs[0],
                outputs[1],
                outputs[5],
                outputs[6],
                outputs[7],
                outputs[8],
                P_zero_stage,
                v_zero_stage,
            )
        P_zero = np.zeros((self.m * self.N, self.n * self.N * self.T), order='F')
        v_zero = np.zeros((self.m * self.N, self.T), order='F')
        self.policy_rollout_casadi(
            u,
            x_np,
            P_zero,
            v_zero,
            self.config.step_size,
            *params,
        )
        self._casadi_primed = True
        logger.info('Priming ILQGame CasADi functions... Done')

    def _rollout_open_loop(self, u):
        """Roll out stacked open-loop controls and return x1..xT."""
        x = self.rollout_casadi(self.game.config.x0, u, *self._params_np())
        return np.array(x, dtype=float, order='F')

    def _full_x(self, x):
        return np.hstack([
            self.game.config.x0.reshape((self.n * self.N, 1), order='F'),
            np.asarray(x, dtype=float, order='F'),
        ])

    def _stage_context_np(self, x_stage):
        context = self.get_context_stage_casadi(x_stage)
        return np.asarray(context, dtype=float).reshape((self.n_c * self.N, 1), order='F')

    def _solve_lq_game(self, full_x, u):
        """Solve the finite-horizon LQ game with JIT matrix assembly."""
        t = self.profiler
        nN = self.n * self.N
        N = self.N
        T = self.T
        mN = self.m * N
        params = self._params_np()

        Z_stack = cas.DM.zeros(nN, nN * N)
        zeta_stack = cas.DM.zeros(nN, N)
        P_policy = np.empty((mN, nN * T), dtype=float, order='F')
        v_policy = np.empty((mN, T), dtype=float, order='F')

        for k in range(T - 1, -1, -1):
            x_dyn = full_x[:, [k]]
            x_cost = full_x[:, [k + 1]]
            u_stage = u[:, [k]]

            t.s('casadi stage assembly')
            (
                A,
                B,
                S_reg_dm,
                Y_dm,
                y_dm,
                M_stack,
                ell_stack,
                R_stack,
                r_stack,
            ) = self.lq_stage_assemble_casadi[k](
                x_dyn,
                x_cost,
                u_stage,
                Z_stack,
                zeta_stack,
                self.barrier_weight,
                *params,
            )
            t.e('casadi stage assembly')

            t.s('_solve_stationary_system')
            S_reg = np.asarray(S_reg_dm, dtype=float, order='F')
            Y = np.asarray(Y_dm, dtype=float, order='F')
            y = np.asarray(y_dm, dtype=float, order='F').reshape((mN, 1), order='F')
            rhs = np.empty((mN, nN + 1), dtype=float, order='F')
            rhs[:, :nN] = Y
            rhs[:, [nN]] = y
            sol = self._solve_stationary_system(S_reg, rhs)
            P = np.array(sol[:, :nN], dtype=float, order='F', copy=True)
            v = np.array(-sol[:, [nN]], dtype=float, order='F', copy=True)
            t.e('_solve_stationary_system')

            P_policy[:, k * nN:(k + 1) * nN] = P
            v_policy[:, [k]] = v

            t.s('casadi value recursion')
            Z_stack, zeta_stack = self.lq_stage_update_casadi(
                A, B, M_stack, ell_stack, R_stack, r_stack, P, v)
            t.e('casadi value recursion')

        feedforward_norm = float(np.linalg.norm(v_policy) / np.sqrt(N * T))
        return ILQPolicy(P=P_policy, v=v_policy, feedforward_norm=feedforward_norm)

    @staticmethod
    def _solve_stationary_system(A, B):
        try:
            return np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            sol, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
            return sol

    def _rollout_policy(self, u_ref, x_ref, policy, step_size):
        """Execute an affine feedback policy on the nonlinear dynamics."""
        x, u, h_vec = self.policy_rollout_casadi(
            u_ref,
            x_ref,
            policy.P,
            policy.v,
            step_size,
            *self._params_np(),
        )
        x = np.asarray(x, dtype=float, order='F')
        u = np.asarray(u, dtype=float, order='F')
        h_vec = np.asarray(h_vec, dtype=float).reshape(-1, order='F')
        violation = 0.0 if h_vec.size == 0 else float(np.max(np.maximum(h_vec, 0.0)))
        return x, u, violation

    def _constraint_violation(self, x, u):
        if self.n_h == 0:
            return 0.0
        params = self._params_np()
        contexts = []
        for k in range(self.T):
            contexts.append(self._stage_context_np(x[:, [k]]))
        context = np.hstack(contexts) if contexts else np.zeros((self.n_c * self.N, 0))
        h_val = self.h_casadi(x, u, context, *params)
        h_np = np.asarray(h_val, dtype=float).reshape(-1, order='F')
        if h_np.size == 0:
            return 0.0
        return float(np.max(np.maximum(h_np, 0.0)))

    def _policy_feedforward_norm(self, policy):
        return policy.feedforward_norm

    def _select_rollout(self, u, x, x_ref, policy, old_residual, old_violation):
        """Line-search over feedforward step sizes for a stable rollout."""
        if not self.config.use_line_search:
            return self._rollout_policy(u, x_ref, policy, self.config.step_size), self.config.step_size

        best = None
        step = self.config.step_size
        old_merit = old_residual + old_violation
        for _ in range(self.config.line_search_max_iter):
            try:
                trial_x, trial_u, violation = self._rollout_policy(u, x_ref, policy, step)
            except RuntimeError:
                step *= self.config.line_search_decay
                continue

            control_step = np.linalg.norm(trial_u - u)
            merit = control_step / np.sqrt(trial_u.size) + violation
            finite = np.isfinite(trial_x).all() and np.isfinite(trial_u).all()
            if finite and control_step <= self.config.max_control_step_norm:
                if best is None or merit < best[2]:
                    best = ((trial_x, trial_u, violation), step, merit)
                if merit <= max(old_merit, self.config.tolerance):
                    break
            step *= self.config.line_search_decay

        if best is None:
            logger.info('ILQGame line search failed; keeping previous rollout')
            return (x, u, old_violation), 0.0
        del old_residual, old_violation
        return best[0], best[1]

    def step(self, x, u, violation):
        """Run one timed iLQ-game iteration."""
        t = self.profiler
        t.s()

        x_ref = np.array(x, dtype=float, order='F', copy=True)
        full_x = self._full_x(x_ref)

        t.s('hybrid lq solve')
        policy = self._solve_lq_game(full_x, u)
        self.last_policy = policy
        feedforward_norm = self._policy_feedforward_norm(policy)
        t.e('hybrid lq solve')

        t.s('rollout policy')
        (new_x, new_u, new_violation), accepted_step = self._select_rollout(
            u, x_ref, x_ref, policy, feedforward_norm, violation)
        t.e('rollout policy')

        t.s('residual checks')
        control_delta = float(np.linalg.norm(new_u - u) / np.sqrt(new_u.size))
        state_delta = float(np.linalg.norm(new_x - x_ref) / np.sqrt(new_x.size))
        residual = max(feedforward_norm, control_delta, state_delta)
        if new_violation > self.config.constraint_tolerance:
            self.barrier_weight = min(
                self.config.barrier_max,
                self.barrier_weight * self.config.barrier_growth,
            )
        converged = (
            residual <= self.config.tolerance
            and new_violation <= self.config.constraint_tolerance
        )
        t.e('residual checks')

        t.e()

        return (
            new_x,
            new_u,
            new_violation,
            residual,
            converged,
            accepted_step,
            feedforward_norm,
            control_delta,
            state_delta,
        )

    def solve(self, u_ref=None):
        """Run iterative LQ-game approximations from an open-loop control guess."""
        u = self._initial_control(u_ref)
        x = self._rollout_open_loop(u)
        violation = self._constraint_violation(x, u)
        residual = np.inf
        converged = False
        optimal = True
        msg = 'Max iteration has been reached'

        t0 = time()
        iteration = 0
        for iteration in range(self.config.iterations):
            logger.info(
                '--- ILQGame iter %d, barrier=%.6g, violation=%.6g ---',
                iteration,
                self.barrier_weight,
                violation,
            )

            (
                new_x,
                new_u,
                new_violation,
                residual,
                converged,
                accepted_step,
                feedforward_norm,
                control_delta,
                state_delta,
            ) = self.step(x, u, violation)
            self.residual_vec.append(residual)
            self.violation_vec.append(new_violation)
            logger.info(
                'feedforward=%.6g, du=%.6g, dx=%.6g, violation=%.6g, step=%.6g',
                feedforward_norm,
                control_delta,
                state_delta,
                new_violation,
                accepted_step,
            )

            u = new_u
            x = new_x
            violation = new_violation
            if converged:
                msg = 'Converged to an iLQ-game fixed point'
                break

        dt = time() - t0
        logger.info('Stop after %d iteration because %s', iteration, msg)
        return Solution(
            elapsed_time=dt,
            iterations=iteration,
            u=u.reshape((self.m, self.N, self.T), order='F'),
            x=x.reshape((self.n, self.N, self.T), order='F'),
            residual=float(residual),
            has_converged=converged,
            is_optimal=optimal,
        )

    def _rollout_full_x(self, u_ref, x_ref=None):
        """Roll out from u_ref and prepend x0 for visualization."""
        u_ref = u_ref.reshape((self.m, self.N, self.T), order='F')
        assert u_ref.shape == (self.m, self.N, self.T)
        if x_ref is None:
            u_cat = u_ref.reshape((self.m * self.N, self.T), order='F')
            x_ref = self._rollout_open_loop(u_cat)
        x_ref = np.asarray(x_ref).reshape((self.n, self.N, self.T), order='F')
        full_x = np.dstack([self.game.config.x0[:, :, np.newaxis], x_ref])
        assert full_x.shape == (self.n, self.N, self.T + 1)
        return full_x

    def visualize(self, u_ref, x_ref=None, save=False, show=True):
        """Visualize a solution with the game-specific renderer."""
        u_ref = u_ref.reshape((self.m, self.N, self.T), order='F')
        full_x = self._rollout_full_x(u_ref, x_ref)
        return self.game.visualize(u_ref, full_x, show=show, save=save)

    def animate(self, u_ref, x_ref=None, save_gif=False, save_snapshots=False):
        """Animate a solution with the game-specific renderer."""
        u_ref = u_ref.reshape((self.m, self.N, self.T), order='F')
        full_x = self._rollout_full_x(u_ref, x_ref)
        return self.game.animate(
            u_ref, full_x, show=True, save_gif=save_gif, save_snapshots=save_snapshots)

    def final(self):
        """Plot residual and constraint violation history."""
        self.profiler.summary()
        if len(self.residual_vec) == 0:
            logger.warning('No ILQGame residual history available to plot')
            return
        _, ax = plt.subplots()
        ax.plot(self.residual_vec, '*-', label='fixed-point residual')
        if len(self.violation_vec) > 0:
            ax.plot(self.violation_vec, '.-', label='constraint violation')
        ax.set_yscale('log')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Residual')
        ax.legend()
        plt.show()


ILQGameSolver = ILQGame
