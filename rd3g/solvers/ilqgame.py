"""Iterative linear-quadratic game solver built on CasADi derivatives."""

import logging
from dataclasses import dataclass
from time import time

import casadi as cas
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse

from rd3g.core.base_solver import BaseSolver, BaseSolverConfig, Solution
from rd3g.utilities.casadi_util import dm_to_csc
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
    """JIT compile the combined CasADi LQ-approximation function."""


@dataclass
class ILQPolicy:
    """Time-indexed affine feedback policy in deviation coordinates."""
    P: list
    """P[i][k] maps global state deviation to player i control feedback."""
    v: list
    """v[i][k] is player i feedforward control deviation."""


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
        """Construct sparse CasADi derivative functions for the iLQ loop."""
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
        barrier_weight = cas.SX.sym('barrier_weight', 1, 1)

        f_val = self._stage_dynamics_expr(x_stage, u_stage, context_stage)
        A_val = cas.jacobian(f_val, x_stage)
        B_val = cas.jacobian(f_val, u_stage)
        self.stage_dynamics_casadi = cas.Function(
            'ilq_stage_dynamics',
            [x_stage, u_stage, context_stage] + config_params,
            [f_val, A_val, B_val],
        )

        self.get_context_stage_casadi = self._construct_context_stage_function()

        self.stage_cost_quad_casadi = []
        for k in range(T):
            stage_funs = []
            for i in range(N):
                cost = self._stage_cost_expr(
                    k, i, x_stage, u_stage, context_stage, barrier_weight)
                q_x = cas.jacobian(cost, x_stage).T
                Q_xx = cas.hessian(cost, x_stage)[0]

                outputs = [Q_xx, q_x]
                for j in range(N):
                    u_j = u_stage[j * m:(j + 1) * m]
                    R_ij = cas.hessian(cost, u_j)[0]
                    r_ij = cas.jacobian(cost, u_j).T
                    outputs.extend([R_ij, r_ij])

                stage_funs.append(cas.Function(
                    f'ilq_cost_quad_k{k}_i{i}',
                    [x_stage, u_stage, context_stage, barrier_weight] + config_params,
                    outputs,
                ))
            self.stage_cost_quad_casadi.append(stage_funs)

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

        self._construct_lq_approx_casadi(config_params)

        logger.info('Constructing ILQGame CasADi functions... Done')

    def _casadi_function_options(self):
        """Return options for heavyweight CasADi functions."""
        return {'jit': True} if self.config.jit else {}

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

    def _construct_lq_approx_casadi(self, config_params):
        """Build one JIT-able function for rollout, linearization, and costs."""
        gc = self.game.config
        nN = self.n * self.N
        mN = self.m * self.N
        eye_x = cas.SX.eye(nN)
        eye_u = cas.SX.eye(self.m)

        u = cas.SX.sym('u_lq_approx', mN, self.T)
        barrier_weight = cas.SX.sym('barrier_weight', 1, 1)
        x0 = gc.get_param('x0')
        X = self.game.rollout(x0, u)
        full_x = [cas.reshape(x0, nN, 1)] + [X[:, k] for k in range(self.T)]

        A_blocks = []
        B_blocks = []
        Q_blocks = []
        q_cols = []
        R_blocks = []
        r_cols = []
        h_context_cols = []

        for k in range(self.T):
            x_dyn = full_x[k]
            u_stage = u[:, k]
            context_dyn = self._stage_context_expr(x_dyn)
            _, A, B = self.stage_dynamics_casadi(
                x_dyn, u_stage, context_dyn, *config_params)
            A_blocks.append(A)
            B_blocks.append(B)

            x_cost = full_x[k + 1]
            context_cost = self._stage_context_expr(x_cost)
            h_context_cols.append(context_cost)
            for i in range(self.N):
                outputs = self.stage_cost_quad_casadi[k][i](
                    x_cost, u_stage, context_cost, barrier_weight, *config_params)
                Q = 0.5 * (outputs[0] + outputs[0].T)
                Q += self.config.state_regularization * eye_x
                Q_blocks.append(Q)
                q_cols.append(outputs[1])

                idx = 2
                for j in range(self.N):
                    R = 0.5 * (outputs[idx] + outputs[idx].T)
                    if i == j:
                        R += self.config.control_regularization * eye_u
                    R_blocks.append(R)
                    r_cols.append(outputs[idx + 1])
                    idx += 2

        context_traj = (
            cas.horzcat(*h_context_cols)
            if h_context_cols
            else cas.SX.zeros(self.n_c * self.N, 0)
        )
        h_val = self.game.h(X, u, context_traj)

        A_stack = cas.horzcat(*A_blocks)
        B_stack = cas.horzcat(*B_blocks)
        Q_stack = cas.horzcat(*Q_blocks)
        q_stack = cas.horzcat(*q_cols)
        R_stack = cas.horzcat(*R_blocks)
        r_stack = cas.horzcat(*r_cols)
        h_vec = cas.vec(h_val)

        self.lq_approx_casadi = cas.Function(
            'ilq_lq_approx',
            [u, barrier_weight] + config_params,
            [X, A_stack, B_stack, Q_stack, q_stack, R_stack, r_stack, h_vec],
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
        self.lq_approx_casadi(u, self.barrier_weight, *params)
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

    def _linearize_dynamics(self, full_x, u):
        """Evaluate sparse dynamics Jacobians along the reference trajectory."""
        params = self._params_np()
        As = []
        Bs = [[] for _ in range(self.N)]
        for k in range(self.T):
            xk = full_x[:, [k]]
            uk = u[:, [k]]
            context = self._stage_context_np(xk)
            _, A_dm, B_dm = self.stage_dynamics_casadi(xk, uk, context, *params)
            As.append(dm_to_csc(A_dm))
            B = dm_to_csc(B_dm)
            for i in range(self.N):
                Bs[i].append(B[:, i * self.m:(i + 1) * self.m].tocsc())
        return As, Bs

    def _quadraticize_costs(self, full_x, u):
        """Evaluate sparse cost Hessians and gradients along the trajectory."""
        params = self._params_np()
        Qs = [[] for _ in range(self.N)]
        qs = [[] for _ in range(self.N)]
        Rs = [[[] for _ in range(self.N)] for _ in range(self.N)]
        rs = [[[] for _ in range(self.N)] for _ in range(self.N)]

        state_eye = scipy.sparse.eye(self.n * self.N, format='csc')
        control_eye = scipy.sparse.eye(self.m, format='csc')
        for k in range(self.T):
            x_stage = full_x[:, [k + 1]]
            u_stage = u[:, [k]]
            context = self._stage_context_np(x_stage)
            for i in range(self.N):
                outputs = self.stage_cost_quad_casadi[k][i](
                    x_stage, u_stage, context, self.barrier_weight, *params)
                Q = dm_to_csc(outputs[0])
                q = np.asarray(outputs[1], dtype=float).reshape((self.n * self.N, 1), order='F')
                Q = self._symmetrize_sparse(Q) + self.config.state_regularization * state_eye
                Qs[i].append(Q)
                qs[i].append(q)

                idx = 2
                for j in range(self.N):
                    R = dm_to_csc(outputs[idx])
                    r = np.asarray(outputs[idx + 1], dtype=float).reshape((self.m, 1), order='F')
                    R = self._symmetrize_sparse(R)
                    if i == j:
                        R = R + self.config.control_regularization * control_eye
                    Rs[i][j].append(R)
                    rs[i][j].append(r)
                    idx += 2
        return Qs, qs, Rs, rs

    def _evaluate_lq_approx(self, u):
        """Evaluate and unpack the combined CasADi LQ approximation function."""
        outputs = self.lq_approx_casadi(u, self.barrier_weight, *self._params_np())
        return self._unpack_lq_approx(outputs)

    def _unpack_lq_approx(self, outputs):
        """Convert stacked CasADi LQ approximation outputs to solver lists."""
        nN = self.n * self.N
        mN = self.m * self.N

        x = np.asarray(outputs[0], dtype=float, order='F')
        A_stack = dm_to_csc(outputs[1])
        B_stack = dm_to_csc(outputs[2])
        Q_stack = dm_to_csc(outputs[3])
        q_stack = np.asarray(outputs[4], dtype=float, order='F')
        R_stack = dm_to_csc(outputs[5])
        r_stack = np.asarray(outputs[6], dtype=float, order='F')
        h_vec = np.asarray(outputs[7], dtype=float).reshape(-1, order='F')

        As = []
        Bs = [[] for _ in range(self.N)]
        for k in range(self.T):
            As.append(A_stack[:, k * nN:(k + 1) * nN].tocsc())
            Bk = B_stack[:, k * mN:(k + 1) * mN].tocsc()
            for i in range(self.N):
                Bs[i].append(Bk[:, i * self.m:(i + 1) * self.m].tocsc())

        Qs = [[] for _ in range(self.N)]
        qs = [[] for _ in range(self.N)]
        Rs = [[[] for _ in range(self.N)] for _ in range(self.N)]
        rs = [[[] for _ in range(self.N)] for _ in range(self.N)]
        for k in range(self.T):
            for i in range(self.N):
                qi_idx = k * self.N + i
                Qs[i].append(Q_stack[:, qi_idx * nN:(qi_idx + 1) * nN].tocsc())
                qs[i].append(q_stack[:, [qi_idx]])
                for j in range(self.N):
                    rij_idx = (k * self.N + i) * self.N + j
                    Rs[i][j].append(
                        R_stack[:, rij_idx * self.m:(rij_idx + 1) * self.m].tocsc()
                    )
                    rs[i][j].append(r_stack[:, [rij_idx]])

        ref_violation = 0.0 if h_vec.size == 0 else float(np.max(np.maximum(h_vec, 0.0)))
        return x, As, Bs, Qs, qs, Rs, rs, ref_violation

    @staticmethod
    def _symmetrize_sparse(mat):
        mat = mat.tocsc()
        return ((mat + mat.T) * 0.5).tocsc()

    def _solve_lq_game(self, As, Bs, Qs, qs, Rs, rs):
        """Solve the local finite-horizon LQ game by backward recursion."""
        nN = self.n * self.N
        m = self.m
        N = self.N
        T = self.T

        Z = [scipy.sparse.csc_matrix((nN, nN)) for _ in range(N)]
        zeta = [np.zeros((nN, 1)) for _ in range(N)]
        Ps_reversed = []
        vs_reversed = []

        for k in range(T - 1, -1, -1):
            S = np.zeros((N * m, N * m))
            Y = np.zeros((N * m, nN))
            y = np.zeros((N * m, 1))

            M = []
            ell = []
            for i in range(N):
                M_i = (Qs[i][k] + Z[i]).tocsc()
                ell_i = qs[i][k] + zeta[i]
                M.append(M_i)
                ell.append(ell_i)

                Bi = Bs[i][k]
                for j in range(N):
                    block = Bi.T @ M_i @ Bs[j][k]
                    if i == j:
                        block = block + Rs[i][i][k]
                    S[i * m:(i + 1) * m, j * m:(j + 1) * m] = block.toarray()

                Y_i = Bi.T @ M_i @ As[k]
                y_i = Bi.T @ ell_i + rs[i][i][k]
                Y[i * m:(i + 1) * m, :] = Y_i.toarray()
                y[i * m:(i + 1) * m, :] = y_i

            S_reg = S + self.config.lq_regularization * np.eye(N * m)
            P_stack = self._solve_stationary_system(S_reg, Y)
            v_stack = -self._solve_stationary_system(S_reg, y)
            P_k = [
                P_stack[i * m:(i + 1) * m, :].reshape((m, nN), order='F')
                for i in range(N)
            ]
            v_k = [
                v_stack[i * m:(i + 1) * m, :].reshape((m, 1), order='F')
                for i in range(N)
            ]
            Ps_reversed.append(P_k)
            vs_reversed.append(v_k)

            F = As[k].copy().tocsc()
            beta = np.zeros((nN, 1))
            for j in range(N):
                F = F - Bs[j][k] @ scipy.sparse.csc_matrix(P_k[j])
                beta += Bs[j][k] @ v_k[j]
            F = F.tocsc()

            next_Z = []
            next_zeta = []
            for i in range(N):
                Zi = F.T @ M[i] @ F
                zi = F.T @ (M[i] @ beta + ell[i])
                for j in range(N):
                    Pj = scipy.sparse.csc_matrix(P_k[j])
                    R_ij = Rs[i][j][k]
                    vj = v_k[j]
                    r_ij = rs[i][j][k]
                    Zi = Zi + Pj.T @ R_ij @ Pj
                    zi = zi - Pj.T @ (R_ij @ vj + r_ij)
                next_Z.append(self._symmetrize_sparse(Zi))
                next_zeta.append(np.asarray(zi).reshape((nN, 1), order='F'))
            Z = next_Z
            zeta = next_zeta

        Ps = [[] for _ in range(N)]
        vs = [[] for _ in range(N)]
        for k in range(T):
            P_k = Ps_reversed[T - 1 - k]
            v_k = vs_reversed[T - 1 - k]
            for i in range(N):
                Ps[i].append(P_k[i])
                vs[i].append(v_k[i])

        return ILQPolicy(P=Ps, v=vs)

    @staticmethod
    def _solve_stationary_system(A, B):
        try:
            return np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            sol, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
            return sol

    def _rollout_policy(self, u_ref, full_x_ref, policy, step_size):
        """Execute an affine feedback policy on the nonlinear dynamics."""
        params = self._params_np()
        nN = self.n * self.N
        mN = self.m * self.N
        x = np.zeros((nN, self.T), order='F')
        u = np.zeros((mN, self.T), order='F')
        xk = self.game.config.x0.reshape((nN, 1), order='F')

        for k in range(self.T):
            dx = xk - full_x_ref[:, [k]]
            uk = u_ref[:, [k]].copy()
            for i in range(self.N):
                du_i = (
                    -self.config.feedback_scale * policy.P[i][k] @ dx
                    + step_size * policy.v[i][k]
                )
                uk[i * self.m:(i + 1) * self.m, :] += du_i

            context = self._stage_context_np(xk)
            x_next, _, _ = self.stage_dynamics_casadi(xk, uk, context, *params)
            xk = np.asarray(x_next, dtype=float).reshape((nN, 1), order='F')
            u[:, [k]] = uk
            x[:, [k]] = xk

        return x, u

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
        vals = [
            np.linalg.norm(policy.v[i][k])
            for i in range(self.N)
            for k in range(self.T)
        ]
        if not vals:
            return 0.0
        return float(np.linalg.norm(vals) / np.sqrt(len(vals)))

    def _select_rollout(self, u, x, full_x, policy, old_residual, old_violation):
        """Line-search over feedforward step sizes for a stable rollout."""
        if not self.config.use_line_search:
            return self._rollout_policy(u, full_x, policy, self.config.step_size), self.config.step_size

        best = None
        step = self.config.step_size
        old_merit = old_residual + old_violation
        for _ in range(self.config.line_search_max_iter):
            try:
                trial_x, trial_u = self._rollout_policy(u, full_x, policy, step)
            except RuntimeError:
                step *= self.config.line_search_decay
                continue

            control_step = np.linalg.norm(trial_u - u)
            violation = self._constraint_violation(trial_x, trial_u)
            merit = control_step / np.sqrt(trial_u.size) + violation
            finite = np.isfinite(trial_x).all() and np.isfinite(trial_u).all()
            if finite and control_step <= self.config.max_control_step_norm:
                if best is None or merit < best[2]:
                    best = ((trial_x, trial_u), step, merit)
                if merit <= max(old_merit, self.config.tolerance):
                    break
            step *= self.config.line_search_decay

        if best is None:
            logger.warning('ILQGame line search failed; keeping previous rollout')
            return (x, u), 0.0
        del old_residual, old_violation
        return best[0], best[1]

    def step(self, x, u, violation):
        """Run one timed iLQ-game iteration."""
        t = self.profiler
        t.s()

        t.s('casadi lq approximation')
        lq_outputs = self.lq_approx_casadi(u, self.barrier_weight, *self._params_np())
        t.e('casadi lq approximation')

        t.s('unpack lq approximation')
        x_ref, As, Bs, Qs, qs, Rs, rs, ref_violation = self._unpack_lq_approx(lq_outputs)
        full_x = self._full_x(x_ref)
        t.e('unpack lq approximation')

        t.s('solve lq game')
        policy = self._solve_lq_game(As, Bs, Qs, qs, Rs, rs)
        self.last_policy = policy
        feedforward_norm = self._policy_feedforward_norm(policy)
        t.e('solve lq game')

        t.s('rollout policy')
        (new_x, new_u), accepted_step = self._select_rollout(
            u, x_ref, full_x, policy, feedforward_norm, max(violation, ref_violation))
        t.e('rollout policy')

        t.s('residual checks')
        control_delta = float(np.linalg.norm(new_u - u) / np.sqrt(new_u.size))
        state_delta = float(np.linalg.norm(new_x - x_ref) / np.sqrt(new_x.size))
        new_violation = self._constraint_violation(new_x, new_u)
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
