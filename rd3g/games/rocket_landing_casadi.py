"""Rocket landing game, CasADi version."""
import os
import logging
from dataclasses import dataclass, field
from math import degrees, radians
from typing import Any

import numpy as np
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.animation import FuncAnimation
import casadi as cas

from rd3g.utilities.util import BASEDIR, resolve_logname
from rd3g.core.casadi_game import CasadiGame, CasadiGameConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass(frozen=False)
class RocketLandingCasadiConfig(CasadiGameConfig):
    """Configuration for the rocket landing game.

    Agent 0 is the rocket:
        x = [x, y, theta, vx, vy, omega]
        u = [Tx, Ty]

    Agent 1 is the ship, packed into the same state/control dimensions:
        x = [y, vy, 0, 0, 0, 0]
        u = [ay, unused]
    """
    T: int = 20
    dt: float = 0.2
    N: int = 2
    n: int = 6
    m: int = 2
    n_h: int = 0
    """Total number of canonical inequality constraints."""
    n_c: int = 0
    """Dimension of context variable per agent per stage; unused here."""
    variational_gne: bool = False
    """If True, use one shared multiplier per canonical constraint."""
    constraints_per_agent: int = 7

    x0: Any = field(
        default_factory=lambda: np.array(
            [
                [-10.0, -3.0],
                [0.0, -0.5],
                [radians(10.0), 0.0],
                [-0.3, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=float,
            order='F',
        )
    )
    """Initial state for all agents, dim: (n,N)."""
    target_x_ref: Any = field(
        default_factory=lambda: np.zeros((6, 2), dtype=float, order='F')
    )
    """Target state for all agents, dim: (n,N)."""

    q_rocket_diag: Any = field(
        default_factory=lambda: np.array(
            [1.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=float
        ).reshape((6, 1), order='F')
    )
    q_ship_diag: Any = field(
        default_factory=lambda: np.zeros((6, 1), dtype=float, order='F')
    )
    q_dock_diag: Any = field(
        default_factory=lambda: np.array([1.0, 0.0], dtype=float).reshape(
            (2, 1), order='F')
    )
    r_rocket_diag: Any = field(
        default_factory=lambda: np.ones((2, 1), dtype=float, order='F') * 1e-2
    )
    r_ship_diag: Any = field(
        default_factory=lambda: 10*np.ones((2, 1), dtype=float, order='F') * 1e-2
    )
    terminal_weight: float = 20.0

    main_thrust_max: float = 3.0
    side_thrust_max: float = 2.0
    ship_accel_max: float = 1.0
    unused_ship_control_max: float = 0.25
    max_tilt: float = radians(60.0)
    height_scale: float = 10.0
    ship_y_max: float = 5.0
    ship_speed_max: float = 2.0

    def __post_init__(self):
        assert self.N == 2, 'RocketLandingCasadi currently models one rocket and one ship'
        assert self.x0.shape == (self.n, self.N), (
            'Incorrect self.x0 dimension, '
            f'should be {(self.n, self.N)}, but got {self.x0.shape}')
        assert self.target_x_ref.shape == (self.n, self.N)
        assert self.q_rocket_diag.shape == (self.n, 1)
        assert self.q_ship_diag.shape == (self.n, 1)
        assert self.q_dock_diag.shape == (self.m, 1)
        assert self.r_rocket_diag.shape == (self.m, 1)
        assert self.r_ship_diag.shape == (self.m, 1)
        if self.n_h == 0:
            self.n_h = self.constraints_per_agent * self.N * self.T
        return super().__post_init__()


class RocketLandingCasadi(CasadiGame):
    """Rocket landing game with a moving ship deck.

    The coordinate convention follows the older residual-game implementation:
    the displayed altitude is ``-x`` and the deck is at ``x = 0``.
    """

    def __init__(self, config: RocketLandingCasadiConfig):
        super().__init__(config)
        self.visual_x_lim = [-8, 8]
        self.visual_y_lim = [-2, 14]
        self.rocket_scale = 0.01 / 2
        self.ship_scale = 0.01 / 2
        self.rocket_img = mpimg.imread(
            os.path.join(BASEDIR, 'rd3g', 'resources', 'rocket_alpha.png'))
        self.ship_img = mpimg.imread(
            os.path.join(BASEDIR, 'rd3g', 'resources', 'ship_alpha.png'))

    @staticmethod
    def _sparse_selector(rows, cols, shape):
        """Return a sparse CasADi selector matrix with ones at rows/cols."""
        sparsity = cas.Sparsity.triplet(shape[0], shape[1], rows, cols)
        return cas.SX(sparsity, [1.0] * len(rows))

    @staticmethod
    def _sparse_diag(diag_values):
        """Return a sparse diagonal matrix from a CasADi vector."""
        return cas.diag(diag_values)

    def _rocket_dock_selector(self):
        return self._sparse_selector([0, 1], [1, 4], (2, self.config.n))

    def _ship_dock_selector(self):
        return self._sparse_selector([0, 1], [0, 1], (2, self.config.n))

    def initial_control_guess(self):
        """Return a simple feasible-ish open-loop descent guess, shape (m*N,T)."""
        gc = self.config
        u = np.zeros((gc.m, gc.N, gc.T), dtype=float, order='F')
        u[0, 0, :] = 0.3
        return u.reshape((gc.m * gc.N, gc.T), order='F')

    def visualize(self, u, x, show=True, save=False):
        """Visualize trajectories and final poses.

        Args:
            u: (m,N,T)
            x: (n,N,T+1)
        """
        gc = self.config
        if (not show) and (not save):
            return None
        assert u.shape == (gc.m, gc.N, gc.T)
        assert x.shape == (gc.n, gc.N, gc.T + 1)

        fig, ax = plt.subplots()
        ax.set_facecolor((54 / 255, 69 / 255, 79 / 255))
        ax.hlines(y=0, xmin=self.visual_x_lim[0], xmax=self.visual_x_lim[1],
                  colors='white', linewidth=1.5)

        self._draw_rocket(ax, x[:, 0, 0], alpha=0.55)
        self._draw_rocket(ax, x[:, 0, -1], alpha=1.0)
        self._draw_ship(ax, x[:, 1, 0], alpha=0.55)
        self._draw_ship(ax, x[:, 1, -1], alpha=1.0)

        ax.plot(x[1, 0, :], -x[0, 0, :], '*-', color='tab:orange', zorder=4)
        ax.plot(x[0, 1, :], np.zeros(gc.T + 1), '*-', color='tab:blue', zorder=4)

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(*self.visual_x_lim)
        ax.set_ylim(*self.visual_y_lim)
        ax.set_xlabel('ship-frame lateral position')
        ax.set_ylabel('altitude')
        if save:
            filename = resolve_logname(suffix='png')
            fig.savefig(filename)
            logger.info('saved figure to %s', filename)
        if show:
            plt.show()
        return ax

    def _draw_rocket(self, ax, pose, alpha=1.0):
        rotated = np.clip(
            rotate(self.rocket_img, degrees(pose[2]), reshape=True),
            0.0,
            1.0,
        )
        length, width, _ = rotated.shape
        ax.imshow(
            rotated,
            extent=[
                pose[1] - width * self.rocket_scale,
                pose[1] + width * self.rocket_scale,
                1.0-pose[0] - length * self.rocket_scale,
                1.0-pose[0] + length * self.rocket_scale,
            ],
            alpha=alpha,
            zorder=3,
        )

    def _draw_ship(self, ax, pose, alpha=1.0):
        length, width, _ = self.ship_img.shape
        ax.imshow(
            self.ship_img,
            extent=[
                pose[0] - width * self.ship_scale,
                pose[0] + width * self.ship_scale,
                -length * self.ship_scale,
                length * self.ship_scale,
            ],
            alpha=alpha,
            zorder=2,
        )

    def animate(self, u, x, show=True, save_gif=False, save_snapshots=False):
        """Animate the rocket and ship trajectories.

        Args:
            u: (m,N,T)
            x: (n,N,T+1)
        """
        gc = self.config
        if (not show) and (not save_gif) and (not save_snapshots):
            return
        assert u.shape == (gc.m, gc.N, gc.T)
        assert x.shape == (gc.n, gc.N, gc.T + 1)

        fig, ax = plt.subplots()
        ax.set_facecolor((54 / 255, 69 / 255, 79 / 255))
        ax.hlines(y=0, xmin=self.visual_x_lim[0], xmax=self.visual_x_lim[1],
                  colors='white', linewidth=1.5)
        rocket_trail, = ax.plot([], [], '*-', color='tab:orange')
        ship_trail, = ax.plot([], [], '*-', color='tab:blue')

        def update(frame):
            for image in list(ax.images):
                image.remove()
            rocket_trail.set_data(x[1, 0, :frame + 1], -x[0, 0, :frame + 1])
            ship_trail.set_data(x[0, 1, :frame + 1], np.zeros(frame + 1))
            self._draw_ship(ax, x[:, 1, frame])
            self._draw_rocket(ax, x[:, 0, frame])
            return [rocket_trail, ship_trail]

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(*self.visual_x_lim)
        ax.set_ylim(*self.visual_y_lim)
        anim = FuncAnimation(fig, update, frames=gc.T + 1, blit=False)

        folder = os.path.join(BASEDIR, 'outputs', 'gifs')
        os.makedirs(folder, exist_ok=True)
        gif_filename = os.path.join(folder, 'rocket_landing.gif')
        if save_gif:
            anim.save(gif_filename, writer='pillow')
            logger.info('Gif saved to %s', gif_filename)
        if show:
            plt.show()
        if show and save_snapshots:
            logger.error(
                'When show and save_snapshots are both on,'
                ' matplotlib has weird problems, do one at a time')
        if save_snapshots:
            from PIL import Image  # pylint: disable=import-outside-toplevel
            folder = os.path.join(BASEDIR, 'outputs', 'pics')
            os.makedirs(folder, exist_ok=True)
            for frame, name in [(0, 'initial'), (gc.T // 2, 'middle'), (gc.T, 'final')]:
                update(frame)
                fig.canvas.draw()
                image = Image.frombytes('RGB', fig.canvas.get_width_height(),
                                        fig.canvas.tostring_rgb())
                filename = os.path.join(folder, f'rocket_landing_{name}.png')
                image.save(filename)
                logger.info('saved snapshots to %s', filename)

    def _state_cost(self, x_k, i_onehot):
        target_x_ref = self.config.get_param('target_x_ref')
        q_rocket_diag = self.config.get_param('q_rocket_diag')
        q_ship_diag = self.config.get_param('q_ship_diag')
        q_dock_diag = self.config.get_param('q_dock_diag')

        is_rocket = i_onehot[0, 0]
        is_ship = i_onehot[1, 0]
        x_k_i = x_k @ i_onehot
        dx_i = x_k_i - target_x_ref @ i_onehot
        Q_i = self._sparse_diag(is_rocket * q_rocket_diag + is_ship * q_ship_diag)

        dock_delta = (
            self._rocket_dock_selector() @ x_k[:, 0]
            - self._ship_dock_selector() @ x_k[:, 1]
        )
        Q_dock = self._sparse_diag(q_dock_diag)
        return dx_i.T @ Q_i @ dx_i + dock_delta.T @ Q_dock @ dock_delta

    def J(self, x_k, u_k_i, i_onehot):
        """Stage cost for one agent."""
        assert x_k.shape == (self.config.n, self.config.N)
        assert u_k_i.shape == (self.config.m, 1)
        assert i_onehot.shape == (self.config.N, 1)

        r_rocket_diag = self.config.get_param('r_rocket_diag')
        r_ship_diag = self.config.get_param('r_ship_diag')
        is_rocket = i_onehot[0, 0]
        is_ship = i_onehot[1, 0]
        R_i = self._sparse_diag(is_rocket * r_rocket_diag + is_ship * r_ship_diag)
        return self._state_cost(x_k, i_onehot) + u_k_i.T @ R_i @ u_k_i

    def Jfi(self, x_T, i_onehot):
        """Final cost."""
        terminal_weight = self.config.get_param('terminal_weight')
        return terminal_weight * self._state_cost(x_T, i_onehot)

    def f(self, x_k_i, u_k_i, i_onehot, context_i_k):
        """Dynamics x_{t+1} = f(x_t, u_t, i)."""
        del context_i_k
        assert x_k_i.shape == (self.config.n, 1)
        assert u_k_i.shape == (self.config.m, 1)
        assert i_onehot.shape == (self.config.N, 1)

        dt = self.config.get_param('dt')
        theta = x_k_i[2, 0]
        I = 1.0
        L = 1.0
        rocket_dx = cas.vertcat(
            x_k_i[3, 0],
            x_k_i[4, 0],
            x_k_i[5, 0],
            u_k_i[0, 0] * cas.cos(theta) - u_k_i[1, 0] * cas.sin(theta) + 1.0,
            u_k_i[1, 0] * cas.cos(theta) + u_k_i[0, 0] * cas.sin(theta),
            u_k_i[1, 0] * L / I,
        )
        ship_dx = cas.vertcat(
            x_k_i[1, 0],
            u_k_i[0, 0],
            0,
            0,
            0,
            0,
        )

        is_rocket = i_onehot[0, 0]
        is_ship = i_onehot[1, 0]
        dx = is_rocket * rocket_dx + is_ship * ship_dx
        return x_k_i + dx * dt

    def h(self, x, u, context):
        """Construct all inequality constraints h(x,u) <= 0."""
        del context
        gc = self.config
        h_rows = []
        eye = cas.SX.eye(gc.N)
        if gc.variational_gne:
            for k in range(1, gc.T + 1):
                xk = cas.reshape(x[:, k - 1], gc.n, gc.N)
                uk = cas.reshape(u[:, k - 1], gc.m, gc.N)
                for i in range(gc.N):
                    h_rows.append(self.agent_h(xk[:, i], uk[:, i], eye[:, i]))
            h_vec = cas.vertcat(*h_rows)
            assert h_vec.shape == (gc.n_h, 1)
            return h_vec

        for agent_idx in range(gc.N):
            hi_rows = []
            for k in range(1, gc.T + 1):
                xk = cas.reshape(x[:, k - 1], gc.n, gc.N)
                uk = cas.reshape(u[:, k - 1], gc.m, gc.N)
                for i in range(gc.N):
                    if agent_idx == i:
                        hi_rows.append(self.agent_h(xk[:, i], uk[:, i], eye[:, i]))
                    else:
                        hi_rows.append(-cas.DM.ones(gc.constraints_per_agent, 1))
            h_rows.append(cas.vertcat(*hi_rows))

        h_vec = cas.horzcat(*h_rows)
        assert h_vec.shape == (gc.n_h, gc.N)
        return h_vec

    def agent_h(self, x_i, u_i, i_onehot):
        """Per-agent state and control bounds."""
        rocket_h = cas.vertcat(self.rocket_state_h(x_i), self.rocket_control_h(u_i))
        ship_h = cas.vertcat(self.ship_state_h(x_i), self.ship_control_h(u_i))
        is_rocket = i_onehot[0, 0]
        is_ship = i_onehot[1, 0]
        h_val = is_rocket * rocket_h + is_ship * ship_h
        assert h_val.shape == (self.config.constraints_per_agent, 1)
        return h_val

    def rocket_state_h(self, x_i):
        """Rocket altitude and attitude bounds."""
        height_scale = self.config.get_param('height_scale')
        max_tilt = self.config.get_param('max_tilt')
        return cas.vertcat(
            x_i[0, 0] / height_scale,
            (x_i[2, 0] - max_tilt) / max_tilt,
            (-x_i[2, 0] - max_tilt) / max_tilt,
        )

    def ship_state_h(self, x_i):
        """Ship deck position and speed bounds."""
        ship_y_max = self.config.get_param('ship_y_max')
        ship_speed_max = self.config.get_param('ship_speed_max')
        return cas.vertcat(
            (x_i[0, 0] - ship_y_max) / ship_y_max,
            (-x_i[0, 0] - ship_y_max) / ship_y_max,
            (x_i[1, 0] / ship_speed_max)**2 - 1.0,
        )

    def rocket_control_h(self, u_i):
        """Rocket thrust bounds."""
        main_thrust_max = self.config.get_param('main_thrust_max')
        side_thrust_max = self.config.get_param('side_thrust_max')
        return cas.vertcat(
            (u_i[0, 0] - main_thrust_max) / main_thrust_max,
            (-u_i[0, 0] - main_thrust_max) / main_thrust_max,
            (u_i[1, 0] - side_thrust_max) / side_thrust_max,
            (-u_i[1, 0] - side_thrust_max) / side_thrust_max,
        )

    def ship_control_h(self, u_i):
        """Ship acceleration and unused-control bounds."""
        ship_accel_max = self.config.get_param('ship_accel_max')
        unused_ship_control_max = self.config.get_param('unused_ship_control_max')
        return cas.vertcat(
            (u_i[0, 0] - ship_accel_max) / ship_accel_max,
            (-u_i[0, 0] - ship_accel_max) / ship_accel_max,
            (u_i[1, 0] - unused_ship_control_max) / unused_ship_control_max,
            (-u_i[1, 0] - unused_ship_control_max) / unused_ship_control_max,
        )

    def collision_h(self, x_i, x_j):
        """Compatibility hook for solvers that construct a collision helper."""
        del x_i, x_j
        return cas.DM(-1)

    def inspect_h(self, u, x=None, solver=None):
        """Inspect positive residuals by constraint family."""
        gc = self.config
        int_param_dm = cas.DM(gc.get_int_param_np())
        double_param_dm = cas.DM(gc.get_double_param_np())
        params_dm = [int_param_dm, double_param_dm]
        u = cas.DM(u.reshape(gc.m * gc.N, gc.T, order='F'))
        if x is None:
            x = solver.rollout_casadi(gc.x0, u, *params_dm)
        else:
            x = cas.DM(x.reshape(gc.n * gc.N, gc.T, order='F'))
        context_dm = solver.get_full_context_casadi(x)
        h_val = np.asarray(solver.h_casadi(x, u, context_dm, *params_dm))
        h_val = h_val.reshape(gc.n_h, 1 if gc.variational_gne else gc.N, order='F')
        h_pos_res = np.sum(np.clip(h_val, a_min=0, a_max=None)**2)
        logger.info('h_pos_res=%s', h_pos_res)


def create_random_game(horizon=20, variational_gne=False):
    """Create a default RocketLandingCasadi instance.

    The function name mirrors the other CasADi games even though this setup is
    deterministic.
    """
    config = RocketLandingCasadiConfig(
        T=horizon,
        variational_gne=variational_gne,
    )
    return RocketLandingCasadi(config)
