"""Two-car drift game, CasADi version."""
import os
import logging
from dataclasses import dataclass
from math import degrees
from typing import Any

import casadi as cas
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.ndimage import rotate

from rd3g.core.casadi_game import CasadiGame, CasadiGameConfig
from rd3g.utilities.util import BASEDIR, resolve_logname

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass(frozen=False)
class CarDriftCasadiConfig(CasadiGameConfig):
    """Configuration for a two-car dynamic bicycle drift game.

    State ordering is x_i = [s, n, mu, vx, vy, r, theta, beta_r].
    Control ordering is u_i = [dtheta, dbeta_r].
    """
    T: int = 40
    dt: float = 0.05
    N: int = 2
    n: int = 8
    m: int = 2
    n_h: int = 0
    """Total number of canonical inequality constraints."""
    n_c: int = 0
    """Dimension of context variable per agent per stage; unused here."""
    variational_gne: bool = False
    """If True, use one shared multiplier per canonical constraint."""
    constraints_per_agent: int = 12

    # Dynamic bicycle parameters, matching the old example's simple model.
    mass: float = 1.0
    Iz: float = 1.0
    lf: float = 1.0
    lr: float = 1.0
    tire_force_max: float = 0.174
    front_slip_stiffness: float = 1.0
    skidpad_radius: float = 8.0
    slip_sign_smoothing: float = 0.05

    # Reference drift and formation.
    ds_ref: float = 2.0
    mu_ref: float = float(np.deg2rad(17.0))
    vx_ref: float = 1.5148566269
    vy_ref: float = -0.4631381489
    r_ref: float = 0.1980091359
    theta_ref: float = -0.02324605
    beta_r_ref: float = 1.03908555
    formation_weight: float = 0.2
    relative_mu_weight: float = 0.2
    terminal_weight: float = 5.0

    # Normalized state/control bounds.
    n_max: float = 1.5
    vx_min: float = 0.2
    vx_max: float = 3.0
    theta_max: float = 1.2
    beta_r_max: float = 1.4
    dtheta_max: float = 1.5
    dbeta_r_max: float = 1.5

    x0: Any = None
    """Initial state for all agents, dim: (n,N)."""
    target_x_ref: Any = None
    """Reference state for all agents, dim: (n,N)."""
    q_diag: Any = None
    """Sparse state-cost diagonal, dim: (n,1)."""
    r_diag: Any = None
    """Sparse control-cost diagonal, dim: (m,1)."""

    def __post_init__(self):
        if self.N != 2:
            raise ValueError('CarDriftCasadi currently models exactly two cars')

        if self.x0 is None:
            x0_single = np.array([
                0.0,
                0.0,
                self.mu_ref,
                self.vx_ref,
                self.vy_ref,
                self.r_ref,
                self.theta_ref,
                self.beta_r_ref,
            ], dtype=float)
            x1_single = x0_single.copy()
            x1_single[0] = self.ds_ref
            self.x0 = np.vstack([x0_single, x1_single]).T.copy(order='F')

        if self.target_x_ref is None:
            self.target_x_ref = np.array(self.x0, dtype=float, order='F')

        if self.q_diag is None:
            self.q_diag = np.array(
                [0.0, 0.3, 1.0, 0.1, 0.0, 0.0, 0.0, 0.0],
                dtype=float,
            ).reshape(self.n, 1, order='F')

        if self.r_diag is None:
            self.r_diag = (1e-2 * np.ones((self.m, 1), dtype=float)).copy(order='F')

        if self.n_h == 0:
            self.n_h = self.constraints_per_agent * self.N * self.T

        assert self.x0.shape == (self.n, self.N), (
            'Incorrect self.x0 dimension, '
            f'should be {(self.n, self.N)}, but got {self.x0.shape}')
        assert self.target_x_ref.shape == (self.n, self.N)
        assert self.q_diag.shape == (self.n, 1)
        assert self.r_diag.shape == (self.m, 1)
        return super().__post_init__()


class CarDriftCasadi(CasadiGame):
    """Two dynamic-bicycle cars drifting on a constant-curvature skidpad."""

    def __init__(self, config: CarDriftCasadiConfig):
        super().__init__(config)
        radius = self.config.skidpad_radius
        self.visual_x_lim = [-2.0, 2.0 * radius + 2.0]
        self.visual_y_lim = [-2.0, 2.0 * radius + 2.0]
        self.car_scale = 0.004 / 2 * 0.85
        color_names = ['orange', 'blue']
        self.car_img_vec = [
            mpimg.imread(
                os.path.join(BASEDIR, 'rd3g', 'resources', f'porsche_{color}.png'))
            for color in color_names
        ]

    @staticmethod
    def _sparse_selector(rows, cols, shape):
        """Return a sparse CasADi selector matrix with ones at rows/cols."""
        sparsity = cas.Sparsity.triplet(shape[0], shape[1], rows, cols)
        return cas.SX(sparsity, [1.0] * len(rows))

    @staticmethod
    def _sparse_diag(diag_values):
        """Return a sparse diagonal matrix from a CasADi vector."""
        return cas.diag(diag_values)

    def _agent_swap(self):
        return self._sparse_selector([0, 1], [1, 0], (self.config.N, self.config.N))

    def _s_selector(self):
        return self._sparse_selector([0], [0], (1, self.config.n))

    def _mu_selector(self):
        return self._sparse_selector([0], [2], (1, self.config.n))

    def initial_control_guess(self):
        """Return a zero-rate drift guess, shape (m*N,T)."""
        gc = self.config
        return np.zeros((gc.m * gc.N, gc.T), dtype=float, order='F')

    def _frenet_to_cart_np(self, state):
        radius = self.config.skidpad_radius
        s_val, n_val, mu_val = state[:3]
        angle = s_val / radius
        ref = np.array([radius * np.sin(angle),
                        radius * (1.0 - np.cos(angle))])
        normal = np.array([-np.sin(angle), np.cos(angle)])
        pos = ref + n_val * normal
        heading = angle + mu_val
        return pos[0], pos[1], heading

    def _draw_skidpad(self, ax, s_max):
        radius = self.config.skidpad_radius
        s_vec = np.linspace(0.0, max(s_max, 2.0 * np.pi * radius), 400)
        angle = s_vec / radius
        ax.plot(radius * np.sin(angle),
                radius * (1.0 - np.cos(angle)),
                color='white',
                linewidth=1.2,
                alpha=0.8)

    def _draw_car(self, ax, pose, car_idx, alpha=1.0):
        rotated_car_img = np.clip(
            rotate(self.car_img_vec[car_idx % len(self.car_img_vec)],
                   degrees(pose[2]),
                   reshape=True),
            0.0,
            1.0,
        )
        length, width, _ = rotated_car_img.shape
        ax.imshow(
            rotated_car_img,
            extent=[
                pose[0] - width * self.car_scale,
                pose[0] + width * self.car_scale,
                pose[1] - length * self.car_scale,
                pose[1] + length * self.car_scale,
            ],
            alpha=alpha,
            zorder=3,
        )

    def visualize(self, u, x, show=True, save=False):
        """Visualize drift trajectories and initial/final car poses.

        Args:
            u: (m,N,T)
            x: (n,N,T+1)
        """
        del u
        gc = self.config
        if (not show) and (not save):
            return None
        assert x.shape == (gc.n, gc.N, gc.T + 1)

        fig, ax = plt.subplots()
        ax.set_facecolor((54 / 255, 69 / 255, 79 / 255))
        self._draw_skidpad(ax, float(np.max(x[0, :, :])))

        for i in range(gc.N):
            poses = np.array([self._frenet_to_cart_np(x[:, i, k])
                              for k in range(gc.T + 1)])
            ax.plot(poses[:, 0], poses[:, 1], '*-', zorder=2)
            self._draw_car(ax, poses[0], i, alpha=0.55)
            self._draw_car(ax, poses[-1], i, alpha=1.0)

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(*self.visual_x_lim)
        ax.set_ylim(*self.visual_y_lim)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        if save:
            filename = resolve_logname(suffix='png')
            fig.savefig(filename)
            logger.info('saved figure to %s', filename)
        if show:
            plt.show()
        return ax

    def animate(self, u, x, show=True, save_gif=False, save_snapshots=False):
        """Animate the drift trajectories.

        Args:
            u: (m,N,T)
            x: (n,N,T+1)
        """
        del u
        gc = self.config
        if (not show) and (not save_gif) and (not save_snapshots):
            return
        assert x.shape == (gc.n, gc.N, gc.T + 1)

        car_pose_vec = []
        for k in range(gc.T + 1):
            car_pose_vec.append([
                self._frenet_to_cart_np(x[:, i, k]) for i in range(gc.N)
            ])

        fig, ax = plt.subplots()
        ax.set_facecolor((54 / 255, 69 / 255, 79 / 255))
        self._draw_skidpad(ax, float(np.max(x[0, :, :])))
        trails = [ax.plot([], [], '*-')[0] for _ in range(gc.N)]
        im_vec = []

        for i in range(gc.N):
            pose = car_pose_vec[0][i]
            rotated_car_img = np.clip(
                rotate(self.car_img_vec[i % len(self.car_img_vec)],
                       degrees(pose[2]),
                       reshape=True),
                0.0,
                1.0,
            )
            length, width, _ = rotated_car_img.shape
            im = ax.imshow(
                rotated_car_img,
                extent=[
                    pose[0] - width * self.car_scale,
                    pose[0] + width * self.car_scale,
                    pose[1] - length * self.car_scale,
                    pose[1] + length * self.car_scale,
                ],
                zorder=3,
            )
            im_vec.append(im)

        def update(frame):
            for i in range(gc.N):
                pose = car_pose_vec[frame][i]
                trail = np.array(car_pose_vec[:frame + 1], dtype=float)[:, i, :]
                trails[i].set_data(trail[:, 0], trail[:, 1])
                rotated_car_img = np.clip(
                    rotate(self.car_img_vec[i % len(self.car_img_vec)],
                           degrees(pose[2]),
                           reshape=True),
                    0.0,
                    1.0,
                )
                length, width, _ = rotated_car_img.shape
                im_vec[i].set_data(rotated_car_img)
                im_vec[i].set_extent(
                    (pose[0] - width * self.car_scale,
                     pose[0] + width * self.car_scale,
                     pose[1] - length * self.car_scale,
                     pose[1] + length * self.car_scale))
            return trails + im_vec

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(*self.visual_x_lim)
        ax.set_ylim(*self.visual_y_lim)
        anim = FuncAnimation(fig, update, frames=gc.T + 1, blit=False)

        folder = os.path.join(BASEDIR, 'outputs', 'gifs')
        os.makedirs(folder, exist_ok=True)
        gif_filename = os.path.join(folder, 'drift_2car.gif')
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
                filename = os.path.join(folder, f'drift_2car_{name}.png')
                image.save(filename)
                logger.info('saved snapshots to %s', filename)
        return None

    def _state_cost(self, x_k, i_onehot):
        target_x_ref = self.config.get_param('target_x_ref')
        q_diag = self.config.get_param('q_diag')
        ds_ref = self.config.get_param('ds_ref')
        formation_weight = self.config.get_param('formation_weight')
        relative_mu_weight = self.config.get_param('relative_mu_weight')

        x_i = x_k @ i_onehot
        other_onehot = self._agent_swap() @ i_onehot
        x_j = x_k @ other_onehot

        dx_i = x_i - target_x_ref @ i_onehot
        Q_i = self._sparse_diag(q_diag)
        state_cost = dx_i.T @ Q_i @ dx_i

        formation_sign = i_onehot[0, 0] - i_onehot[1, 0]
        ds_error = formation_sign * (self._s_selector() @ (x_j - x_i)) - ds_ref
        dmu_error = self._mu_selector() @ (x_i - x_j)
        return (state_cost + formation_weight * ds_error.T @ ds_error
                + relative_mu_weight * dmu_error.T @ dmu_error)

    def J(self, x_k, u_k_i, i_onehot):
        """Stage cost for an agent."""
        assert x_k.shape == (self.config.n, self.config.N)
        assert u_k_i.shape == (self.config.m, 1)
        assert i_onehot.shape == (self.config.N, 1)

        r_diag = self.config.get_param('r_diag')
        R_i = self._sparse_diag(r_diag)
        return self._state_cost(x_k, i_onehot) + u_k_i.T @ R_i @ u_k_i

    def Jfi(self, x_T, i_onehot):
        """Final cost."""
        terminal_weight = self.config.get_param('terminal_weight')
        return terminal_weight * self._state_cost(x_T, i_onehot)

    def f(self, x_k_i, u_k_i, i_onehot, context_i_k):
        """Dynamic bicycle model in a constant-curvature Frenet frame."""
        del i_onehot, context_i_k
        assert x_k_i.shape == (self.config.n, 1)
        assert u_k_i.shape == (self.config.m, 1)

        s_val = x_k_i[0, 0]
        n_val = x_k_i[1, 0]
        mu_val = x_k_i[2, 0]
        vx = x_k_i[3, 0]
        vy = x_k_i[4, 0]
        r_val = x_k_i[5, 0]
        theta = x_k_i[6, 0]
        beta_r = x_k_i[7, 0]
        dtheta = u_k_i[0, 0]
        dbeta_r = u_k_i[1, 0]
        del s_val

        mass = self.config.get_param('mass')
        Iz = self.config.get_param('Iz')
        lf = self.config.get_param('lf')
        lr = self.config.get_param('lr')
        tire_force_max = self.config.get_param('tire_force_max')
        front_slip_stiffness = self.config.get_param('front_slip_stiffness')
        skidpad_radius = self.config.get_param('skidpad_radius')
        slip_sign_smoothing = self.config.get_param('slip_sign_smoothing')
        dt = self.config.get_param('dt')

        curvature = 1.0 / skidpad_radius
        vy_sign = cas.tanh(vy / slip_sign_smoothing)
        fry = tire_force_max * cas.sin(beta_r) * (-vy_sign)
        frx = tire_force_max * cas.cos(beta_r)
        beta_f = -cas.atan((vy + lf * r_val) / vx) + theta
        ffy = front_slip_stiffness * beta_f

        ds = (vx * cas.cos(mu_val) - vy * cas.sin(mu_val)) / (
            1.0 - n_val * curvature)
        dn = vx * cas.sin(mu_val) + vy * cas.cos(mu_val)
        dmu = r_val - curvature * ds
        dvx = (frx - fry * cas.sin(theta) + mass * vy * r_val) / mass
        dvy = (fry + ffy * cas.cos(theta) - mass * vx * r_val) / mass
        dr = (ffy * cas.cos(theta) * lf - fry * lr) / Iz

        dx = cas.vertcat(ds, dn, dmu, dvx, dvy, dr, dtheta, dbeta_r)
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
        del i_onehot
        h_val = cas.vertcat(self.state_h(x_i), self.control_h(u_i))
        assert h_val.shape == (self.config.constraints_per_agent, 1)
        return h_val

    def state_h(self, x_i):
        """State bounds, normalized so feasible values satisfy h <= 0."""
        n_max = self.config.get_param('n_max')
        vx_min = self.config.get_param('vx_min')
        vx_max = self.config.get_param('vx_max')
        theta_max = self.config.get_param('theta_max')
        beta_r_max = self.config.get_param('beta_r_max')
        return cas.vertcat(
            (x_i[1, 0] - n_max) / n_max,
            (-x_i[1, 0] - n_max) / n_max,
            (vx_min - x_i[3, 0]) / vx_min,
            (x_i[3, 0] - vx_max) / vx_max,
            (x_i[6, 0] - theta_max) / theta_max,
            (-x_i[6, 0] - theta_max) / theta_max,
            (x_i[7, 0] - beta_r_max) / beta_r_max,
            (-x_i[7, 0] - beta_r_max) / beta_r_max,
        )

    def control_h(self, u_i):
        """Control-rate bounds, normalized so feasible values satisfy h <= 0."""
        dtheta_max = self.config.get_param('dtheta_max')
        dbeta_r_max = self.config.get_param('dbeta_r_max')
        return cas.vertcat(
            (u_i[0, 0] - dtheta_max) / dtheta_max,
            (-u_i[0, 0] - dtheta_max) / dtheta_max,
            (u_i[1, 0] - dbeta_r_max) / dbeta_r_max,
            (-u_i[1, 0] - dbeta_r_max) / dbeta_r_max,
        )

    def collision_h(self, x_i, x_j):
        """Compatibility hook for solvers that construct a collision helper."""
        del x_i, x_j
        return cas.DM(-1)


def create_random_game(car_count=2, horizon=40, variational_gne=False):
    """Create a deterministic two-car drift game.

    The function name mirrors the other CasADi game factories.
    """
    if car_count != 2:
        raise ValueError('CarDriftCasadi currently supports car_count=2')
    config = CarDriftCasadiConfig(
        T=horizon,
        N=car_count,
        variational_gne=variational_gne,
    )
    return CarDriftCasadi(config)
