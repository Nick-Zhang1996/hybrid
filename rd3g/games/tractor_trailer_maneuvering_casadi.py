"""Multi tractor-trailer maneuvering game, CasADi version."""
import os
import logging
from dataclasses import dataclass, field
from typing import Any
from math import radians

import casadi as cas
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon

from rd3g.core.casadi_game import CasadiGame, CasadiGameConfig
from rd3g.utilities.util import BASEDIR, resolve_logname

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass(frozen=True)
class Obstacle:
    """Convex rotated rectangular obstacle.

    x, y: rectangle center.
    w, h: width along the local x/y axes before rotation.
    angle: counter-clockwise rotation of the local x-axis.
    """
    x: float
    y: float
    w: float
    h: float
    angle: float = 0.0


def obstacle_arrays(obstacles):
    """Convert Obstacle objects to CasADi-friendly config arrays."""
    if obstacles is None:
        obstacles = []
    obstacle_vec = list(obstacles)
    if not obstacle_vec:
        return (
            np.zeros((2, 0), dtype=float, order='F'),
            np.zeros((2, 0), dtype=float, order='F'),
            np.zeros((0, 1), dtype=float, order='F'),
        )
    positions = np.array(
        [[obs.x for obs in obstacle_vec],
         [obs.y for obs in obstacle_vec]],
        dtype=float,
        order='F',
    )
    sizes = np.array(
        [[obs.w for obs in obstacle_vec],
         [obs.h for obs in obstacle_vec]],
        dtype=float,
        order='F',
    )
    angles = np.array(
        [[obs.angle] for obs in obstacle_vec],
        dtype=float,
        order='F',
    )
    return positions, sizes, angles


def obstacle_kwargs(obstacles):
    """Return keyword args for TractorTrailerManeuveringCasadiConfig."""
    positions, sizes, angles = obstacle_arrays(obstacles)
    return {
        'obstacle_positions': positions,
        'obstacle_sizes': sizes,
        'obstacle_angles': angles,
    }


def _default_obstacles():
    """Left and right corridor boundaries as large obstacle boxes."""
    return [
        Obstacle(x=-1.0, y=3.4, w=18.0, h=2.0, angle=0.0),
        Obstacle(x=-1.0, y=-3.4, w=18.0, h=2.0, angle=0.0),
        # Obstacle(x=-1.0, y=-3.4, w=5.0, h=0.5, angle=radians(10)),
    ]


def _default_obstacle_positions():
    return obstacle_arrays(_default_obstacles())[0]


def _default_obstacle_sizes():
    return obstacle_arrays(_default_obstacles())[1]


def _default_obstacle_angles():
    return obstacle_arrays(_default_obstacles())[2]


def _default_side_by_side_states(agent_count):
    """Return side-by-side start and target states in a straight corridor."""
    x0_vec = []
    target_vec = []
    for i in range(agent_count):
        queue_idx = i // 2
        lane_y = -0.9 if i % 2 == 0 else 0.9
        start_x = -3.0 - 1.2 * queue_idx
        target_x = 1.0 - 1.2 * queue_idx
        x0_vec.append(np.array([start_x, lane_y, 0.9, 0.0, 0.0]))
        target_vec.append(np.array([target_x, lane_y, 0.9, 0.0, 0.0]))
    return (
        np.vstack(x0_vec).T.copy(order='F'),
        np.vstack(target_vec).T.copy(order='F'),
    )


def _as_2_by_k(value, name):
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f'{name} must be a 2D array')
    if arr.shape[0] == 2:
        return arr.copy(order='F')
    if arr.shape[1] == 2:
        return arr.T.copy(order='F')
    raise ValueError(f'{name} must have shape (2,K) or (K,2), got {arr.shape}')


def _as_k_by_1(value, name, count):
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim == 2 and arr.shape[0] == 1 and arr.shape[1] == count:
        arr = arr.T
    if arr.shape != (count, 1):
        raise ValueError(f'{name} must have shape ({count},1), got {arr.shape}')
    return arr.copy(order='F')


@dataclass(frozen=False)
class TractorTrailerManeuveringCasadiConfig(CasadiGameConfig):
    """Configuration for tight-space tractor-trailer maneuvering.

    State ordering is x_i = [x, y, v, psi, phi].
        x, y: hitch / tractor rear-axle position.
        v: signed tractor longitudinal speed.
        psi: tractor heading.
        phi: articulation angle, tractor heading minus trailer heading.

    Control ordering is u_i = [a, delta], longitudinal acceleration and
    Ackermann steering angle.
    """
    T: int = 30
    dt: float = 0.15
    N: int = 2
    n: int = 5
    m: int = 2
    n_h: int = 0
    """Total number of canonical inequality constraints."""
    n_c: int = 0
    """Dimension of context variable per agent per stage; unused here."""
    variational_gne: bool = False
    """If True, use one shared multiplier per canonical constraint."""

    # Tractor and trailer dimensions.
    tractor_length: float = 2.2
    tractor_width: float = 1.2
    tractor_wheelbase: float = 2.3
    trailer_length: float = 4.0
    trailer_width: float = 1.2
    trailer_hitch_length: float = 3.2
    tractor_circle_count: int = 2
    trailer_circle_count: int = 3
    collision_buffer: float = 0.05

    # Bounds.
    v_min: float = -0.8
    v_max: float = 1.0
    a_max: float = 1.2
    steering_max: float = float(np.deg2rad(32.0))
    articulation_max: float = float(np.deg2rad(70.0))

    # Smooth obstacle avoidance. obstacle_softmax_gain has units 1/m.
    obstacle_count: int = 2
    obstacle_positions: Any = field(default_factory=_default_obstacle_positions)
    """Obstacle centers, dim: (2,obstacle_count)."""
    obstacle_sizes: Any = field(default_factory=_default_obstacle_sizes)
    """Obstacle length/width, dim: (2,obstacle_count)."""
    obstacle_angles: Any = field(default_factory=_default_obstacle_angles)
    """Obstacle headings, dim: (obstacle_count,1)."""
    obstacle_clearance: float = 0.05
    obstacle_distance_scale: float = 1.0
    """Distance scale for obstacle constraints after circle inflation."""
    obstacle_softmax_gain: float = 8.0

    # Costs.
    terminal_weight: float = 8.0
    x0: Any = None
    """Initial state for all agents, dim: (n,N)."""
    target_x_ref: Any = None
    """Reference state for all agents, dim: (n,N)."""
    q_diag: Any = None
    """Sparse state-cost diagonal, dim: (n,1)."""
    r_diag: Any = None
    """Sparse control-cost diagonal, dim: (m,1)."""

    def __post_init__(self):
        if self.n != 5 or self.m != 2:
            raise ValueError('TractorTrailerManeuveringCasadi uses n=5 and m=2')
        if self.tractor_circle_count < 1 or self.trailer_circle_count < 1:
            raise ValueError('tractor_circle_count and trailer_circle_count must be positive')

        self.obstacle_positions = _as_2_by_k(
            self.obstacle_positions, 'obstacle_positions')
        self.obstacle_sizes = _as_2_by_k(self.obstacle_sizes, 'obstacle_sizes')
        if self.obstacle_positions.shape[1] != self.obstacle_sizes.shape[1]:
            raise ValueError('obstacle_positions and obstacle_sizes must have the same count')
        self.obstacle_count = self.obstacle_positions.shape[1]
        self.obstacle_angles = _as_k_by_1(
            self.obstacle_angles, 'obstacle_angles', self.obstacle_count)

        if self.x0 is None or self.target_x_ref is None:
            default_x0, default_target = _default_side_by_side_states(self.N)
            if self.x0 is None:
                self.x0 = default_x0
            if self.target_x_ref is None:
                self.target_x_ref = default_target
        self.x0 = np.asarray(self.x0, dtype=float).copy(order='F')
        self.target_x_ref = np.asarray(self.target_x_ref, dtype=float).copy(order='F')

        if self.q_diag is None:
            self.q_diag = np.array([1.0, 1.0, 0.15, 0.45, 0.2],
                                   dtype=float).reshape(self.n, 1, order='F')
        if self.r_diag is None:
            self.r_diag = np.array([0.05, 0.25],
                                   dtype=float).reshape(self.m, 1, order='F')
        self.q_diag = np.asarray(self.q_diag, dtype=float).reshape(
            self.n, 1, order='F')
        self.r_diag = np.asarray(self.r_diag, dtype=float).reshape(
            self.m, 1, order='F')

        body_circle_count = self.tractor_circle_count + self.trailer_circle_count
        pair_count = self.N * (self.N - 1) // 2
        rows_per_vehicle_pair = body_circle_count**2
        # Temporarily disabled: tractor-trailer self-collision constraints.
        self_collision_rows = 0
        obstacle_rows = self.obstacle_count * body_circle_count
        bound_rows = 8
        if self.n_h == 0:
            self.n_h = self.T * (
                pair_count * rows_per_vehicle_pair
                + self.N * (self_collision_rows + obstacle_rows + bound_rows)
            )

        assert self.x0.shape == (self.n, self.N), (
            'Incorrect self.x0 dimension, '
            f'should be {(self.n, self.N)}, but got {self.x0.shape}')
        assert self.target_x_ref.shape == (self.n, self.N)
        assert self.q_diag.shape == (self.n, 1)
        assert self.r_diag.shape == (self.m, 1)
        return super().__post_init__()


class TractorTrailerManeuveringCasadi(CasadiGame):
    """Kinematic Ackermann tractor towing a single-axle trailer."""

    def __init__(self, config: TractorTrailerManeuveringCasadiConfig):
        super().__init__(config)
        self.color_vec = [
            'tab:purple', 'tab:orange', 'tab:red', 'tab:green',
            'tab:blue', 'tab:pink', 'tab:cyan', 'black'
        ]
        self._set_visual_bounds()

    @staticmethod
    def _sparse_diag(diag_values):
        """Return a sparse diagonal matrix from a CasADi vector."""
        return cas.diag(diag_values)

    @property
    def _body_circle_count(self):
        return self.config.tractor_circle_count + self.config.trailer_circle_count

    @property
    def _rows_per_vehicle_pair(self):
        return self._body_circle_count**2

    @property
    def _self_collision_rows(self):
        # Temporarily disabled: tractor-trailer self-collision constraints.
        return 0

    @property
    def _obstacle_rows_per_agent(self):
        return self.config.obstacle_count * self._body_circle_count

    @property
    def _bound_rows_per_agent(self):
        return 8

    @property
    def _rows_per_agent(self):
        return (
            self._self_collision_rows
            + self._obstacle_rows_per_agent
            + self._bound_rows_per_agent
        )

    def initial_control_guess(self):
        """Return a straight-driving control guess, shape (m*N,T)."""
        gc = self.config
        return np.zeros((gc.m * gc.N, gc.T), dtype=float, order='F')

    def _set_visual_bounds(self):
        pts = [self.config.x0[:2, :].T, self.config.target_x_ref[:2, :].T]
        for obs_idx in range(self.config.obstacle_count):
            pts.append(self._obstacle_polygon_np(obs_idx))
        pts = np.vstack(pts)
        margin = max(4.0, self.config.trailer_length)
        self.visual_x_lim = [float(np.min(pts[:, 0]) - margin),
                             float(np.max(pts[:, 0]) + margin)]
        self.visual_y_lim = [float(np.min(pts[:, 1]) - margin),
                             float(np.max(pts[:, 1]) + margin)]

    def _obstacle_polygon_np(self, obstacle_idx):
        center = self.config.obstacle_positions[:, obstacle_idx]
        size = self.config.obstacle_sizes[:, obstacle_idx]
        angle = self.config.obstacle_angles[obstacle_idx, 0]
        long_axis = np.array([np.cos(angle), np.sin(angle)])
        lat_axis = np.array([-np.sin(angle), np.cos(angle)])
        half_l = 0.5 * size[0]
        half_w = 0.5 * size[1]
        return np.vstack([
            center - half_l * long_axis - half_w * lat_axis,
            center + half_l * long_axis - half_w * lat_axis,
            center + half_l * long_axis + half_w * lat_axis,
            center - half_l * long_axis + half_w * lat_axis,
        ])

    def _body_polygon_np(self, x_i, body):
        hitch = x_i[:2]
        psi = x_i[3]
        phi = x_i[4]
        if body == 'tractor':
            heading = psi
            length = self.config.tractor_length
            width = self.config.tractor_width
            offsets = np.array([0.0, length])
        elif body == 'trailer':
            heading = psi - phi
            length = self.config.trailer_length
            width = self.config.trailer_width
            offsets = np.array([0.0, -length])
        else:
            raise ValueError(f'unknown body {body}')
        long_axis = np.array([np.cos(heading), np.sin(heading)])
        lat_axis = np.array([-np.sin(heading), np.cos(heading)])
        return np.vstack([
            hitch + offsets[0] * long_axis - 0.5 * width * lat_axis,
            hitch + offsets[1] * long_axis - 0.5 * width * lat_axis,
            hitch + offsets[1] * long_axis + 0.5 * width * lat_axis,
            hitch + offsets[0] * long_axis + 0.5 * width * lat_axis,
        ])

    def _draw_obstacles(self, ax):
        for obs_idx in range(self.config.obstacle_count):
            polygon = self._obstacle_polygon_np(obs_idx)
            ax.add_patch(Polygon(polygon, closed=True, facecolor='dimgray',
                                 edgecolor='white', linewidth=1.0, zorder=1))

    def _draw_vehicle(self, ax, x_i, color, alpha=1.0):
        tractor = self._body_polygon_np(x_i, 'tractor')
        trailer = self._body_polygon_np(x_i, 'trailer')
        ax.add_patch(Polygon(trailer, closed=True, facecolor=color,
                             edgecolor='black', alpha=0.45 * alpha, zorder=3))
        ax.add_patch(Polygon(tractor, closed=True, facecolor=color,
                             edgecolor='black', alpha=0.7 * alpha, zorder=4))
        ax.plot(x_i[0], x_i[1], 'o', color='black', markersize=3, zorder=5)

    def visualize(self, u, x, show=True, save=False):
        """Visualize trajectories with initial and final tractor-trailer poses.

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
        self._draw_obstacles(ax)

        for i in range(gc.N):
            color = self.color_vec[i % len(self.color_vec)]
            ax.plot(x[0, i, :], x[1, i, :], '*-', color=color, zorder=2)
            ax.plot(gc.target_x_ref[0, i], gc.target_x_ref[1, i],
                    'x', color=color, markersize=8, zorder=6)
            self._draw_vehicle(ax, x[:, i, 0], color, alpha=0.45)
            self._draw_vehicle(ax, x[:, i, -1], color, alpha=1.0)

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(*self.visual_x_lim)
        ax.set_ylim(*self.visual_y_lim)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        if save:
            filename = resolve_logname(suffix='png')
            fig.savefig(filename)
            logger.info('saved figure to %s', filename)
        if show:
            plt.show()
        return ax

    def animate(self, u, x, show=True, save_gif=False, save_snapshots=False):
        """Animate tractor-trailer poses.

        Args:
            u: (m,N,T)
            x: (n,N,T+1)
        """
        del u
        gc = self.config
        if (not show) and (not save_gif) and (not save_snapshots):
            return None
        assert x.shape == (gc.n, gc.N, gc.T + 1)

        fig, ax = plt.subplots()
        ax.set_facecolor((54 / 255, 69 / 255, 79 / 255))
        self._draw_obstacles(ax)

        trails = []
        tractor_patches = []
        trailer_patches = []
        for i in range(gc.N):
            color = self.color_vec[i % len(self.color_vec)]
            trail, = ax.plot([], [], '*-', color=color, zorder=2)
            trails.append(trail)
            trailer_patch = Polygon(self._body_polygon_np(x[:, i, 0], 'trailer'),
                                    closed=True, facecolor=color, edgecolor='black',
                                    alpha=0.45, zorder=3)
            tractor_patch = Polygon(self._body_polygon_np(x[:, i, 0], 'tractor'),
                                    closed=True, facecolor=color, edgecolor='black',
                                    alpha=0.7, zorder=4)
            ax.add_patch(trailer_patch)
            ax.add_patch(tractor_patch)
            trailer_patches.append(trailer_patch)
            tractor_patches.append(tractor_patch)

        def update(frame):
            for i in range(gc.N):
                trails[i].set_data(x[0, i, :frame + 1], x[1, i, :frame + 1])
                trailer_patches[i].set_xy(
                    self._body_polygon_np(x[:, i, frame], 'trailer'))
                tractor_patches[i].set_xy(
                    self._body_polygon_np(x[:, i, frame], 'tractor'))
            return trails + trailer_patches + tractor_patches

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(*self.visual_x_lim)
        ax.set_ylim(*self.visual_y_lim)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        anim = FuncAnimation(fig, update, frames=gc.T + 1, blit=False)

        folder = os.path.join(BASEDIR, 'outputs', 'gifs')
        os.makedirs(folder, exist_ok=True)
        gif_filename = os.path.join(folder, f'trailer_{gc.N}tractor.gif')
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
                filename = os.path.join(folder, f'trailer_{gc.N}tractor_{name}.png')
                image.save(filename)
                logger.info('saved snapshots to %s', filename)
        return None

    def _state_cost(self, x_k, i_onehot):
        target_x_ref = self.config.get_param('target_x_ref')
        q_diag = self.config.get_param('q_diag')
        x_i = x_k @ i_onehot
        dx = x_i - target_x_ref @ i_onehot
        Q_i = self._sparse_diag(q_diag)
        return dx.T @ Q_i @ dx

    def J(self, x_k, u_k_i, i_onehot):
        """Stage cost for one tractor-trailer."""
        assert x_k.shape == (self.config.n, self.config.N)
        assert u_k_i.shape == (self.config.m, 1)
        assert i_onehot.shape == (self.config.N, 1)

        r_diag = self.config.get_param('r_diag')
        R_i = self._sparse_diag(r_diag)
        return self._state_cost(x_k, i_onehot) + u_k_i.T @ R_i @ u_k_i

    def Jfi(self, x_T, i_onehot):
        """Terminal tracking cost."""
        terminal_weight = self.config.get_param('terminal_weight')
        return terminal_weight * self._state_cost(x_T, i_onehot)

    def f(self, x_k_i, u_k_i, i_onehot, context_i_k):
        """Ackermann tractor with one single-axle trailer."""
        del i_onehot, context_i_k
        assert x_k_i.shape == (self.config.n, 1)
        assert u_k_i.shape == (self.config.m, 1)

        v_val = x_k_i[2, 0]
        psi = x_k_i[3, 0]
        phi = x_k_i[4, 0]
        acceleration = u_k_i[0, 0]
        steering = u_k_i[1, 0]

        tractor_wheelbase = self.config.get_param('tractor_wheelbase')
        trailer_hitch_length = self.config.get_param('trailer_hitch_length')
        dt = self.config.get_param('dt')

        dpsi = v_val / tractor_wheelbase * cas.tan(steering)
        dphi = dpsi - v_val / trailer_hitch_length * cas.sin(phi)
        dx = cas.vertcat(
            v_val * cas.cos(psi),
            v_val * cas.sin(psi),
            acceleration,
            dpsi,
            dphi,
        )
        return x_k_i + dx * dt

    def h(self, x, u, context):
        """Construct all smooth inequality constraints h(x,u) <= 0."""
        del context
        gc = self.config
        h_rows = []
        eye = cas.SX.eye(gc.N)

        if gc.variational_gne:
            for k in range(1, gc.T + 1):
                xk = cas.reshape(x[:, k - 1], gc.n, gc.N)
                uk = cas.reshape(u[:, k - 1], gc.m, gc.N)
                for i in range(gc.N):
                    for j in range(i + 1, gc.N):
                        h_rows.append(self.collision_h(xk[:, i], xk[:, j]))
                for i in range(gc.N):
                    h_rows.append(self.agent_h(xk[:, i], uk[:, i], eye[:, i]))
            h_vec = cas.vertcat(*h_rows)
            assert h_vec.shape == (gc.n_h, 1), "n_h must be consistent with h().shape[0]"
            return h_vec

        for agent_idx in range(gc.N):
            hi_rows = []
            for k in range(1, gc.T + 1):
                xk = cas.reshape(x[:, k - 1], gc.n, gc.N)
                uk = cas.reshape(u[:, k - 1], gc.m, gc.N)
                for i in range(gc.N):
                    for j in range(i + 1, gc.N):
                        if agent_idx == i or agent_idx == j:
                            hi_rows.append(self.collision_h(xk[:, i], xk[:, j]))
                        else:
                            hi_rows.append(-cas.DM.ones(self._rows_per_vehicle_pair, 1))
                for i in range(gc.N):
                    if agent_idx == i:
                        hi_rows.append(self.agent_h(xk[:, i], uk[:, i], eye[:, i]))
                    else:
                        hi_rows.append(-cas.DM.ones(self._rows_per_agent, 1))
            h_rows.append(cas.vertcat(*hi_rows))

        h_vec = cas.horzcat(*h_rows)
        assert h_vec.shape == (gc.n_h, gc.N), "n_h must be consistent with h().shape[0]"
        return h_vec

    def agent_h(self, x_i, u_i, i_onehot):
        """Per-agent obstacle, state, and control constraints."""
        del i_onehot
        h_val = cas.vertcat(
            # Temporarily disabled: self.self_collision_h(x_i),
            self.obstacle_h(x_i),
            self.state_h(x_i),
            self.control_h(u_i),
        )
        assert h_val.shape == (self._rows_per_agent, 1)
        return h_val

    def _tractor_circle(self, x_i, circle_idx):
        tractor_length = self.config.get_param('tractor_length')
        tractor_width = self.config.get_param('tractor_width')
        collision_buffer = self.config.get_param('collision_buffer')
        offset = tractor_length * (circle_idx + 0.5) / self.config.tractor_circle_count
        center = self._point_along_heading(x_i[:2], x_i[3, 0], offset)
        radius = 0.5 * tractor_width + collision_buffer
        return center, radius

    def _trailer_circle(self, x_i, circle_idx):
        trailer_length = self.config.get_param('trailer_length')
        trailer_width = self.config.get_param('trailer_width')
        collision_buffer = self.config.get_param('collision_buffer')
        trailer_heading = x_i[3, 0] - x_i[4, 0]
        offset = -trailer_length * (circle_idx + 0.5) / self.config.trailer_circle_count
        center = self._point_along_heading(x_i[:2], trailer_heading, offset)
        radius = 0.5 * trailer_width + collision_buffer
        return center, radius

    @staticmethod
    def _point_along_heading(origin, heading, offset):
        return origin + offset * cas.vertcat(cas.cos(heading), cas.sin(heading))

    def _body_circles(self, x_i):
        circles = []
        for idx in range(self.config.tractor_circle_count):
            circles.append(self._tractor_circle(x_i, idx))
        for idx in range(self.config.trailer_circle_count):
            circles.append(self._trailer_circle(x_i, idx))
        return circles

    @staticmethod
    def _circle_pair_h(circle_i, circle_j):
        center_i, radius_i = circle_i
        center_j, radius_j = circle_j
        delta = center_i - center_j
        min_dist = radius_i + radius_j
        return min_dist**2 - delta.T @ delta

    def collision_h(self, x_i, x_j):
        """Pairwise smooth body-disk collision constraints between two vehicles."""
        assert x_i.shape == (self.config.n, 1)
        assert x_j.shape == (self.config.n, 1)
        rows = []
        for circle_i in self._body_circles(x_i):
            for circle_j in self._body_circles(x_j):
                rows.append(self._circle_pair_h(circle_i, circle_j))
        h_val = cas.vertcat(*rows)
        assert h_val.shape == (self._rows_per_vehicle_pair, 1)
        return h_val

    def self_collision_h(self, x_i):
        """Smooth tractor-body to trailer-body constraints for one vehicle."""
        rows = []
        tractor_circles = [
            self._tractor_circle(x_i, idx)
            for idx in range(self.config.tractor_circle_count)
        ]
        trailer_circles = [
            self._trailer_circle(x_i, idx)
            for idx in range(self.config.trailer_circle_count)
        ]
        for tractor_circle in tractor_circles:
            for trailer_circle in trailer_circles:
                rows.append(self._circle_pair_h(tractor_circle, trailer_circle))
        h_val = cas.vertcat(*rows)
        assert h_val.shape == (self._self_collision_rows, 1)
        return h_val

    def obstacle_h(self, x_i):
        """Smooth outside-of-rotated-rectangle constraints for each body circle."""
        rows = []
        obstacle_positions = self.config.get_param('obstacle_positions')
        obstacle_sizes = self.config.get_param('obstacle_sizes')
        obstacle_angles = self.config.get_param('obstacle_angles')
        obstacle_clearance = self.config.get_param('obstacle_clearance')
        obstacle_distance_scale = self.config.get_param('obstacle_distance_scale')
        gain = self.config.get_param('obstacle_softmax_gain')

        for center, radius in self._body_circles(x_i):
            for obs_idx in range(self.config.obstacle_count):
                obs_center = obstacle_positions[:, obs_idx]
                obs_size = obstacle_sizes[:, obs_idx]
                angle = obstacle_angles[obs_idx, 0]
                dx = center[0, 0] - obs_center[0, 0]
                dy = center[1, 0] - obs_center[1, 0]
                local_x = dx * cas.cos(angle) + dy * cas.sin(angle)
                local_y = -dx * cas.sin(angle) + dy * cas.cos(angle)
                half_l = 0.5 * obs_size[0, 0] + radius + obstacle_clearance
                half_w = 0.5 * obs_size[1, 0] + radius + obstacle_clearance
                signed_halfspace = cas.vertcat(
                    local_x - half_l,
                    -local_x - half_l,
                    local_y - half_w,
                    -local_y - half_w,
                )
                smooth_outside = (
                    cas.logsumexp(gain * signed_halfspace) - np.log(4.0)
                ) / gain
                rows.append(-smooth_outside / obstacle_distance_scale)
        if rows:
            h_val = cas.vertcat(*rows)
        else:
            h_val = cas.SX.zeros(0, 1)
        assert h_val.shape == (self._obstacle_rows_per_agent, 1)
        return h_val

    def state_h(self, x_i):
        """Signed-speed and articulation bounds, normalized as h <= 0."""
        v_min = self.config.get_param('v_min')
        v_max = self.config.get_param('v_max')
        articulation_max = self.config.get_param('articulation_max')
        speed = x_i[2, 0]
        phi = x_i[4, 0]
        speed_scale = v_max - v_min
        return cas.vertcat(
            (v_min - speed) / speed_scale,
            (speed - v_max) / speed_scale,
            (phi - articulation_max) / articulation_max,
            (-phi - articulation_max) / articulation_max,
        )

    def control_h(self, u_i):
        """Acceleration and steering bounds, normalized as h <= 0."""
        a_max = self.config.get_param('a_max')
        steering_max = self.config.get_param('steering_max')
        acceleration = u_i[0, 0]
        steering = u_i[1, 0]
        return cas.vertcat(
            (acceleration - a_max) / a_max,
            (-acceleration - a_max) / a_max,
            (steering - steering_max) / steering_max,
            (-steering - steering_max) / steering_max,
        )

    def _body_circle_label(self, circle_idx):
        if circle_idx < self.config.tractor_circle_count:
            return f'tractor circle {circle_idx}'
        trailer_idx = circle_idx - self.config.tractor_circle_count
        return f'trailer circle {trailer_idx}'

    def _numeric_tractor_circle(self, x_i, circle_idx):
        offset = (
            self.config.tractor_length
            * (circle_idx + 0.5)
            / self.config.tractor_circle_count
        )
        heading = x_i[3]
        center = x_i[:2] + offset * np.array([np.cos(heading), np.sin(heading)])
        radius = 0.5 * self.config.tractor_width + self.config.collision_buffer
        return center, radius

    def _numeric_trailer_circle(self, x_i, circle_idx):
        offset = (
            -self.config.trailer_length
            * (circle_idx + 0.5)
            / self.config.trailer_circle_count
        )
        heading = x_i[3] - x_i[4]
        center = x_i[:2] + offset * np.array([np.cos(heading), np.sin(heading)])
        radius = 0.5 * self.config.trailer_width + self.config.collision_buffer
        return center, radius

    def _numeric_body_circles(self, x_i):
        circles = []
        for idx in range(self.config.tractor_circle_count):
            circles.append((*self._numeric_tractor_circle(x_i, idx),
                            self._body_circle_label(idx)))
        for idx in range(self.config.trailer_circle_count):
            body_idx = self.config.tractor_circle_count + idx
            circles.append((*self._numeric_trailer_circle(x_i, idx),
                            self._body_circle_label(body_idx)))
        return circles

    @staticmethod
    def _circle_h_debug(circle_a, circle_b):
        center_a, radius_a, _ = circle_a
        center_b, radius_b, _ = circle_b
        dist = float(np.linalg.norm(center_a - center_b))
        required = float(radius_a + radius_b)
        h_val = required**2 - dist**2
        return h_val, dist, required

    def _prepare_inspect_u(self, u):
        gc = self.config
        u_arr = np.asarray(u)
        if u_arr.shape == (gc.m, gc.N, gc.T):
            return u_arr.reshape((gc.m * gc.N, gc.T), order='F')
        if u_arr.shape == (gc.m * gc.N, gc.T):
            return np.asarray(u_arr, dtype=float, order='F')
        raise ValueError(
            f'u must have shape {(gc.m, gc.N, gc.T)} or '
            f'{(gc.m * gc.N, gc.T)}, got {u_arr.shape}')

    def _prepare_inspect_x(self, x):
        gc = self.config
        x_arr = np.asarray(x)
        if x_arr.shape == (gc.n, gc.N, gc.T + 1):
            x_arr = x_arr[:, :, 1:]
        if x_arr.shape == (gc.n, gc.N, gc.T):
            return x_arr.reshape((gc.n * gc.N, gc.T), order='F')
        if x_arr.shape == (gc.n * gc.N, gc.T):
            return np.asarray(x_arr, dtype=float, order='F')
        raise ValueError(
            f'x must have shape {(gc.n, gc.N, gc.T + 1)}, '
            f'{(gc.n, gc.N, gc.T)}, or {(gc.n * gc.N, gc.T)}, got {x_arr.shape}')

    def inspect_h(self, u, x=None, solver=None, tol=1e-6, log_level=logging.INFO):
        """Log where positive inequality residuals h(x,u) > 0 come from.

        Args:
            u: Control trajectory, shape (m,N,T) or (m*N,T).
            x: Optional state trajectory as (n,N,T+1), (n,N,T), or (n*N,T).
            solver: Solver with h_casadi/rollout_casadi functions. Required if x is None.
            tol: Only rows with positive violation above this tolerance get detailed logs.
            log_level: Python logging level used for detail and summary lines.
        """
        gc = self.config
        if solver is None:
            raise ValueError('inspect_h requires a solver with h_casadi functions')

        int_param_dm = cas.DM(gc.get_int_param_np())
        double_param_dm = cas.DM(gc.get_double_param_np())
        params_dm = [int_param_dm, double_param_dm]
        u_flat = cas.DM(self._prepare_inspect_u(u))
        if x is None:
            x_flat = solver.rollout_casadi(gc.x0, u_flat, *params_dm)
        else:
            x_flat = cas.DM(self._prepare_inspect_x(x))
        context_dm = solver.get_full_context_casadi(x_flat)
        h_val = np.asarray(solver.h_casadi(
            x_flat, u_flat, context_dm, *params_dm))
        h_cols = 1 if gc.variational_gne else gc.N
        h_val = h_val.reshape(gc.n_h, h_cols, order='F')
        x_np = np.asarray(x_flat).reshape((gc.n * gc.N, gc.T), order='F')
        u_np = np.asarray(u_flat).reshape((gc.m * gc.N, gc.T), order='F')

        family_res = {
            'collision_h': 0.0,
            'self_collision_h': 0.0,
            'obstacle_h': 0.0,
            'state_h': 0.0,
            'control_h': 0.0,
        }
        family_max = {name: 0.0 for name in family_res}
        detail_count = 0
        max_details = 80

        def positive_res(rows, cols):
            vals = h_val[rows, :][:, cols]
            pos = np.clip(vals, a_min=0.0, a_max=None)
            return float(np.linalg.norm(pos)), float(np.max(pos)) if pos.size else 0.0

        def log_detail(message, *args):
            nonlocal detail_count
            if detail_count < max_details:
                logger.log(log_level, message, *args)
            detail_count += 1

        row = 0
        for k in range(gc.T):
            xk_np = x_np[:, k].reshape((gc.n, gc.N), order='F')
            uk_np = u_np[:, k].reshape((gc.m, gc.N), order='F')
            for i in range(gc.N):
                for j in range(i + 1, gc.N):
                    rows = slice(row, row + self._rows_per_vehicle_pair)
                    cols = [0] if gc.variational_gne else [i, j]
                    res, max_res = positive_res(rows, cols)
                    family_res['collision_h'] += res**2
                    family_max['collision_h'] = max(
                        family_max['collision_h'], max_res)
                    if max_res > tol:
                        local_vals = h_val[rows, :][:, cols]
                        local_pos = np.clip(local_vals, a_min=0.0, a_max=None)
                        local_idx = int(np.argmax(local_pos))
                        row_idx = local_idx // len(cols)
                        circle_i = row_idx // self._body_circle_count
                        circle_j = row_idx % self._body_circle_count
                        circle_i_val = self._numeric_body_circles(xk_np[:, i])[circle_i]
                        circle_j_val = self._numeric_body_circles(xk_np[:, j])[circle_j]
                        _, dist, required = self._circle_h_debug(
                            circle_i_val, circle_j_val)
                        log_detail(
                            'h infeasible: collision_h k=%s agents=(%s,%s) '
                            '%s vs %s h=%.6g = required_dist^2 %.6g - dist^2 %.6g; '
                            'dist=%.6g required_dist=%.6g centers=%s,%s radii=(%.6g,%.6g)',
                            k + 1, i, j, circle_i_val[2], circle_j_val[2],
                            max_res, required**2, dist**2, dist, required,
                            np.round(circle_i_val[0], 4).tolist(),
                            np.round(circle_j_val[0], 4).tolist(),
                            circle_i_val[1], circle_j_val[1])
                    row += self._rows_per_vehicle_pair

            for i in range(gc.N):
                col = 0 if gc.variational_gne else i

                if self._self_collision_rows > 0:
                    rows = slice(row, row + self._self_collision_rows)
                    res, max_res = positive_res(rows, [col])
                    family_res['self_collision_h'] += res**2
                    family_max['self_collision_h'] = max(
                        family_max['self_collision_h'], max_res)
                    if max_res > tol:
                        local_pos = np.clip(h_val[rows, col], a_min=0.0, a_max=None)
                        local_idx = int(np.argmax(local_pos))
                        tractor_idx = local_idx // gc.trailer_circle_count
                        trailer_idx = local_idx % gc.trailer_circle_count
                        tractor_circle = (
                            *self._numeric_tractor_circle(xk_np[:, i], tractor_idx),
                            f'tractor circle {tractor_idx}',
                        )
                        trailer_circle = (
                            *self._numeric_trailer_circle(xk_np[:, i], trailer_idx),
                            f'trailer circle {trailer_idx}',
                        )
                        _, dist, required = self._circle_h_debug(
                            tractor_circle, trailer_circle)
                        log_detail(
                            'h infeasible: self_collision_h k=%s agent=%s '
                            'tractor circle %s vs trailer circle %s '
                            'h=%.6g = required_dist^2 %.6g - dist^2 %.6g; '
                            'dist=%.6g required_dist=%.6g centers=%s,%s radii=(%.6g,%.6g)',
                            k + 1, i, tractor_idx, trailer_idx, max_res,
                            required**2, dist**2, dist, required,
                            np.round(tractor_circle[0], 4).tolist(),
                            np.round(trailer_circle[0], 4).tolist(),
                            tractor_circle[1], trailer_circle[1])
                    row += self._self_collision_rows

                rows = slice(row, row + self._obstacle_rows_per_agent)
                res, max_res = positive_res(rows, [col])
                family_res['obstacle_h'] += res**2
                family_max['obstacle_h'] = max(family_max['obstacle_h'], max_res)
                if max_res > tol:
                    local_pos = np.clip(h_val[rows, col], a_min=0.0, a_max=None)
                    local_idx = int(np.argmax(local_pos))
                    body_idx = local_idx // max(gc.obstacle_count, 1)
                    obstacle_idx = local_idx % max(gc.obstacle_count, 1)
                    center, radius, body_label = self._numeric_body_circles(
                        xk_np[:, i])[body_idx]
                    obs_center = gc.obstacle_positions[:, obstacle_idx]
                    obs_size = gc.obstacle_sizes[:, obstacle_idx]
                    angle = gc.obstacle_angles[obstacle_idx, 0]
                    delta = center - obs_center
                    local_x = delta[0] * np.cos(angle) + delta[1] * np.sin(angle)
                    local_y = -delta[0] * np.sin(angle) + delta[1] * np.cos(angle)
                    half_l = 0.5 * obs_size[0] + radius + gc.obstacle_clearance
                    half_w = 0.5 * obs_size[1] + radius + gc.obstacle_clearance
                    log_detail(
                        'h infeasible: obstacle_h k=%s agent=%s obstacle=%s %s '
                        'h=%.6g; local=(%.6g,%.6g) inflated_half_extents=(%.6g,%.6g) '
                        'circle_center=%s circle_radius=%.6g',
                        k + 1, i, obstacle_idx, body_label, max_res,
                        local_x, local_y, half_l, half_w,
                        np.round(center, 4).tolist(), radius)
                row += self._obstacle_rows_per_agent

                rows = slice(row, row + 4)
                res, max_res = positive_res(rows, [col])
                family_res['state_h'] += res**2
                family_max['state_h'] = max(family_max['state_h'], max_res)
                if max_res > tol:
                    state_labels = ['v_min', 'v_max', 'phi_max', 'phi_min']
                    local_idx = int(np.argmax(np.clip(
                        h_val[rows, col], a_min=0.0, a_max=None)))
                    state_values = {
                        'v_min': xk_np[2, i],
                        'v_max': xk_np[2, i],
                        'phi_max': xk_np[4, i],
                        'phi_min': xk_np[4, i],
                    }
                    state_limits = {
                        'v_min': gc.v_min,
                        'v_max': gc.v_max,
                        'phi_max': gc.articulation_max,
                        'phi_min': -gc.articulation_max,
                    }
                    log_detail(
                        'h infeasible: state_h k=%s agent=%s bound=%s '
                        'h=%.6g value=%.6g limit=%.6g',
                        k + 1, i, state_labels[local_idx], max_res,
                        state_values[state_labels[local_idx]],
                        state_limits[state_labels[local_idx]])
                row += 4

                rows = slice(row, row + 4)
                res, max_res = positive_res(rows, [col])
                family_res['control_h'] += res**2
                family_max['control_h'] = max(family_max['control_h'], max_res)
                if max_res > tol:
                    control_labels = ['a_max', 'a_min', 'steering_max', 'steering_min']
                    local_idx = int(np.argmax(np.clip(
                        h_val[rows, col], a_min=0.0, a_max=None)))
                    control_values = {
                        'a_max': uk_np[0, i],
                        'a_min': uk_np[0, i],
                        'steering_max': uk_np[1, i],
                        'steering_min': uk_np[1, i],
                    }
                    control_limits = {
                        'a_max': gc.a_max,
                        'a_min': -gc.a_max,
                        'steering_max': gc.steering_max,
                        'steering_min': -gc.steering_max,
                    }
                    log_detail(
                        'h infeasible: control_h k=%s agent=%s bound=%s '
                        'h=%.6g value=%.6g limit=%.6g',
                        k + 1, i, control_labels[local_idx], max_res,
                        control_values[control_labels[local_idx]],
                        control_limits[control_labels[local_idx]])
                row += 4

        h_pos_res = float(np.sum(np.clip(h_val, a_min=0.0, a_max=None)**2))
        if detail_count > max_details:
            logger.log(log_level, 'h infeasible: suppressed %s additional detail rows',
                       detail_count - max_details)
        if h_pos_res <= tol**2:
            logger.log(log_level, 'h infeasibility: no positive rows above tol=%s', tol)
        logger.log(log_level, 'h_pos_res=%s, family_res=%s, family_max=%s',
                   h_pos_res, family_res, family_max)
        return {
            'h_pos_res': h_pos_res,
            'family_res': family_res,
            'family_max': family_max,
        }


def create_random_game(tractor_count=2, horizon=30, variational_gne=False,
                       car_count=None, obstacles=None):
    """Create the default side-by-side tractor-trailer game.

    The factory keeps the local naming convention used by other games; the
    default instance is deterministic rather than random.
    """
    if car_count is not None:
        tractor_count = car_count
    obstacle_config = (
        obstacle_kwargs(_default_obstacles())
        if obstacles is None else obstacle_kwargs(obstacles)
    )
    config = TractorTrailerManeuveringCasadiConfig(
        T=horizon,
        N=tractor_count,
        variational_gne=variational_gne,
        **obstacle_config,
    )
    return TractorTrailerManeuveringCasadi(config)
