"""Air Traffic Control game, CasADi version."""
import os
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import casadi as cas

from rd3g.utilities.util import BASEDIR, resolve_logname
from rd3g.core.casadi_game import CasadiGame, CasadiGameConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def create_multi_valley_fun(targets: list[float], width_fraction: float = 0.18,
                            beta: float = 50.0, amplitude: float = 1.0):
    """Create an unbounded, scale-normalized CasADi cost with valleys at targets.

    Args:
        targets: Target values where the returned function should have valleys.
        width_fraction: Dimensionless valley width, relative to the target span.
        beta: Exponential soft-min sharpness. Larger values select one valley harder.
        amplitude: Multiplies the returned cost.
    """
    targets_np = np.asarray(targets, dtype=float).reshape(-1)
    if targets_np.size == 0:
        raise ValueError('targets must contain at least one value')
    if not np.all(np.isfinite(targets_np)):
        raise ValueError('targets must be finite')
    if width_fraction <= 0.0:
        raise ValueError('width_fraction must be positive')
    if beta <= 0.0:
        raise ValueError('beta must be positive')
    if amplitude <= 0.0:
        raise ValueError('amplitude must be positive')

    target_span = float(np.max(targets_np) - np.min(targets_np))
    target_magnitude = float(np.max(np.abs(targets_np)))
    scale = max(target_span, target_magnitude, 1.0)

    val = cas.SX.sym('val')
    valley_terms = []
    for target in targets_np:
        normalized_error = (val - target) / (scale * width_fraction)
        valley_terms.append(normalized_error**2)

    valley_vec = cas.vertcat(*valley_terms)
    softmin = -(1.0 / beta) * cas.logsumexp(-beta * valley_vec)
    return cas.Function('multi_valley_fun', [val], [amplitude * softmin],
                        ['val'], ['cost'])


def _multi_valley_cost_np(values, targets, width_fraction: float = 0.12,
                          beta: float = 30.0, amplitude: float = 1.0):
    """Numpy equivalent of create_multi_valley_fun() for visualization grids."""
    targets_np = np.asarray(targets, dtype=float).reshape(-1)
    target_span = float(np.max(targets_np) - np.min(targets_np))
    target_magnitude = float(np.max(np.abs(targets_np)))
    scale = max(target_span, target_magnitude, 1.0)
    values = np.asarray(values)

    valley_terms = []
    for target in targets_np:
        normalized_error = (values - target) / (scale * width_fraction)
        valley_terms.append(normalized_error**2)

    valley_arr = np.stack(valley_terms, axis=0)
    terms = -beta * valley_arr
    max_term = np.max(terms, axis=0)
    logsumexp = max_term + np.log(np.sum(np.exp(terms - max_term), axis=0))
    return amplitude * (-(1.0 / beta) * logsumexp)


@dataclass(frozen=False)
class AirTrafficControlCasadiConfig(CasadiGameConfig):
    """Base class for the air traffic control game configuration.

    State ordering is x_i = [x, y, V, psi].
    Control ordering is u_i = [a, omega].
    """
    T: int = 20
    dt: float = 1.0
    N: int = 3
    n: int = 4
    m: int = 2
    n_h: int = 0
    """Total number of canonical inequality constraints."""
    n_c: int = 0
    """Dimension of context variable per agent per stage; unused here."""
    variational_gne: bool = False
    """If True, use one shared multiplier per canonical constraint."""

    # Aircraft limits from the problem statement.
    V_min: float = 70.0
    V_max: float = 250.0
    a_max: float = 0.3
    omega_max: float = 0.1  # 0.05

    # Airport model. Columns in runway_positions are runway thresholds [x_R, y_R].
    runway_count: int = 2
    runway_positions: Any = field(
        default_factory=lambda: np.array([[0.0, 0.0], [0.0, 300.0]], dtype=float)
    )
    # The TeX writes "-20 deg = 0.35 rad"; use the signed angle from -20 deg.
    runway_headings: Any = field(
        default_factory=lambda: np.array([[0.0], [-np.deg2rad(20.0)]], dtype=float)
    )

    # Approach cone smoothed indicator.
    alpha: float = 4.0
    rho_long: float = 150.0
    rho_lat: float = 25.0

    # Inter-agent and localized landing constraints.
    D_min: float = 200.0
    delta_psi: float = np.deg2rad(5.0)
    delta_y: float = 15.0
    M_lat: float = 2.5e7
    M_psi: float = 2.0

    # Soft-min cost parameters.
    beta: float = 5.0
    distance_y_weight: float = 5.0
    distance_cost_scale: float = 300.0
    """Distance scale in meters used to make runway distance costs dimensionless."""
    landing_gate_M: float = 1.0
    """Dimensionless Big-M used after landing constraints are normalized."""

    # Visualization parameters.
    runway_visual_length: float = 450.0
    aircraft_visual_length: float = 80.0

    x0: Any = field(
        default_factory=lambda: np.array(
            [
                [-1600.0, -1900.0, -1400.0],
                [-180.0, 1800.0, 2300.0],
                [100.0, 95.0, 105.0],
                [0.05, -0.30, -0.45],
            ],
            dtype=float,
        )
    )
    """Initial state for all agents, dim: (n,N)."""
    J_R: Any = field(default_factory=lambda: np.eye(2, dtype=float) * 0.5)
    """Cost matrix for normalized controls [a/a_max, omega/omega_max]."""

    def __post_init__(self):
        assert self.x0.shape == (self.n, self.N), (
            'Incorrect self.x0 dimension, '
            f'should be {(self.n, self.N)}, but got {self.x0.shape}'
        )
        assert self.runway_positions.shape == (2, self.runway_count)
        assert self.runway_headings.shape == (self.runway_count, 1)
        assert self.J_R.shape == (self.m, self.m)
        if self.n_h == 0:
            pair_count = self.N * (self.N - 1) // 2
            rows_per_agent = 2 * self.runway_count + 2 + 4
            self.n_h = (pair_count + rows_per_agent * self.N) * self.T
        return super().__post_init__()


class AirTrafficControlCasadi(CasadiGame):
    """Air traffic control game with simple point-mass aircraft dynamics.

    u = [a, omega]
    x = [x, y, V, psi]
    x, y: 2D aircraft position in meters
    V: ground speed in m/s
    psi: heading angle in radians

    Agent count: N, time step: 1..T+1
    X (game state) = concatenated state, first by agent, then by time.
        x has shape (n*N,T), storing x_1..x_T.
    U (control) = concatenated control with shape (m*N,T), storing u_0..u_{T-1}.
    To obtain x_i_k, use cas.reshape(x[:, k], n, N)[:, i].
    """

    def __init__(self, config: AirTrafficControlCasadiConfig):
        super().__init__(config)
        self._runway_heading_cost_fun = create_multi_valley_fun(
            config.runway_headings[:, 0].tolist()
        )
        self._runway_lateral_cost_fun = create_multi_valley_fun(
            [0.0],
        )
        self.color_vec = [
            'tab:purple', 'tab:orange', 'tab:red', 'tab:green',
            'tab:blue', 'tab:pink', 'tab:cyan', 'black'
        ]
        self._set_visual_bounds()

    def _set_visual_bounds(self):
        """Pick plot bounds that include runways and the configured initial states."""
        pts = [self.config.x0[:2, :].T]
        for runway_idx in range(self.config.runway_count):
            pts.append(self._runway_polygon_np(runway_idx))
        pts = np.vstack(pts)
        margin = max(500.0, 2.0 * self.config.D_min)
        self.visual_x_lim = [float(np.min(pts[:, 0]) - margin),
                             float(np.max(pts[:, 0]) + margin)]
        self.visual_y_lim = [float(np.min(pts[:, 1]) - margin),
                             float(np.max(pts[:, 1]) + margin)]

    def _runway_polygon_np(self, runway_idx):
        """Return four runway rectangle corners for matplotlib drawing."""
        p = self.config.runway_positions[:, runway_idx]
        psi = self.config.runway_headings[runway_idx, 0]
        long_axis = np.array([np.cos(psi), np.sin(psi)])
        lat_axis = np.array([-np.sin(psi), np.cos(psi)])
        half_width = self.config.delta_y
        length = self.config.runway_visual_length
        return np.vstack([
            p - half_width * lat_axis,
            p + half_width * lat_axis,
            p + length * long_axis + half_width * lat_axis,
            p + length * long_axis - half_width * lat_axis,
        ])

    def _draw_runways(self, ax):
        """Draw runway pavement, threshold bars, and centerlines."""
        for runway_idx in range(self.config.runway_count):
            polygon = self._runway_polygon_np(runway_idx)
            ax.fill(polygon[:, 0], polygon[:, 1], color='dimgray',
                    edgecolor='white', linewidth=1.5, zorder=1)

            p = self.config.runway_positions[:, runway_idx]
            psi = self.config.runway_headings[runway_idx, 0]
            long_axis = np.array([np.cos(psi), np.sin(psi)])
            lat_axis = np.array([-np.sin(psi), np.cos(psi)])
            end = p + self.config.runway_visual_length * long_axis
            ax.plot([p[0], end[0]], [p[1], end[1]], '--', color='white',
                    linewidth=1.0, zorder=2)
            threshold = np.vstack([p - self.config.delta_y * lat_axis,
                                   p + self.config.delta_y * lat_axis])
            ax.plot(threshold[:, 0], threshold[:, 1], color='white',
                    linewidth=2.0, zorder=2)

    def _draw_approach_weight_heatmap(self, ax, runway_idx):
        """Draw the approach-cone weight W for one runway as a background heatmap."""
        if not 0 <= runway_idx < self.config.runway_count:
            raise ValueError(
                f'runway_idx must be in [0, {self.config.runway_count}), got {runway_idx}'
            )

        x_vec = np.linspace(*self.visual_x_lim, 220)
        y_vec = np.linspace(*self.visual_y_lim, 220)
        grid_x, grid_y = np.meshgrid(x_vec, y_vec)
        weight = self._approach_weight_np(grid_x, grid_y, runway_idx)

        im = ax.imshow(weight,
                       extent=[*self.visual_x_lim, *self.visual_y_lim],
                       origin='lower',
                       cmap='coolwarm',
                       alpha=0.65,
                       vmin=0.0,
                       vmax=1.0,
                       zorder=0)
        self._attach_heatmap_view_autoscale(ax, im, x_vec, y_vec, weight)
        return im

    def _attach_heatmap_view_autoscale(self, ax, im, x_vec, y_vec, values):
        """Rescale heatmap colors to visible data whenever the axes are re-zoomed."""
        values = np.asarray(values)

        def update_clim(_ax=None):
            x_low, x_high = sorted(ax.get_xlim())
            y_low, y_high = sorted(ax.get_ylim())
            x_mask = (x_vec >= x_low) & (x_vec <= x_high)
            y_mask = (y_vec >= y_low) & (y_vec <= y_high)
            visible_values = values[np.ix_(y_mask, x_mask)]
            if visible_values.size == 0:
                visible_values = values

            finite_values = visible_values[np.isfinite(visible_values)]
            if finite_values.size == 0:
                return
            vmin = float(np.min(finite_values))
            vmax = float(np.max(finite_values))
            if np.isclose(vmin, vmax):
                pad = max(1e-6, abs(vmin) * 1e-3)
                vmin -= pad
                vmax += pad
            im.set_clim(vmin, vmax)
            im.figure.canvas.draw_idle()

        update_clim()
        ax.callbacks.connect('xlim_changed', update_clim)
        ax.callbacks.connect('ylim_changed', update_clim)

    def _approach_weight_np(self, grid_x, grid_y, runway_idx):
        """Numpy version of approach_weight() for visualization grids."""
        p = self.config.runway_positions[:, runway_idx]
        psi = self.config.runway_headings[runway_idx, 0]
        dx = grid_x - p[0]
        dy = grid_y - p[1]
        d_long = dx * np.cos(psi) + dy * np.sin(psi)
        d_lat = -dx * np.sin(psi) + dy * np.cos(psi)

        before_threshold = 0.5 * (1.0 - np.tanh(0.5 * self.config.alpha * d_long))
        cone_decay = np.exp(-(d_long**2) / (self.config.rho_long**2) -
                            (d_lat**2) / (self.config.rho_lat**2))
        return before_threshold * cone_decay

    def _draw_step_cost_heatmap(self, ax, speed_heading):
        """Draw the zero-control stage cost over x/y for a fixed speed/heading."""
        try:
            speed, heading = speed_heading
        except (TypeError, ValueError) as exc:
            raise ValueError(
                'show_step_cost_heatmap must be a tuple of (v, psi)'
            ) from exc
        del speed

        x_vec = np.linspace(*self.visual_x_lim, 220)
        y_vec = np.linspace(*self.visual_y_lim, 220)
        grid_x, grid_y = np.meshgrid(x_vec, y_vec)

        lateral_costs = []
        for runway_idx in range(self.config.runway_count):
            p = self.config.runway_positions[:, runway_idx]
            psi_r = self.config.runway_headings[runway_idx, 0]
            dx = grid_x - p[0]
            dy = grid_y - p[1]
            d_lat = -dx * np.sin(psi_r) + dy * np.cos(psi_r)
            lateral_cost = _multi_valley_cost_np(
                d_lat / self.config.rho_lat, [0.0], width_fraction=1.0)
            lateral_costs.append(lateral_cost)

        lateral_cost_arr = np.stack(lateral_costs, axis=0)
        terms = -self.config.beta * lateral_cost_arr
        max_term = np.max(terms, axis=0)
        logsumexp = max_term + np.log(np.sum(np.exp(terms - max_term), axis=0))
        lateral_destination_cost = -(1.0 / self.config.beta) * logsumexp
        heading_destination_cost = _multi_valley_cost_np(
            heading, self.config.runway_headings[:, 0])
        step_cost = lateral_destination_cost + heading_destination_cost

        im = ax.imshow(step_cost,
                       extent=[*self.visual_x_lim, *self.visual_y_lim],
                       origin='lower',
                       cmap='coolwarm',
                       alpha=0.65,
                       zorder=0)
        self._attach_heatmap_view_autoscale(ax, im, x_vec, y_vec, step_cost)
        return im

    def visualize(self, u, x, show=True, save=False,
                  show_approach_weight_heatmap=None,
                  show_step_cost_heatmap=None):
        """Visualize trajectories and aircraft multi-exposure in one frame.

        Args:
            u: (m,N,T)
            x: (n,N,T+1)
            show_approach_weight_heatmap: None for no heatmap, otherwise runway index.
            show_step_cost_heatmap: None for no heatmap, otherwise a tuple (v, psi).

        Aircraft sprites are overlaid along the horizon with later states more opaque.
        """
        n = self.config.n
        m = self.config.m
        T = self.config.T
        N = self.config.N

        # Publication figure tuning knobs.
        aircraft_visual_length = 200.0  # meters
        aircraft_exposure_step_skip = 3
        aircraft_min_alpha = 0.18
        aircraft_alpha_prominence_power = 2.0
        aircraft_tint_alpha = 0.55
        aircraft_colors = [
            '#56B4E9', '#E69F00', '#009E73', '#D55E00',
            '#CC79A7', '#0072B2', '#F0E442', '#999999',
        ]
        runway_visual_scale = 1.5
        aircraft_sprite_path = os.path.join(
            BASEDIR, 'rd3g', 'resources', 'aircraft_topdown.png')
        trajectory_alpha = 0.72
        save_dpi = 600

        if (not show) and (not save):
            return None
        assert u.shape == (m, N, T)
        assert x.shape == (n, N, T + 1)

        fig, ax = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)
        ax.set_facecolor((54 / 255, 69 / 255, 79 / 255))
        if (show_approach_weight_heatmap is not None and
                show_step_cost_heatmap is not None):
            raise ValueError(
                'show_approach_weight_heatmap and show_step_cost_heatmap '
                'are mutually exclusive')
        if show_approach_weight_heatmap is not None:
            im = self._draw_approach_weight_heatmap(
                ax, int(show_approach_weight_heatmap))
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='approach weight')
        if show_step_cost_heatmap is not None:
            im = self._draw_step_cost_heatmap(ax, show_step_cost_heatmap)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='step cost')

        for runway_idx in range(self.config.runway_count):
            p = self.config.runway_positions[:, runway_idx]
            psi = self.config.runway_headings[runway_idx, 0]
            long_axis = np.array([np.cos(psi), np.sin(psi)])
            lat_axis = np.array([-np.sin(psi), np.cos(psi)])
            half_width = runway_visual_scale * self.config.delta_y
            length = runway_visual_scale * self.config.runway_visual_length
            polygon = np.vstack([
                p - half_width * lat_axis,
                p + half_width * lat_axis,
                p + length * long_axis + half_width * lat_axis,
                p + length * long_axis - half_width * lat_axis,
            ])
            ax.fill(polygon[:, 0], polygon[:, 1], color='dimgray',
                    edgecolor='white', linewidth=1.5, zorder=1)
            end = p + length * long_axis
            ax.plot([p[0], end[0]], [p[1], end[1]], '--', color='white',
                    linewidth=1.0, zorder=2)
            threshold = np.vstack(
                [p - half_width * lat_axis, p + half_width * lat_axis])
            ax.plot(threshold[:, 0], threshold[:, 1], color='white',
                    linewidth=2.0, zorder=2)

        if aircraft_exposure_step_skip <= 0:
            raise ValueError('aircraft_exposure_step_skip must be positive')
        if aircraft_alpha_prominence_power <= 0:
            raise ValueError('aircraft_alpha_prominence_power must be positive')
        if not 0 <= aircraft_min_alpha <= 1:
            raise ValueError('aircraft_min_alpha must be in [0, 1]')
        if not 0 <= aircraft_tint_alpha <= 1:
            raise ValueError('aircraft_tint_alpha must be in [0, 1]')
        from matplotlib.transforms import (  # pylint: disable=import-outside-toplevel
            Affine2D,
        )
        from matplotlib.colors import to_rgba  # pylint: disable=import-outside-toplevel
        sprite = plt.imread(aircraft_sprite_path)
        exposure_steps = np.arange(0, T + 1, aircraft_exposure_step_skip)
        weights = (
            np.arange(1, len(exposure_steps) + 1, dtype=float)
            ** aircraft_alpha_prominence_power
        )
        aircraft_alphas = (
            aircraft_min_alpha + (1 - aircraft_min_alpha) * weights / weights[-1]
        )
        for i in range(N):
            color = aircraft_colors[i % len(aircraft_colors)]
            ax.plot(
                x[0, i, :],
                x[1, i, :],
                color=color,
                alpha=trajectory_alpha,
                linewidth=2.2,
                solid_capstyle='round',
                solid_joinstyle='round',
                antialiased=True,
                zorder=3,
            )
            color_rgb = np.array(to_rgba(color)[:3])
            tinted_sprite = np.array(sprite, copy=True)
            tinted_sprite[..., :3] = (
                (1 - aircraft_tint_alpha) * tinted_sprite[..., :3]
                + aircraft_tint_alpha * color_rgb
            )
            for k, alpha in zip(exposure_steps, aircraft_alphas):
                pos_x = x[0, i, k]
                pos_y = x[1, i, k]
                heading = x[3, i, k]
                half_length = aircraft_visual_length / 2.0
                half_width = half_length * sprite.shape[0] / sprite.shape[1]
                transform = (
                    Affine2D().rotate_around(pos_x, pos_y, heading) + ax.transData
                )
                ax.imshow(
                    tinted_sprite,
                    extent=[
                        pos_x - half_length, pos_x + half_length,
                        pos_y - half_width, pos_y + half_width,
                    ],
                    transform=transform,
                    interpolation='none',
                    alpha=alpha,
                    resample=False,
                    zorder=4 + alpha,
                )

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(*self.visual_x_lim)
        ax.set_ylim(*self.visual_y_lim)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.tick_params(direction='in', top=True, right=True)
        ax.grid(color='white', linewidth=0.5, alpha=0.08)
        if save:
            filename = resolve_logname(suffix='png')
            fig.savefig(filename, dpi=save_dpi, bbox_inches='tight')
            logger.info('saved figure to %s', filename)
        if show:
            plt.show()
        return ax

    def animate(self, u, x, show=True, save_gif=False, save_snapshots=False):
        """Animate trajectories and aircraft heading arrows.

        Args:
            u: (m,N,T)
            x: (n,N,T+1)
        """
        n = self.config.n
        m = self.config.m
        T = self.config.T
        N = self.config.N
        if (not show) and (not save_gif) and (not save_snapshots):
            return
        assert u.shape == (m, N, T)
        assert x.shape == (n, N, T + 1)

        fig, ax = plt.subplots()
        ax.set_facecolor((54 / 255, 69 / 255, 79 / 255))
        self._draw_runways(ax)

        colors = [self.color_vec[i % len(self.color_vec)] for i in range(N)]
        trail_vec = []
        for i, color in enumerate(colors):
            (trail,) = ax.plot([], [], '*-', color=color, zorder=3)
            trail_vec.append(trail)

        arrow_len = self.config.aircraft_visual_length
        q = ax.quiver(
            x[0, :, 0],
            x[1, :, 0],
            arrow_len * np.cos(x[3, :, 0]),
            arrow_len * np.sin(x[3, :, 0]),
            angles='xy',
            scale_units='xy',
            scale=1.0,
            color=colors,
            width=0.006,
            zorder=4,
        )

        def update(frame):
            for i, trail in enumerate(trail_vec):
                trail.set_data(x[0, i, :frame + 1], x[1, i, :frame + 1])
            q.set_offsets(np.vstack([x[0, :, frame], x[1, :, frame]]).T)
            q.set_UVC(arrow_len * np.cos(x[3, :, frame]),
                      arrow_len * np.sin(x[3, :, frame]))
            return trail_vec + [q]

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(*self.visual_x_lim)
        ax.set_ylim(*self.visual_y_lim)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')

        anim = FuncAnimation(fig, update, frames=self.config.T + 1, blit=False)

        folder = os.path.join(BASEDIR, 'outputs', 'gifs')
        os.makedirs(folder, exist_ok=True)
        gif_filename = os.path.join(folder, f'atc_{self.config.N}aircraft.gif')
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
            for frame, name in [
                    (0, 'initial'),
                    (self.config.T // 2, 'middle'),
                    (self.config.T, 'final')]:
                update(frame)
                fig.canvas.draw()
                image = Image.frombytes('RGB', fig.canvas.get_width_height(),
                                        fig.canvas.tostring_rgb())
                filename = os.path.join(
                    folder, f'atc_{self.config.N}aircraft_{name}.png')
                image.save(filename)
                logger.info('saved snapshots to %s', filename)
        return

    def runway_coordinates(self, x_i, runway_idx):
        """Return longitudinal and lateral coordinates relative to a runway."""
        runway_positions = self.config.get_param('runway_positions')
        runway_headings = self.config.get_param('runway_headings')
        psi_r = runway_headings[runway_idx, 0]
        dx = x_i[0, 0] - runway_positions[0, runway_idx]
        dy = x_i[1, 0] - runway_positions[1, runway_idx]
        d_long = dx * cas.cos(psi_r) + dy * cas.sin(psi_r)
        d_lat = -dx * cas.sin(psi_r) + dy * cas.cos(psi_r)
        return d_long, d_lat

    def approach_weight(self, x_i, runway_idx):
        """Smooth spatial gate for the final approach cone of one runway."""
        d_long, d_lat = self.runway_coordinates(x_i, runway_idx)
        alpha = self.config.get_param('alpha')
        rho_long = self.config.get_param('rho_long')
        rho_lat = self.config.get_param('rho_lat')

        # Stable equivalent of 1 / (1 + exp(alpha * d_long)).
        before_threshold = 0.5 * (1.0 - cas.tanh(0.5 * alpha * d_long))
        cone_decay = cas.exp(-(d_long**2) / (rho_long**2) -
                             (d_lat**2) / (rho_lat**2))
        return before_threshold * cone_decay

    def runway_distance_sq(self, x_i, runway_idx):
        """Weighted squared distance to a runway threshold, as in the TeX cost."""
        runway_positions = self.config.get_param('runway_positions')
        distance_y_weight = self.config.get_param('distance_y_weight')
        dx = x_i[0, 0] - runway_positions[0, runway_idx]
        dy = x_i[1, 0] - runway_positions[1, runway_idx]
        distance_cost_scale = self.config.get_param('distance_cost_scale')
        return (dx**2 + distance_y_weight * dy**2) / (distance_cost_scale**2)

    def lateral_destination_cost(self, x_i):
        """Soft-min lateral alignment cost across runway centerlines."""
        beta = self.config.get_param('beta')
        rho_lat = self.config.get_param('rho_lat')

        lateral_costs = []
        for runway_idx in range(self.config.runway_count):
            _, d_lat = self.runway_coordinates(x_i, runway_idx)
            lateral_cost = self._runway_lateral_cost_fun(d_lat / rho_lat)
            lateral_costs.append(lateral_cost)

        lateral_cost_vec = cas.vertcat(*lateral_costs)
        return -(1.0 / beta) * cas.logsumexp(-beta * lateral_cost_vec)

    def J(self, x_k, u_k_i, i_onehot):
        """Stage cost for one aircraft.

        x_k.shape (n,N)
        u_k_i.shape (m,1)
        i_onehot: (N,1) agent id in one-hot encoding.
        """
        assert x_k.shape == (self.config.n, self.config.N)
        assert u_k_i.shape == (self.config.m, 1)
        assert i_onehot.shape == (self.config.N, 1)

        x_k_i = x_k @ i_onehot
        heading_cost = self._runway_heading_cost_fun(x_k_i[3, 0])
        lateral_cost = self.lateral_destination_cost(x_k_i)

        destination_cost = heading_cost + 0.1*lateral_cost

        J_R = self.config.get_param('J_R')
        a_max = self.config.get_param('a_max')
        omega_max = self.config.get_param('omega_max')
        u_norm = cas.vertcat(u_k_i[0, 0] / a_max, u_k_i[1, 0] / omega_max)
        control_cost = u_norm.T @ J_R @ u_norm
        return destination_cost + control_cost

    def Jfi(self, x_T, i_onehot):
        """Final cost."""
        return self.J(x_T, cas.SX.zeros(self.config.m), i_onehot)

    def f(self, x_k_i, u_k_i, i_onehot, context_i_k):
        """Aircraft dynamics x_{t+1} = f(x_t, u_t, i).

        Args:
            x_k_i: (n,1), [x, y, V, psi]
            u_k_i: (m,1), [a, omega]
            i_onehot: (N,1), unused because aircraft are homogeneous
            context_i_k: (n_c,1), unused
        Return:
            (n,1) next state after one Euler step.
        """
        del context_i_k
        assert x_k_i.shape == (self.config.n, 1)
        assert u_k_i.shape == (self.config.m, 1)
        assert i_onehot.shape == (self.config.N, 1)

        speed = x_k_i[2, 0]
        heading = x_k_i[3, 0]
        acceleration = u_k_i[0, 0]
        turn_rate = u_k_i[1, 0]
        dt = self.config.get_param('dt')

        dx = cas.vertcat(speed * cas.cos(heading),
                         speed * cas.sin(heading),
                         acceleration,
                         turn_rate)
        return x_k_i + dx * dt

    def h(self, x, u, context):
        """Construct all inequality constraints h(x,u) <= 0.

        Args:
            x: (n*N,T), states x_1..x_T
            u: (m*N,T), controls u_0..u_{T-1}
            context: (n_c*N,T), unused
        Returns:
            h_vec:
                variational mode: (n_h,1)
                non-variational mode: (n_h,N), with -1 for nonparticipants.
        """
        del context
        gc = self.config
        h_rows = []
        if gc.variational_gne:
            for k in range(1, gc.T + 1):
                xk = cas.reshape(x[:, k - 1], gc.n, gc.N)
                uk = cas.reshape(u[:, k - 1], gc.m, gc.N)
                for i in range(gc.N):
                    for j in range(i + 1, gc.N):
                        h_rows.append(self.collision_h(xk[:, i], xk[:, j]))
                for i in range(gc.N):
                    for runway_idx in range(gc.runway_count):
                        h_rows.append(self.landing_h(xk[:, i], runway_idx))
                    h_rows.append(self.speed_h(xk[:, i]))
                    h_rows.append(self.control_h(uk[:, i]))

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
                            hi_rows.append(cas.DM(-1))
                for i in range(gc.N):
                    for runway_idx in range(gc.runway_count):
                        if agent_idx == i:
                            hi_rows.append(self.landing_h(xk[:, i], runway_idx))
                        else:
                            hi_rows.append(-cas.DM.ones(2, 1))
                    if agent_idx == i:
                        hi_rows.append(self.speed_h(xk[:, i]))
                        hi_rows.append(self.control_h(uk[:, i]))
                    else:
                        hi_rows.append(-cas.DM.ones(2, 1))
                        hi_rows.append(-cas.DM.ones(4, 1))
            h_rows.append(cas.vertcat(*hi_rows))

        h_vec = cas.horzcat(*h_rows)
        assert h_vec.shape == (gc.n_h, gc.N), "n_h must be consistent with h().shape[0]"
        return h_vec

    def collision_h(self, x_i, x_j):
        """Pairwise airspace separation constraint.

        Return h <= 0, i.e. ||p_i - p_j|| >= D_min.
        """
        assert x_i.shape == (self.config.n, 1)
        assert x_j.shape == (self.config.n, 1)
        D_min = self.config.get_param('D_min')
        dx = x_i[0, 0] - x_j[0, 0]
        dy = x_i[1, 0] - x_j[1, 0]
        dist = cas.sqrt(dx**2 + dy**2 + 1e-6)
        return D_min / dist - 1.0

    def landing_h(self, x_i, runway_idx):
        """Localized heading and lateral runway-approach constraints."""
        assert x_i.shape == (self.config.n, 1)
        runway_headings = self.config.get_param('runway_headings')
        delta_psi = self.config.get_param('delta_psi')
        delta_y = self.config.get_param('delta_y')
        landing_gate_M = self.config.get_param('landing_gate_M')

        d_long, d_lat = self.runway_coordinates(x_i, runway_idx)
        del d_long
        W = self.approach_weight(x_i, runway_idx)
        psi_error = x_i[3, 0] - runway_headings[runway_idx, 0]
        # Normalize active violations by the allowed error. The spatial gate keeps
        # irrelevant runway constraints order-one instead of meter-squared large.
        heading_scale = 1.0 - cas.cos(delta_psi)
        heading_active_h = (cas.cos(delta_psi) - cas.cos(psi_error)) / heading_scale
        lateral_active_h = (d_lat**2 - delta_y**2) / (delta_y**2)
        heading_h = W * heading_active_h - landing_gate_M * (1.0 - W)
        lateral_h = W * lateral_active_h - landing_gate_M * (1.0 - W)
        return cas.vertcat(heading_h, lateral_h)

    def speed_h(self, x_i):
        """Speed bounds V_min <= V <= V_max as h <= 0 rows."""
        assert x_i.shape == (self.config.n, 1)
        V_min = self.config.get_param('V_min')
        V_max = self.config.get_param('V_max')
        speed = x_i[2, 0]
        speed_scale = V_max - V_min
        return cas.vertcat((V_min - speed) / speed_scale,
                           (speed - V_max) / speed_scale)

    def control_h(self, u_i):
        """Acceleration and turn-rate bounds as h <= 0 rows."""
        assert u_i.shape == (self.config.m, 1)
        a_max = self.config.get_param('a_max')
        omega_max = self.config.get_param('omega_max')
        acceleration = u_i[0, 0]
        turn_rate = u_i[1, 0]
        return cas.vertcat((acceleration - a_max) / a_max,
                           (-acceleration - a_max) / a_max,
                           (turn_rate - omega_max) / omega_max,
                           (-turn_rate - omega_max) / omega_max)

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
        tol = 1e-2

        pair_count = gc.N * (gc.N - 1) // 2
        row = 0
        family_res = {
            'separation': 0.0,
            'landing': 0.0,
            'speed': 0.0,
            'control': 0.0,
        }
        for k in range(gc.T):
            for i in range(gc.N):
                for j in range(i + 1, gc.N):
                    cols = [0] if gc.variational_gne else [i, j]
                    res = max(np.clip(h_val[row, col], a_min=0, a_max=None)
                              for col in cols)
                    family_res['separation'] += res**2
                    if res > tol:
                        logger.info('aircraft %s, %s, k=%s separation %s',
                                    i, j, k + 1, res)
                    row += 1
            assert row == (k + 1) * (pair_count + gc.N * (2 * gc.runway_count + 6)) - (
                gc.N * (2 * gc.runway_count + 6))
            for i in range(gc.N):
                col = 0 if gc.variational_gne else i
                landing_rows = 2 * gc.runway_count
                landing_res = np.linalg.norm(
                    np.clip(h_val[row:row + landing_rows, col], a_min=0, a_max=None))
                family_res['landing'] += landing_res**2
                row += landing_rows
                speed_res = np.linalg.norm(
                    np.clip(h_val[row:row + 2, col], a_min=0, a_max=None))
                family_res['speed'] += speed_res**2
                row += 2
                control_res = np.linalg.norm(
                    np.clip(h_val[row:row + 4, col], a_min=0, a_max=None))
                family_res['control'] += control_res**2
                row += 4

        h_pos_res = np.sum(np.clip(h_val, a_min=0, a_max=None)**2)
        logger.info('h_pos_res=%s, family_res=%s', h_pos_res, family_res)


def create_random_game(aircraft_count=3, horizon=20, variational_gne=False,
                       car_count=None):
    """Create an AirTrafficControlCasadi instance with random initial states."""
    if car_count is not None:
        aircraft_count = car_count
    default = AirTrafficControlCasadiConfig()
    T = horizon
    N = aircraft_count
    n = default.n
    m = default.m

    def sample_aircraft(runway_idx, queue_idx=0):
        """Sample one aircraft on short final, then rotate to world coordinates."""
        p = default.runway_positions[:, runway_idx]
        psi_r = default.runway_headings[runway_idx, 0]
        # Keep aircraft close to the runway threshold while staggering any queue
        # assigned to the same runway enough to satisfy the 200 m separation rule.
        d_long = -(np.random.uniform(800.0, 1800.0) +
                   queue_idx * (default.D_min + 100.0))
        d_lat = np.random.uniform(-150.0, 150.0)
        long_axis = np.array([np.cos(psi_r), np.sin(psi_r)])
        lat_axis = np.array([-np.sin(psi_r), np.cos(psi_r)])
        pos = p + d_long * long_axis + d_lat * lat_axis
        speed = np.random.uniform(75.0, 95.0)
        heading = psi_r + np.pi/180.0*np.random.uniform(-20, 20)
        return np.array([pos[0], pos[1], speed, heading])

    x0_vec = []
    runway_idx_vec = np.arange(N) % default.runway_count
    np.random.shuffle(runway_idx_vec)
    runway_queue_count = np.zeros(default.runway_count, dtype=int)
    for runway_idx in runway_idx_vec:
        queue_idx = runway_queue_count[runway_idx]
        x0_vec.append(sample_aircraft(runway_idx, queue_idx))
        runway_queue_count[runway_idx] += 1

    D_min_sq = default.D_min**2
    while True:
        positions = np.asarray(x0_vec)[:, :2]
        colliding_idx = None
        for i in range(N - 1):
            delta = positions[i + 1:] - positions[i]
            dist_sq = np.sum(delta * delta, axis=1)
            hits = np.flatnonzero(dist_sq < D_min_sq)
            if hits.size > 0:
                colliding_idx = i + 1 + hits[0]
                break
        if colliding_idx is None:
            break
        runway_idx = runway_idx_vec[colliding_idx]
        same_runway_before = np.sum(runway_idx_vec[:colliding_idx] == runway_idx)
        x0_vec[colliding_idx] = sample_aircraft(runway_idx, same_runway_before)

    x0 = np.vstack(x0_vec).T
    J_R = np.eye(m) * 0.5
    pair_count = N * (N - 1) // 2
    rows_per_agent = 2 * default.runway_count + 2 + 4
    n_h = (pair_count + rows_per_agent * N) * T

    config = AirTrafficControlCasadiConfig(
        T=T,
        dt=default.dt,
        N=N,
        n=n,
        m=m,
        n_h=n_h,
        n_c=default.n_c,
        variational_gne=variational_gne,
        V_min=default.V_min,
        V_max=default.V_max,
        a_max=default.a_max,
        omega_max=default.omega_max,
        runway_count=default.runway_count,
        runway_positions=default.runway_positions.copy(order='F'),
        runway_headings=default.runway_headings.copy(order='F'),
        alpha=default.alpha,
        rho_long=default.rho_long,
        rho_lat=default.rho_lat,
        D_min=default.D_min,
        delta_psi=default.delta_psi,
        delta_y=default.delta_y,
        M_lat=default.M_lat,
        M_psi=default.M_psi,
        beta=default.beta,
        distance_y_weight=default.distance_y_weight,
        distance_cost_scale=default.distance_cost_scale,
        landing_gate_M=default.landing_gate_M,
        runway_visual_length=default.runway_visual_length,
        aircraft_visual_length=default.aircraft_visual_length,
        x0=x0.copy(order='F'),
        J_R=J_R.copy(order='F'),
    )
    return AirTrafficControlCasadi(config)
