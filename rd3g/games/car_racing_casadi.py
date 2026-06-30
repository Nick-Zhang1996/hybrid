""" Car Racing game, CasADi version """
import os
import logging
from typing import Any
from math import degrees, radians
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.animation import FuncAnimation
import casadi as cas

from buzzracer.types import CurvilinearState
from buzzracer.tracks.curvilinear_track import CurvilinearTrack
from buzzracer.tracks.survey_track import SurveyTrack
from buzzracer.cars.car_param import CarConfig
from buzzracer.tracks.track_factory import TrackFactory

from rd3g.utilities.util import BASEDIR, resolve_logname
from rd3g.core.casadi_game import CasadiGame, CasadiGameConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class _ShiftedRacelineTrack:
    """Track-like wrapper used to build Stanley-seeded initial guesses."""

    def __init__(self,
                 r_vec,
                 s_vec,
                 phi_vec,
                 curvature_vec,
                 left_width_vec,
                 right_width_vec,
                 speed_vec):
        self.r_vec = r_vec
        self.s_vec = s_vec
        self.phi_vec = phi_vec
        self.curvature_vec = curvature_vec
        self.left_width_vec = left_width_vec
        self.right_width_vec = right_width_vec
        self.speed_vec = speed_vec

    @classmethod
    def from_track(cls, track, requested_shift, boundary_buffer):
        """Build a boundary-aware shifted copy of the track raceline."""
        data = track.data
        applied_shift = np.full_like(data.left_width_vec, requested_shift, dtype=float)
        if requested_shift >= 0:
            max_shift = np.maximum(data.left_width_vec - boundary_buffer, 0.0)
            applied_shift = np.minimum(applied_shift, max_shift)
        else:
            max_shift = np.maximum(data.right_width_vec - boundary_buffer, 0.0)
            applied_shift = -np.minimum(-applied_shift, max_shift)

        lateral = np.column_stack((
            np.cos(data.phi_vec + np.pi / 2.0),
            np.sin(data.phi_vec + np.pi / 2.0)))
        shifted_points = data.r_vec + lateral * applied_shift[:, np.newaxis]
        tangent = np.roll(shifted_points, -1, axis=0) - shifted_points
        shifted_heading = np.arctan2(tangent[:, 1], tangent[:, 0])
        left_width = data.left_width_vec - applied_shift
        right_width = data.right_width_vec + applied_shift

        return cls(
            r_vec=shifted_points,
            s_vec=data.s_vec,
            phi_vec=shifted_heading,
            curvature_vec=data.curvature_vec,
            left_width_vec=left_width,
            right_width_vec=right_width,
            speed_vec=data.speed_vec,
        )

    def local_trajectory(self, state):
        """Mimic track.local_trajectory() on the shifted geometry."""
        from buzzracer.tracks.track import LocalTrajOutput

        x = state.x
        y = state.y

        dxx = self.r_vec[:, 0] - x
        dyy = self.r_vec[:, 1] - y
        index = int(np.argmin(dxx**2 + dyy**2))
        point_count = len(self.r_vec)

        dr = self.r_vec[(index + 1) % point_count] - self.r_vec[index]
        dr_norm = np.linalg.norm(dr)
        if dr_norm < 1e-9:
            track_tangent = np.array([np.cos(self.phi_vec[index]), np.sin(self.phi_vec[index])])
        else:
            track_tangent = dr / dr_norm
        track_to_car = (x - self.r_vec[index, 0], y - self.r_vec[index, 1])
        offset = track_tangent[0] * track_to_car[1] - track_tangent[1] * track_to_car[0]
        left_margin = self.left_width_vec[index] - offset
        right_margin = self.right_width_vec[index] + offset
        return LocalTrajOutput(
            ref_point=self.r_vec[index],
            lateral_err=offset,
            raceline_dir=self.phi_vec[index],
            curvature=self.curvature_vec[index],
            v_target=self.speed_vec[index],
            progress=self.s_vec[index],
            left_margin=left_margin,
            right_margin=right_margin,
        )


@dataclass(frozen=False)
class CarRacingCasadiConfig(CasadiGameConfig):
    """ Base Class for game configuration. """
    T: int = 20
    dt: float = 0.02
    N: int = 4
    n: int = 5
    m: int = 2
    n_h: int = 4 * (4 * 3 // 2) * 20 + 2 * 4 * 20
    """ Total number of canonical inequality constraints. """
    n_c: int = 3  # Size of context variable for per agent per stage
    variational_gne: bool = True
    """ If True, use one shared multiplier per canonical constraint """

    collision_radius: float = 90e-3  # 80e-3
    """ Minimum distance between the origin of two cars"""
    double_circle_h: bool = True
    """ Use two circles instead of one for collision"""
    bdry_margin: float = 0.05
    """ Margin to boundary, use in boundary constraints"""
    use_stanley_control_guess: bool = False
    """ If True, initial_control_guess() seeds steering with a Stanley rollout. """
    stanley_guess_shift_margin: float = 0.04
    """ Lateral shift for left/right Stanley seed racelines. """
    stanley_guess_boundary_buffer: float = 1e-3
    """ Keep shifted Stanley seed racelines inside the track boundary. """

    x0: Any = None
    """ Initial state for all agents, dim: (n,N)"""
    target_x_ref: Any = None
    """ Target state for all agents, dim: (n,N)"""
    J_Qr: Any = None
    """ Cost matrix for tracking reference state dim: (n,n)"""
    J_R: Any = None
    """ Cost matrix for control effort dim: (m,m)"""

    def __post_init__(self):
        assert self.x0.shape == (self.n, self.N), (
            'Incorrect self.x0 dimension, '
            f'should be {(self.n, self.N)}, but got {self.x0.shape}')
        assert self.target_x_ref.shape == (self.n, self.N)
        assert self.J_Qr.shape == (self.n, self.n)
        assert self.J_R.shape == (self.m, self.m)
        return super().__post_init__()


class CarRacingCasadi(CasadiGame):
    ''' Car Racing Game with Kinematic Bicycle Model, with CasADi
        u = [steering, throttle]
        x = [s, n, phi, v_forward, v_sideway]
        s: progress along raceline/reference curve
        v: velocity
        n: lateral offset from ref curve, left positive
        phi: heading from ref curve tangend, ccw positive

        agent count: N, time step: 1..T+1
        X (game state) = concatenated state, first by agent, then by time)
        state p of agent i at time k: X[k,i,p] or X.flatten()[k*N*m + i*m + p]
        U (control) = concatenated control  dim: T*N*m
        control p of agent i at time k: U[k,i,p] or U.flatten()[k*N*m + i*m + p]
        x_i_k: 1..T, T*N*n  NOTE starts from 1
        u_i_k: 0..T-1, T*N*m
        lamda_i_k: 0..T-1 T*N*n
        mu_k_i_j: 1..T T*N*N NOTE starts from 1
    '''

    def __init__(self, config: CarRacingCasadiConfig, track: CurvilinearTrack):
        self.track = track
        self.car_param = CarConfig.porsche_18.value
        super().__init__(config)

        # bounds for visualization

        self.visual_x_lim = [track.data.x_min, track.data.x_max]
        self.visual_y_lim = [track.data.y_min, track.data.y_max]

        # Image wheelbase: 300px, car wheelbase 98e-3 m
        self.car_scale = 98e-3 / 300

        # Curvature function
        s_vec = track.data.s_vec
        c_vec = track.data.curvature_vec
        left_vec = track.data.left_width_vec-self.config.bdry_margin
        right_vec = track.data.right_width_vec-self.config.bdry_margin
        # To handle wrap around when s is outside [0, track.data.track_len_m] Extend the domain
        max_s = s_vec[-1]
        s_vec = np.hstack([s_vec[:-1] - max_s, s_vec, s_vec[1:] + max_s]).tolist()
        c_vec = np.hstack([c_vec[:-1], c_vec, c_vec[1:]]).tolist()
        left_vec = np.hstack([left_vec[:-1], left_vec, left_vec[1:]]).tolist()
        right_vec = np.hstack([right_vec[:-1], right_vec, right_vec[1:]]).tolist()

        assert np.all(np.diff(np.asarray(s_vec)) > 0), "s_vec must be monotonic"
        assert not np.isnan(np.asarray(c_vec)).any(), "No NaNs in curvature"
        assert not np.isnan(np.asarray(left_vec)).any(), "No NaNs in left width"
        assert not np.isnan(np.asarray(right_vec)).any(), "No NaNs in right width"

        # Arguments: (name, plugin, grid, values)
        # 'bspline' creates a cubic B-spline by default, ensuring smooth gradients.
        # 'linear' creates a linear lookup, simplifying gradient
        self.curvature_fun = cas.interpolant('curvature', 'bspline', [s_vec], c_vec)
        self.left_width_fun = cas.interpolant('left_width', 'bspline', [s_vec], left_vec)
        self.right_width_fun = cas.interpolant('right_width', 'bspline', [s_vec], right_vec)

        color_names = [
            'purple', 'yellow', 'red', 'green', 'orange', 'pink', 'cyan', 'hot_pink'
        ]
        self.car_img_vec = [
            mpimg.imread(
                os.path.join(BASEDIR, 'rd3g', 'resources',
                             f'porsche_{color}.png')) for color in color_names
        ]

    def initial_control_guess(self, use_stanley=None):
        """Return a zero guess or Stanley-seeded steering guess, shape (m*N,T)."""
        gc = self.config
        if use_stanley is None:
            use_stanley = gc.use_stanley_control_guess

        u_ref = np.zeros((gc.m * gc.N, gc.T), dtype=float, order='F')
        if not use_stanley:
            return u_ref

        from buzzracer.controllers.stanley_controller import (
            StanleyController,
            StanleyControllerConfig,
            StanleyControllerState,
        )
        from buzzracer.sysid.kinematic_bicycle_model import KinematicBicycleModelCartesian
        from buzzracer.types import Control

        shifted_tracks = self._make_stanley_initial_guess_tracks()
        u_ref_3d = u_ref.reshape((gc.m, gc.N, gc.T), order='F')
        stanley_config = StanleyControllerConfig(SimpleNamespace(dt=gc.dt), self.car_param)
        dummy_main_state = SimpleNamespace(car_target_v=np.zeros(gc.N, dtype=float))

        for i in range(gc.N):
            curv_state = np.array(gc.x0[:, i], dtype=float, copy=True)
            curv_state[3] = np.clip(curv_state[3], a_min=0.5, a_max=None)
            cart_state = self.track.curv_to_cart(CurvilinearState(
                progress=curv_state[0],
                lateral_err=curv_state[1],
                heading_err=curv_state[2],
                v_forward=curv_state[3],
                v_sideway=curv_state[4],
                rel_omega=0.0,
            ))
            stanley_state = StanleyControllerState(stanley_config)
            ref_track = shifted_tracks[self._stanley_guess_track_name(curv_state)]
            for k in range(gc.T):
                ctrl, _, stanley_state, _ = StanleyController.control(
                    cart_state,
                    self.car_param,
                    ref_track,
                    stanley_config,
                    stanley_state,
                    dummy_main_state,
                    i,
                )
                seed_ctrl = Control(ctrl.steering, 0.0)
                u_ref_3d[:, i, k] = seed_ctrl.to_tuple()
                cart_state = KinematicBicycleModelCartesian.advance_dynamics(
                    cart_state,
                    seed_ctrl,
                    self.car_param,
                    gc.dt,
                    simple_throttle=True,
                )

        return u_ref

    def _make_stanley_initial_guess_tracks(self):
        """Create left/right shifted racelines for the Stanley seed."""
        return {
            'left_raceline': _ShiftedRacelineTrack.from_track(
                self.track,
                self.config.stanley_guess_shift_margin,
                self.config.stanley_guess_boundary_buffer,
            ),
            'right_raceline': _ShiftedRacelineTrack.from_track(
                self.track,
                -self.config.stanley_guess_shift_margin,
                self.config.stanley_guess_boundary_buffer,
            ),
        }

    @staticmethod
    def _stanley_guess_track_name(curv_state):
        """Pick the seed raceline based on the current lateral offset."""
        return 'left_raceline' if curv_state[1] >= 0.0 else 'right_raceline'

    def visualize(self, u, x, show=True, save=False):
        """ Visualize the game with given initial state (x0) and control (u) in a single frame.
        Args:
            u: (m,N,T)
            x: (n, N, T+1)
        """
        n = self.config.n
        m = self.config.m
        T = self.config.T
        N = self.config.N
        track = self.track

        assert u.shape == (m, N, T)
        assert x.shape == (n, N, T + 1)

        fig, ax = plt.subplots()
        p = track.data.left_boundary_vec
        ax.plot(p[:, 0], p[:, 1], color='k')
        p = track.data.right_boundary_vec
        ax.plot(p[:, 0], p[:, 1], color='k')
        p = track.data.r_vec
        ax.plot(p[:, 0], p[:, 1], color='g')

        for i in range(N):
            cart_traj = []
            for k in range(T + 1):
                curv = CurvilinearState(progress=x[0, i, k],
                                        lateral_err=x[1, i, k],
                                        heading_err=x[2, i, k],
                                        v_forward=x[3, i, k],
                                        v_sideway=x[4, i, k],
                                        rel_omega=0.0)
                cart = track.curv_to_cart(curv)
                cart_traj.append(cart)

            xx = [e.x for e in cart_traj]
            yy = [e.y for e in cart_traj]
            ax.plot(xx, yy, '*-')
        ax.set_aspect('equal', adjustable='box')
        if save:
            filename = resolve_logname(suffix='gif')
            fig.savefig(filename)
            logger.info('saved figure to %s', filename)
        if show:
            plt.show()
        return ax

    def visualize_rcp(self, u, x, show=True, save=False):
        """Visualize trajectories with the BuzzRacer image renderer.

        This creates a multiple-exposure photo on top of the track image, with
        older car poses drawn more transparently than newer ones.

        Args:
            u: (m,N,T), optional for steering overlays
            x: (n,N,T+1)
        """
        from buzzracer.common import BASEDIR as BUZZRACER_BASEDIR
        from buzzracer.extensions.visualization import Visualization
        import cv2

        gc = self.config
        if (not show) and (not save):
            return None

        if u is not None:
            u = np.asarray(u)
            assert u.shape == (gc.m, gc.N, gc.T)
        x = np.asarray(x)
        assert x.shape == (gc.n, gc.N, gc.T + 1)

        track = self.track
        cart_traj_vec = []
        for i in range(gc.N):
            cart_traj = []
            for k in range(gc.T + 1):
                curv = CurvilinearState(progress=x[0, i, k],
                                        lateral_err=x[1, i, k],
                                        heading_err=x[2, i, k],
                                        v_forward=x[3, i, k],
                                        v_sideway=x[4, i, k],
                                        rel_omega=0.0)
                cart_traj.append(track.curv_to_cart(curv))
            cart_traj_vec.append(cart_traj)

        base_resolution = track.config.resolution
        track.config.resolution = 4 * base_resolution

        renderer = Visualization.__new__(Visualization)
        renderer.config = SimpleNamespace(car_graphics=True)
        renderer.main = SimpleNamespace(track=track)
        renderer.track = track

        car_choices = [
            CarConfig.porsche_18.value,
            CarConfig.porsche_19.value,
            CarConfig.audi_20.value,
            CarConfig.mclaren_21.value,
            CarConfig.mclaren_22.value,
            CarConfig.lambo_13.value,
            CarConfig.corvette_17.value,
            CarConfig.audi_12.value,
        ]
        cars = []
        for i in range(gc.N):
            param = car_choices[i % len(car_choices)]
            filename = os.path.join(BUZZRACER_BASEDIR, 'assets', param.rendering)
            base_image = cv2.imread(filename, -1)
            if base_image is None:
                raise FileNotFoundError(f'Failed to load car image from {filename}')
            if base_image.ndim == 3 and base_image.shape[2] == 3:
                alpha = 255 * np.ones(base_image.shape[:2] + (1,), dtype=base_image.dtype)
                base_image = np.concatenate([base_image, alpha], axis=2)
            cars.append(SimpleNamespace(param=param,
                                        steering=0.0,
                                        state=cart_traj_vec[i][0],
                                        image=base_image.copy(),
                                        base_image=base_image))

        try:
            img = track.draw_track()
            alpha_vec = np.linspace(0.15, 1.0, gc.T + 1)
            for k, alpha in enumerate(alpha_vec):
                for i, car in enumerate(cars):
                    car.state = cart_traj_vec[i][k]
                    car.steering = 0.0 if u is None else float(u[0, i, min(k, gc.T - 1)])
                    car.image = car.base_image.copy()
                    car.image[:, :, 3] = np.clip(
                        car.image[:, :, 3].astype(float) * alpha, 0.0, 255.0).astype(np.uint8)
                    img = renderer.draw_car(img, car)

            if save:
                filename = resolve_logname(prefix='racing_rcp', suffix='png')
                cv2.imwrite(filename, img)
                logger.info('saved figure to %s', filename)

            if show:
                display = cv2.cvtColor(
                    img, cv2.COLOR_BGRA2RGBA if img.shape[2] == 4 else cv2.COLOR_BGR2RGB)
                fig, ax = plt.subplots()
                ax.imshow(display)
                ax.set_axis_off()
                plt.show()
        finally:
            track.config.resolution = base_resolution

        return img

    def animate(self, u, x, show=True, save_gif=False, save_snapshots=False):
        """ Animate the game with given initial state (x0) and control (u).
        Args:
            u: (m,N,T)
            x: (n, N, T+1)
        """
        n = self.config.n
        m = self.config.m
        T = self.config.T
        N = self.config.N
        track = self.track
        car_imgs = self.car_img_vec

        if (not show) and (not save_gif) and (not save_snapshots):
            return
        assert u.shape == (m, N, T)
        assert x.shape == (n, N, T + 1)

        car_scale = self.car_scale
        cart_traj_vec = []
        for i in range(N):
            cart_traj = []
            for k in range(T + 1):
                curv = CurvilinearState(progress=x[0, i, k],
                                        lateral_err=x[1, i, k],
                                        heading_err=x[2, i, k],
                                        v_forward=x[3, i, k],
                                        v_sideway=x[4, i, k],
                                        rel_omega=0.0)
                cart = track.curv_to_cart(curv)
                cart_traj.append(cart)
            cart_traj_vec.append(cart_traj)

        fig, ax = plt.subplots()
        p = track.data.left_boundary_vec
        ax.plot(p[:, 0], p[:, 1], color='k')
        p = track.data.right_boundary_vec
        ax.plot(p[:, 0], p[:, 1], color='k')
        p = track.data.r_vec
        ax.plot(p[:, 0], p[:, 1], color='g')

        im_vec = []
        for i in range(self.config.N):
            rotated_car_img = rotate(car_imgs[i % len(car_imgs)],
                                     degrees(cart_traj_vec[i][0].heading),
                                     reshape=True)
            rotated_car_img = np.clip(rotated_car_img, 0.0, 1.0)
            L, W, _ = rotated_car_img.shape
            im = ax.imshow(rotated_car_img,
                           extent=[
                               cart_traj_vec[i][0].x - W / 2 * car_scale,
                               cart_traj_vec[i][0].x + W / 2 * car_scale,
                               cart_traj_vec[i][0].y - L / 2 * car_scale,
                               cart_traj_vec[i][0].y + L / 2 * car_scale
                           ])
            im_vec.append(im)

        def update(frame):
            for i in range(self.config.N):
                rotated_car_img = np.clip(
                    rotate(car_imgs[i % len(car_imgs)],
                           degrees(cart_traj_vec[i][frame].heading),
                           reshape=True), 0.0, 1.0)
                L, W, _ = rotated_car_img.shape
                im_vec[i].set_data(rotated_car_img)
                im_vec[i].set_extent(
                    (cart_traj_vec[i][frame].x - W / 2 * car_scale,
                     cart_traj_vec[i][frame].x + W / 2 * car_scale,
                     cart_traj_vec[i][frame].y - L / 2 * car_scale,
                     cart_traj_vec[i][frame].y + L / 2 * car_scale))
            return im_vec

        # fine-tune dark background to mimic tarmac
        # Set the background color of the plot (axes background)
        ax.set_facecolor((54 / 255, 69 / 255, 79 / 255))

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(*self.visual_x_lim)
        ax.set_ylim(*self.visual_y_lim)

        # Create the animation
        anim = FuncAnimation(fig, update, frames=self.config.T, blit=False)

        folder = os.path.join(BASEDIR, 'outputs', 'gifs')
        os.makedirs(folder, exist_ok=True)
        gif_filename = os.path.join(folder, f'racing_{self.config.N}car.gif')
        if save_gif:
            anim.save(gif_filename, writer='pillow')
            logger.info(f'Gif saved to {gif_filename}')
        if show:
            plt.show()
        if show and save_snapshots:
            logger.error('When show and save_snapshots are both on,'
                         ' matplotlib has weird problems, do one at a time')
        # NOTE save initial, middle, final snapshots
        if save_snapshots:
            from PIL import Image  # pylint: disable=import-outside-toplevel
            folder = os.path.join(BASEDIR, 'outputs', 'pics')
            os.makedirs(folder, exist_ok=True)
            update(0)
            fig.canvas.draw()
            frame = Image.frombytes('RGB', fig.canvas.get_width_height(),
                                    fig.canvas.tostring_rgb())
            filename = os.path.join(folder, f'merge_{self.config.N}car_initial.png')
            frame.save(filename)
            logger.info(f'saved snapshots to {filename}')

            update(self.config.T // 2)
            fig.canvas.draw()
            frame = Image.frombytes('RGB', fig.canvas.get_width_height(),
                                    fig.canvas.tostring_rgb())
            filename = os.path.join(folder, f'merge_{self.config.N}car_middle.png')
            frame.save(filename)
            logger.info(f'saved snapshots to {filename}')

            update(self.config.T - 1)
            fig.canvas.draw()
            frame = Image.frombytes('RGB', fig.canvas.get_width_height(),
                                    fig.canvas.tostring_rgb())
            filename = os.path.join(folder, f'merge_{self.config.N}car_final.png')
            frame.save(filename)
            logger.info(f'saved snapshots to {filename}')
        return

    def get_context(self, x_k_i):
        """ Calculate game context for one agent at one step
        Args:
            x_k_i: (n,)
        Return:
            context: (n_c=3, ) curvature, lateral offset low_bound, high bound
        """
        return cas.vertcat(self.curvature_fun(x_k_i[0]),
                           self.left_width_fun(x_k_i[0]),
                           self.right_width_fun(x_k_i[0]))

    def J(self, x_k, u_k_i, i_onehot):
        """
        Stage cost for an agent, given x,u
        x_k.shape (n,N) x_k_i
        u_k_i.shape (m,1) u_k_i
        i_onehot: (N,1) agent id in one-hot encoding, i.e. i=1,N=4 -> [0,1,0,0], column vector
        """
        assert x_k.shape == (self.config.n, self.config.N)
        assert u_k_i.shape == (self.config.m, 1)
        assert i_onehot.shape == (self.config.N, 1)

        target_x_ref = self.config.get_param('target_x_ref')
        J_Qr = self.config.get_param('J_Qr')
        J_R = self.config.get_param('J_R')
        x_k_i = x_k @ i_onehot  # dim: n,1
        dx = x_k_i - target_x_ref @ i_onehot

        val = dx.T @ J_Qr @ dx + u_k_i.T @ J_R @ u_k_i
        return val

    def Jfi(self, x_T, i_onehot):
        """ Final cost"""
        return self.J(x_T, cas.SX.zeros(self.config.m), i_onehot)

    def f(self, x_k_i, u_k_i, i_onehot, context_i_k):
        """ Dynamics function x_{t+1} = f(x_t,u,i) for kinematic bicycle, frenet
            u = [steering, throttle]
            x = [s, n, phi, v_forward, v_sideway]
            NOTE: 0 < s < track.data.raceline_len_m
        Args:
            x_k_i: (n,1) State for agent i
            u_k_i: (m,1) Control for agent i
            i_onehot: agent id, in one-hot encoding (N), i.e. i=1,N=4 -> [0,1,0,0], column vector
            context_i_k: (n_c=3,) Signed curvature value, left, right margin
        Return:
            (n,1) The next state, progressed by self.config.dt

        this problem has homogeneous agents, so [i] is irrelevant"""
        param = self.car_param
        assert x_k_i.shape == (self.config.n, 1)
        assert u_k_i.shape == (self.config.m, 1)
        assert i_onehot.shape == (self.config.N, 1)
        state = SimpleNamespace(progress=x_k_i[0, 0],
                                lateral_err=x_k_i[1, 0],
                                heading_err=x_k_i[2, 0],
                                v_forward=x_k_i[3, 0],
                                v_sideway=x_k_i[4, 0])
        control = SimpleNamespace(steering=u_k_i[0, 0], throttle=u_k_i[1, 0])
        # curvature = self.curvature_fun(state.progress)
        curvature = context_i_k[0]

        beta = cas.arctan(
            cas.tan(control.steering) * param.lr / (param.lf + param.lr))

        dsdt = (state.v_forward * cas.cos(state.heading_err) -
                state.v_sideway * cas.sin(state.heading_err)) / (1 - state.lateral_err * curvature)
        dndt = state.v_forward * cas.sin(state.heading_err) + \
            state.v_sideway * cas.cos(state.heading_err)
        # acceleration at rear wheel
        acc_rw = control.throttle * 3.0
        acc_cg = acc_rw / cas.cos(beta)
        d_v_forward_dt = acc_cg * cas.cos(beta)
        d_v_sideway_dt = acc_cg * cas.sin(beta)

        total_v = cas.sqrt(state.v_forward**2 + state.v_sideway**2)
        d_heading_dt = total_v / param.lr * cas.sin(beta)
        d_rel_heading_dt = d_heading_dt - curvature * dsdt

        dt = self.config.get_param('dt')
        progress = state.progress + dsdt * dt
        lateral_err = state.lateral_err + dndt * dt
        heading_err = state.heading_err + d_rel_heading_dt * dt
        v_forward = state.v_forward + d_v_forward_dt * dt
        v_sideway = state.v_sideway + d_v_sideway_dt * dt
        next_x = cas.vertcat(progress, lateral_err, heading_err, v_forward, v_sideway)

        return next_x

    def h(self, x, u, context):
        """ Construct the inequality constraint function.
        h() is a mapping from (x,u) to all constraints.
        Args:
            x: (n*N,T), states, casadi.SX symbolic variable
            u: (m*N,T), controls, casadi.SX symbolic variable
            context: (n_c*N, T), context variable. [curvature, left margin, right margin]
        Returns:
            h_vec:
                variational mode: (n_h, 1)
                non-variational mode: (n_h, N), with -1 for agents not
                participating in a canonical constraint
        """
        gc = self.config
        h_rows = []
        if gc.variational_gne:
            for k in range(1, gc.T + 1):
                context_i_k = cas.reshape(context[:, k-1], gc.n_c, gc.N)
                xk = cas.reshape(x[:, k - 1], gc.n, gc.N)
                for i in range(gc.N):
                    for j in range(i + 1, gc.N):
                        h_rows.append(self.collision_h(xk[:, i], xk[:, j]))
                for i in range(gc.N):
                    h_rows.append(self.boundary_h(xk[:, i], context_i_k[:, i]))

            h_vec = cas.vertcat(*h_rows)
            assert h_vec.shape == (gc.n_h, 1), "n_h must be consistent to h().shape[0]"
            return h_vec

        for agent_idx in range(gc.N):
            hi_rows = []
            for k in range(1, gc.T + 1):
                context_i_k = cas.reshape(context[:, k-1], gc.n_c, gc.N)
                xk = cas.reshape(x[:, k - 1], gc.n, gc.N)
                for i in range(gc.N):
                    for j in range(i + 1, gc.N):
                        if agent_idx == i or agent_idx == j:
                            hi_rows.append(self.collision_h(xk[:, i], xk[:, j]))
                        else:
                            rows_per_collision = 4 if gc.double_circle_h else 1
                            hi_rows.append(-cas.DM.ones(rows_per_collision, 1))
                for i in range(gc.N):
                    if agent_idx == i:
                        hi_rows.append(self.boundary_h(xk[:, i], context_i_k[:, i]))
                    else:
                        hi_rows.append(-cas.DM.ones(2, 1))
            h_rows.append(cas.vertcat(*hi_rows))

        h_vec = cas.horzcat(*h_rows)
        assert h_vec.shape == (gc.n_h, gc.N), "n_h must be consistent to h().shape[0]"
        return h_vec

    def collision_h(self, x_i, x_j):
        """ Collision constraints for x_i, anx x_j agent, h <= 0
        Approximate each car as two tangent circles. Creates 4 constraints in total
        Args:
            x_i: (n,1) State of i at time k
            x_j: (n,1) State of j at time k
        Return:
            h_val: (4,1), h_val <= 0 means no collision
        """
        # car distance larger than 1.2 normalized
        gc = self.config
        assert x_i.shape == (gc.n, 1)
        assert x_j.shape == (gc.n, 1)
        d = self.config.get_param('collision_radius')

        # x = [s, n, phi, v_forward, v_sideway]
        if self.config.double_circle_h:
            offset = 45e-3
            f1x = x_i[0, 0] + offset * cas.cos(x_i[2, 0])
            f1y = x_i[1, 0] + offset * cas.sin(x_i[2, 0])
            f2x = x_j[0, 0] + offset * cas.cos(x_j[2, 0])
            f2y = x_j[1, 0] + offset * cas.sin(x_j[2, 0])
            r1x = x_i[0, 0] - offset * cas.cos(x_i[2, 0])
            r1y = x_i[1, 0] - offset * cas.sin(x_i[2, 0])
            r2x = x_j[0, 0] - offset * cas.cos(x_j[2, 0])
            r2y = x_j[1, 0] - offset * cas.sin(x_j[2, 0])

            FF = -(f1x - f2x)**2 - (f1y - f2y)**2 + d**2
            FR = -(f1x - r2x)**2 - (f1y - r2y)**2 + d**2
            RF = -(r1x - f2x)**2 - (r1y - f2y)**2 + d**2
            RR = -(r1x - r2x)**2 - (r1y - r2y)**2 + d**2
            vals = cas.vertcat(FF, FR, RF, RR)
        else:
            # Single circle check center to center distance
            vals = -((x_i[0, 0] - x_j[0, 0]) / 1.0)**2 - (x_i[1, 0] - x_j[1, 0])**2 + d**2
        return vals

    def boundary_h(self, x, context):
        """ Boundary violation constraints
        Args:
            x: (n,1) State
            context: (n_c, 1) Context, [curvature, left_margin, right_margin]
        Return:
            h_val: (2,1) h_val <=0 means car is 
        """
        h_val = cas.vertcat(x[1] - context[1],  -x[1] - context[2])
        return h_val

    def inspect_h(self, u, x=None, solver=None):
        """ Inspect source of constraint residuals."""
        # Calculate h
        gc = self.config
        int_param_dm = cas.DM(gc.get_int_param_np())
        double_param_dm = cas.DM(gc.get_double_param_np())
        params_dm = [int_param_dm, double_param_dm]
        u = cas.DM(u.reshape(gc.m*gc.N, gc.T, order='F'))
        if x is None:
            x = solver.rollout_casadi(gc.x0, u, *params_dm)
        else:
            x = cas.DM(x.reshape(gc.n*gc.N, gc.T, order='F'))
        context_dm = solver.get_full_context_casadi(x)
        h_val = np.asarray(solver.h_casadi(x, u, context_dm, *params_dm))
        tol = 1e-2
        # Retrieve components
        collision_res = 0
        boundary_res = 0
        h_val = h_val.reshape(gc.n_h, 1 if gc.variational_gne else gc.N, order='F')
        rows_per_collision = 4 if gc.double_circle_h else 1
        row = 0
        for k in range(gc.T):
            for i in range(gc.N):
                for j in range(i + 1, gc.N):
                    cols = [0] if gc.variational_gne else [i, j]
                    col_res = 0.0
                    for col in cols:
                        res = np.linalg.norm(
                            np.clip(h_val[row:row + rows_per_collision, col], a_min=0, a_max=None)
                        )
                        col_res = max(col_res, res)
                    collision_res += col_res**2
                    if col_res > tol:
                        logger.info('car %s, %s, k=%s collision %s', i, j, k + 1, col_res)
                    row += rows_per_collision
            for i in range(gc.N):
                col = 0 if gc.variational_gne else i
                left_bdry_res = np.clip(h_val[row, col], a_min=0, a_max=None)
                right_bdry_res = np.clip(h_val[row + 1, col], a_min=0, a_max=None)
                if left_bdry_res > tol:
                    logger.info('car %s, k=%s left %s', i, k + 1, left_bdry_res)
                if right_bdry_res > tol:
                    logger.info('car %s, k=%s right %s', i, k + 1, right_bdry_res)
                boundary_res += left_bdry_res**2 + right_bdry_res**2
                row += 2
        h_pos_res = np.sum(np.clip(h_val, a_min=0, a_max=None)**2)
        inspected_h_pos_res = collision_res + boundary_res
        logger.info(f'{h_pos_res=}, {inspected_h_pos_res=}')

        return


def create_random_game(car_count=3, horizon=20, track=TrackFactory.build('saved'),
                       variational_gne=True, use_stanley_control_guess=False):
    """ Create a Car Racing Game instance with random initial states"""
    default = CarRacingCasadiConfig
    T = horizon
    N: int = car_count
    n: int = default.n
    m: int = default.m

    # x = [s, n, phi, v_forward, v_sideway]
    # J_Qr = np.diag([0, 5.0, 0.1, 1.0, 0.1])
    # J_R = np.eye(m) * 1.0
    J_Qr = np.diag([0, 5.0, 1.0, 1.0, 0.1])
    J_R = np.eye(m) * 1.0

    # NOTE for generating condensed game
    s_low = 1.2  # original benchmark 0.5
    s_high = 2.7  # original benchmark 4.0
    s_vec = np.random.uniform(low=s_low, high=s_high, size=N)
    v_vec = np.random.uniform(low=0.9, high=1.1, size=N)
    phi_vec = np.random.uniform(low=radians(-5), high=radians(5), size=N)
    # s represent the progress on frenet/curvilinear frame, it's like the x coordinate
    # n represents the lateral deviation on frenet frame, it's like the y coordinate
    left_width_vec = np.interp(s_vec, track.data.s_vec, track.data.left_width_vec)
    right_width_vec = np.interp(s_vec, track.data.s_vec, track.data.right_width_vec)
    n_vec = np.random.uniform(low=-right_width_vec, high=left_width_vec, size=N)
    vs_vec = np.zeros(N)
    collision_threshold_sq = (2.0 * default.collision_radius) ** 2
    while True:
        ds = s_vec[:, None] - s_vec[None, :]
        dn = n_vec[:, None] - n_vec[None, :]
        collision_mask = (ds * ds + dn * dn) < collision_threshold_sq
        np.fill_diagonal(collision_mask, False)
        colliding_idx = np.flatnonzero(collision_mask.any(axis=1))
        if colliding_idx.size == 0:
            break

        s_vec[colliding_idx] = np.random.uniform(low=s_low, high=s_high, size=colliding_idx.size)
        left_width_vec[colliding_idx] = np.interp(
            s_vec[colliding_idx], track.data.s_vec, track.data.left_width_vec
        )
        right_width_vec[colliding_idx] = np.interp(
            s_vec[colliding_idx], track.data.s_vec, track.data.right_width_vec
        )
        n_vec[colliding_idx] = np.random.uniform(
            low=-right_width_vec[colliding_idx],
            high=left_width_vec[colliding_idx],
            size=colliding_idx.size,
        )

    # n, N
    x0 = np.vstack([s_vec, n_vec, phi_vec, v_vec, vs_vec])

    x_ref = np.zeros((n, N))
    x_ref[3, :] = v_vec  # target initial speed
    rows_per_collision = 4 if default.double_circle_h else 1
    n_h = (rows_per_collision * (N * (N - 1) // 2) + 2 * N) * T

    config = CarRacingCasadiConfig(
        T=T,
        dt=default.dt,
        N=N,
        n=n,
        m=m,
        n_h=n_h,
        collision_radius=default.collision_radius,
        variational_gne=variational_gne,
        use_stanley_control_guess=use_stanley_control_guess,
        x0=x0.copy(order='F'),
        target_x_ref=x_ref.copy(order='F'),
        J_Qr=J_Qr.copy(order='F'),
        J_R=J_R.copy(order='F'))
    # track_config = TrackConfig()
    # track = NascarTrack(track_config)
    return CarRacingCasadi(config, track)
