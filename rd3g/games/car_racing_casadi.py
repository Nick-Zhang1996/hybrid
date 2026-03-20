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
from buzzracer.cars.car import CarConfig
from buzzracer.tracks.nascar_track import NascarTrack
from buzzracer.tracks.track_factory import TrackFactory

from rd3g.utilities.util import BASEDIR, resolve_logname
from rd3g.core.casadi_game import CasadiGame, CasadiGameConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass(frozen=True)
class CarRacingCasadiConfig(CasadiGameConfig):
    """ Base Class for game configuration. """
    T: int = 20
    dt: float = 0.05
    N: int = 3
    n: int = 5
    m: int = 2
    n_hi: int = 3 * 20  # Total number of constraints for EACH agent, e.g. pairwise collision only: N*T
    n_s: int = 1  # Number of external states per agent per stage

    collision_radius: float = 80e-3
    """ Minimum distance between two cars"""
    double_circle_h: bool = False
    """ Use two circles instead of one for collision"""

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

        # n_hi is a new concept
        self.n_hi = config.n_hi
        self.n_s = config.n_s

        # bounds for visualization

        self.visual_x_lim = [track.data.x_min, track.data.x_max]
        self.visual_y_lim = [track.data.y_min, track.data.y_max]

        # Image wheelbase: 300px, car wheelbase 98e-3 m
        self.car_scale = 98e-3 / 300

        # Moved to state, outside of casadi because the code generated is too long

        # Curvature function
        s_vec = [track.data.s_vec.tolist()]
        c_vec = track.data.curvature_vec.tolist()

        s = np.array(s_vec[0])
        c = np.array(c_vec)
        assert np.all(np.diff(s) > 0), "s_vec must be monotonic"
        assert not np.isnan(c).any(), "No NaNs"

        # Arguments: (name, plugin, grid, values)
        # 'bspline' creates a cubic B-spline by default, ensuring smooth gradients.
        # 'linear' creates a linear lookup, simplifying gradient
        self.curvature_fun = cas.interpolant('curvature', 'bspline', s_vec, c_vec)

        color_names = [
            'purple', 'yellow', 'red', 'green', 'orange', 'pink', 'cyan',
            'hot_pink'
        ]
        self.car_img_vec = [
            mpimg.imread(
                os.path.join(BASEDIR, 'rd3g', 'resources',
                             f'porsche_{color}.png')) for color in color_names
        ]

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

        if (not show) and (not save):
            return
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
            logger.info(f'saved figure to {filename}')
        if show:
            plt.show()

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
            rotated_car_img = np.clip(
                rotate(car_imgs[i % len(car_imgs)],
                       degrees(cart_traj_vec[i][0].heading),
                       reshape=True), 0.0, 1.0)
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
        anim = FuncAnimation(fig, update, frames=self.T, blit=False)

        folder = os.path.join(BASEDIR, 'gifs')
        gif_filename = os.path.join(folder, f'merge_{self.N}car.gif')
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
            from PIL import Image
            folder = os.path.join(BASEDIR, 'pics')
            update(0)
            fig.canvas.draw()
            frame = Image.frombytes('RGB', fig.canvas.get_width_height(),
                                    fig.canvas.tostring_rgb())
            filename = os.path.join(folder, f'merge_{self.N}car_initial.png')
            frame.save(filename)
            logger.info(f'saved snapshots to {filename}')

            update(self.T // 2)
            fig.canvas.draw()
            frame = Image.frombytes('RGB', fig.canvas.get_width_height(),
                                    fig.canvas.tostring_rgb())
            filename = os.path.join(folder, f'merge_{self.N}car_middle.png')
            frame.save(filename)
            logger.info(f'saved snapshots to {filename}')

            update(self.T - 1)
            fig.canvas.draw()
            frame = Image.frombytes('RGB', fig.canvas.get_width_height(),
                                    fig.canvas.tostring_rgb())
            filename = os.path.join(folder, f'merge_{self.N}car_final.png')
            frame.save(filename)
            logger.info(f'saved snapshots to {filename}')
        return

    def get_state(self, x_k_i):
        """ Calculate game state for one agent at one step
        Args:
            x_k_i: (n,) 
        Return:
            Curvature, scalar
        """
        # c = splev(x_k_i[0], self.track.data.curvature_s, der=0)
        # return c[0].item()
        return self.curvature_fun(x_k_i[0])

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
        return self.J(x_T, cas.SX.zeros(self.m), i_onehot)

    def f(self, x_k_i, u_k_i, i_onehot, state_i_k):
        """ Dynamics function x_{t+1} = f(x_t,u,i)
            u = [steering, throttle]
            x = [s, n, phi, v_forward, v_sideway]
            NOTE: 0 < s < track.data.raceline_len_m
        Args:
            x_k_i: (n,1) State for agent i
            u_k_i: (m,1) Control for agent i
            i_onehot: agent id, in one-hot encoding (N), i.e. i=1,N=4 -> [0,1,0,0], column vector
            state_i_k: (n_s=1,) Signed curvature value
        Return:
            (n,1) The next state, progressed by self.dt

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
        curvature = state_i_k

        beta = cas.arctan(
            cas.tan(control.steering) * param.lr / (param.lf + param.lr))

        dsdt = (state.v_forward * cas.cos(state.heading_err) -
                state.v_sideway * cas.sin(state.heading_err)) / (
                    1 - state.lateral_err * curvature)
        dndt = state.v_forward * cas.sin(state.heading_err) + \
            state.v_sideway * cas.cos(state.heading_err)
        # acceleration at rear wheel
        # TODO new sysid
        # acc_rw = 6.17 * (control.throttle - state.v_forward / 15.2 - 0.333)
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
        next_x = cas.vertcat(progress, lateral_err, heading_err, v_forward,
                             v_sideway)

        return next_x

    def h(self, x, u):
        """ Construct the inequality constraint function.
        h() is a mapping from (x,u) to all constraints.
        Args:
            x: (n*N,T), states, casadi.SX symbolic variable
            u: (m*N,T), controls, casadi.SX symbolic variable
        Returns:
            h_vec: (n_hi, N), constraints vector, sadisfied when h_vec <= 0
        """
        h_vec = []
        for i in range(self.N):
            hi_vec = []
            # Collision constraint collison_h(xi, xj) 4*N*T
            for k in range(1, self.T + 1):
                # collision residual for h > 0
                # x[k] -> x_{k+1} due to index alignment
                xk = cas.reshape(x[:, k - 1], self.n, self.N)
                h_vals = [
                    self.collision_h(xk[:, i], xk[:, j]) for j in range(self.N)
                ]
                # ignore self-collision, but keep this dummy constraint to simplify index counting
                # TODO remove this dummy collision
                if self.config.double_circle_h:
                    h_vals[i] = cas.SX.zeros(4, 1)
                else:
                    h_vals[i] = 0.0
                h_vals = cas.vertcat(*h_vals)
                if self.config.double_circle_h:
                    assert h_vals.shape == (self.N * 4, 1)
                else:
                    assert h_vals.shape == (self.N, 1)
                hi_vec.append(h_vals)  # 4*N, agent i vs everyone (N)
            # Additional constraints for agent i, None here
            h_vec.append(cas.vertcat(*hi_vec))  # 4*N*T

        h_vec = cas.horzcat(*h_vec)
        assert h_vec.shape == (
            self.n_hi, self.N), "n_hi must be consistent to h().shape[0]"
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
        assert x_i.shape == (self.n, 1)
        assert x_j.shape == (self.n, 1)
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


def create_random_game(car_count=3, horizon=20):
    """ Create a Car Racing Game instance with random initial states"""
    default = CarRacingCasadiConfig
    T = horizon
    N: int = car_count
    n: int = default.n
    m: int = default.m

    # x = [s, n, phi, v_forward, v_sideway]
    J_Qr = np.diag([0, 5.0, 0.1, 1.0, 0.1])
    J_R = np.eye(m) * 1.0

    # TODO resample if cars collide
    s_vec = np.random.uniform(low=0.0, high=2.0, size=N)
    v_vec = np.random.uniform(low=0.5, high=2.0, size=N)
    phi_vec = np.random.uniform(low=radians(-5), high=radians(5), size=N)
    n_vec = np.random.uniform(low=-0.1, high=0.1, size=N)
    vs_vec = np.zeros(N)

    # n, N
    x0 = np.vstack([s_vec, n_vec, phi_vec, v_vec, vs_vec])
    x_ref = np.zeros((n, N))
    x_ref[3, :] = v_vec  # target initial speed

    config = CarRacingCasadiConfig(
        T=T,
        dt=default.dt,
        N=N,
        n=n,
        m=m,
        n_hi=4 * N * T if default.double_circle_h else N *
        T,  # Collision constraint only
        collision_radius=default.collision_radius,
        x0=x0.copy(order='F'),
        target_x_ref=x_ref.copy(order='F'),
        J_Qr=J_Qr.copy(order='F'),
        J_R=J_R.copy(order='F'))
    # track_config = TrackConfig()
    # track = NascarTrack(track_config)
    track = TrackFactory.build('saved')
    return CarRacingCasadi(config, track)
