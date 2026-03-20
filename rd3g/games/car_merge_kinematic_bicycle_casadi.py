""" CasADi version """
import os
import logging
from typing import Any
from math import degrees
from dataclasses import dataclass

import numpy as np
from scipy import interpolate
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.animation import FuncAnimation
import casadi as cas

from rd3g.utilities.util import BASEDIR, resolve_logname
from rd3g.core.casadi_game import CasadiGame, CasadiGameConfig

logger = logging.getLogger('CarMergeKinematicBicycle')
logger.setLevel(logging.INFO)


@dataclass(frozen=True)
class CarMergeKinematicBicycleCasadiConfig(CasadiGameConfig):
    """ Base Class for game configuration"""
    T: int = 0
    dt: float = 0.2
    N: int = 3
    n: int = 4
    m: int = 2
    n_hi: int = 0  # Total number of constraints for EACH agent, e.g. pairwise collision only: N*T
    n_s: int = 0  # Number of external states per agent per stage, unused in this game
    track_width: float = 2.2
    collision_radius: float = 2.0

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


class CarMergeKinematicBicycleCasadi(CasadiGame):
    ''' Kinematic Bicycle Merging Game, with CasADi
        u = [throttle, steering]
        x = [x,y,v,theta]: x: upwards, y:leftward, theta: ccw (right hand coord)
        collision constraint: [(xi-xj)/dx]**2 + [(yi-yj)/dy]**2 >= 1
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

    def __init__(self, config: CarMergeKinematicBicycleCasadiConfig):
        super().__init__(config)

        # n_hi is a new concept
        self.n_hi = config.n_hi
        # bounds for visualization
        self.visual_x_lim = [-2.5, 2.5]
        self.visual_y_lim = [-2, 30]
        # animation/visualization related
        self.sprite_visualization = True  # True would use car images instead of boaxes

        if self.sprite_visualization:
            self.car_scale = 0.0045 / 2  # 0.005/2
            color_names = [
                'purple', 'yellow', 'red', 'green', 'orange', 'pink', 'cyan',
                'hot_pink'
            ]
            self.car_img_vec = [
                mpimg.imread(
                    os.path.join(BASEDIR, 'rd3g', 'resources', f'porsche_{color}.png'))
                for color in color_names
            ]

        # collision definition
        # this is not a parameter so not in config
        self.h_Qh = np.diag([-1.0, -1, 0, 0])

    def visualize(self, u, x, show=True, save=False):
        """ Visualize the game with given initial state (x0) and control (u) in a single frame.
        Args:
            u: (m,N,T)
            x: (n, N, T+1)
        """
        n = self.n
        m = self.m
        T = self.T
        N = self.N

        if (not show) and (not save):
            return
        assert u.shape == (m, N, T)
        assert x.shape == (n, N, T+1)

        fig, ax = plt.subplots()
        ax.vlines(x=-self.config.track_width,
                  ymin=self.visual_y_lim[0],
                  ymax=self.visual_y_lim[1])
        ax.vlines(x=self.config.track_width,
                  ymin=self.visual_y_lim[0],
                  ymax=self.visual_y_lim[1])
        # dotted line
        for i in np.linspace(self.visual_y_lim[0], self.visual_y_lim[1], 10):
            ax.vlines(x=0, ymin=i, ymax=i + 0.5)

        for i in range(self.N):
            xx = x[0, i, :]
            yy = x[1, i, :]
            ax.plot(-yy, xx, '*-')
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
        n = self.n
        m = self.m
        T = self.T
        N = self.N
        if (not show) and (not save_gif) and (not save_snapshots):
            return
        assert u.shape == (m, N, T)
        assert x.shape == (n, N, T+1)
        fig, ax = plt.subplots()

        if self.sprite_visualization:
            car_scale = self.car_scale
            # draw car sprite
            car_pose_vec = []
            for k in range(T+1):
                states = x[:, :, k]
                car_pose_vec.append(
                    [[-states[1][i], states[0][i], states[3][i] + np.pi / 2]
                     for i in range(self.config.N)])

            im_vec = []
            for i in range(self.config.N):
                rotated_car_img = np.clip(
                    rotate(self.car_img_vec[i % len(self.car_img_vec)],
                           degrees(car_pose_vec[0][i][2]),
                           reshape=True), 0.0, 1.0)
                L, W, _ = rotated_car_img.shape
                im = ax.imshow(rotated_car_img,
                               extent=[
                                   car_pose_vec[0][i][0] - W * car_scale,
                                   car_pose_vec[0][i][0] + W * car_scale,
                                   car_pose_vec[0][i][1] - L * car_scale,
                                   car_pose_vec[0][i][1] + L * car_scale
                               ])
                im_vec.append(im)

            def update(frame):
                for i in range(self.config.N):
                    rotated_car_img = np.clip(
                        rotate(self.car_img_vec[i % len(self.car_img_vec)],
                               degrees(car_pose_vec[frame][i][2]),
                               reshape=True), 0.0, 1.0)
                    L, W, _ = rotated_car_img.shape
                    im_vec[i].set_data(rotated_car_img)
                    im_vec[i].set_extent(
                        (car_pose_vec[frame][i][0] - W * car_scale,
                         car_pose_vec[frame][i][0] + W * car_scale,
                         car_pose_vec[frame][i][1] - L * car_scale,
                         car_pose_vec[frame][i][1] + L * car_scale))
                return im_vec
        else:
            # Show cars as rectangular blocks
            car_pos_vec = []
            car_angle_vec = []
            box_vec = []
            circle_vec = []
            color_vec = ['red', 'green', 'blue', 'black']
            color_vec = [color_vec[i % len(color_vec)] for i in range(self.config.N)]
            # prepare smoothed animation
            for i, color in zip(range(self.config.N), color_vec):
                # interpolate for smooth graphics
                # tt = np.linspace(0,self.config.T*self.config.dt,50)
                tt = np.linspace(0, self.config.dt * self.config.T, self.config.T + 1)
                # for plt.Rectangle, we offset position so this corresponds to top left corner
                # also flip x axis
                xx = x[0, i, :] - 1.0
                yy = -(x[1, i, :]) - 0.5
                angle = x[3, i, :]
                xx_fun = interpolate.interp1d(tt, xx)
                yy_fun = interpolate.interp1d(tt, yy)
                angle_fun = interpolate.interp1d(tt, angle)

                pos_vec = np.vstack([yy_fun(tt), xx_fun(tt)]).T
                angle_vec = angle_fun(tt) / np.pi * 180.0
                car_angle_vec.append(angle_vec)
                car_pos_vec.append(pos_vec)
                box_vec.append(
                    plt.Rectangle(pos_vec[0],
                                  1,
                                  2,
                                  angle=angle_vec[0],
                                  color=color,
                                  rotation_point='center'))
                circle_vec.append(
                    plt.Circle(pos_vec[0] + np.array([0.5, 1.0]),
                               radius=(self.config.collision_radius) / 2,
                               color=color,
                               fill=False))

            def update(frame):
                for i in range(self.N):
                    box_vec[i].set_xy(car_pos_vec[i][frame])
                    box_vec[i].set_angle(car_angle_vec[i][frame])
                    circle_vec[i].set_center(car_pos_vec[i][frame] +
                                             np.array([0.5, 1.0]))
                return box_vec

            # Add the boxes to the plot
            for box in box_vec:
                ax.add_patch(box)
            for circ in circle_vec:
                ax.add_patch(circ)

        # fine-tune dark background to mimic tarmac
        # Set the background color of the plot (axes background)
        ax.set_facecolor((54 / 255, 69 / 255, 79 / 255))

        # lane markings
        # boundary lines
        ax.vlines(x=-self.config.track_width,
                  ymin=self.visual_y_lim[0],
                  ymax=self.visual_y_lim[1],
                  colors='white')
        ax.vlines(x=self.config.track_width,
                  ymin=self.visual_y_lim[0],
                  ymax=self.visual_y_lim[1],
                  colors='white')
        # dotted line
        for i in np.linspace(self.visual_y_lim[0], self.visual_y_lim[1], 20):
            ax.vlines(x=0, ymin=i, ymax=i + 1, colors='white')

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
            logger.error(
                'When show and save_snapshots are both on,'
                ' matplotlib has weird problems, do one at a time')
        # NOTE save initial, middle, final snapshots
        if save_snapshots:
            from PIL import Image
            folder = os.path.join(BASEDIR, 'pics')
            update(0)
            fig.canvas.draw()
            frame = Image.frombytes('RGB',
                                    fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
            filename = os.path.join(folder, f'merge_{self.N}car_initial.png')
            frame.save(filename)
            logger.info(f'saved snapshots to {filename}')

            update(self.T//2)
            fig.canvas.draw()
            frame = Image.frombytes('RGB',
                                    fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
            filename = os.path.join(folder, f'merge_{self.N}car_middle.png')
            frame.save(filename)
            logger.info(f'saved snapshots to {filename}')

            update(self.T-1)
            fig.canvas.draw()
            frame = Image.frombytes('RGB',
                                    fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
            filename = os.path.join(folder, f'merge_{self.N}car_final.png')
            frame.save(filename)
            logger.info(f'saved snapshots to {filename}')
        return

    def J(self, x_k, u_k_i, i_onehot):
        """
        Stage cost for an agent, given
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
        M = np.array([[1, 0, 0, 0]], order='F')  # matrix to pick out x coord

        val = dx.T @ J_Qr @ dx + u_k_i.T @ J_R @ u_k_i + M @ x_k_i - cas.sum(M @ x_k)
        return val

    def Jfi(self, x_T, i_onehot):
        """ Final cost"""
        return self.J(x_T, cas.SX.zeros(self.m), i_onehot)

    def f(self, x_k_i, u_k_i, i_onehot, state_i_k):
        """ Dynamics function x_{t+1} = f(x_t,u,i)
        Args:
            x_k_i: (n,1) State for agent i
            u_k_i: (m,1) Control for agent i
            i_onehot: agent id, in one-hot encoding (N), i.e. i=1,N=4 -> [0,1,0,0], column vector
            state_i_k: (n_s=0,), unused
        Return:
            (n,1) The next state, progressed by self.dt

        this problem has homogeneous agents, so [i] is irrelevant"""
        assert x_k_i.shape == (self.config.n, 1)
        assert u_k_i.shape == (self.config.m, 1)
        assert i_onehot.shape == (self.config.N, 1)
        del state_i_k
        lf = 1.0
        lr = 1.0
        beta = cas.atan(cas.tan(u_k_i[1]) * lr / (lf + lr))
        dx = cas.vertcat(
            x_k_i[2] * cas.cos(x_k_i[3] + beta), x_k_i[2] * cas.sin(x_k_i[3] + beta), u_k_i[0],
            x_k_i[2] / lr * cas.sin(beta)
        )
        dt = self.config.get_param('dt')
        val = x_k_i + dx * dt
        return val

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
            # Collision constraint collison_h(xi, xj) N*T
            for k in range(1, self.T+1):
                # collision residual for h > 0
                # x[k] -> x_{k+1} due to index alignment
                xk = cas.reshape(x[:, k-1], self.n, self.N)
                h_vals = [self.collision_h(xk[:, i], xk[:, j]) for j in range(self.N)]
                # ignore self-collision, but keep this dummy constraint to simplify index counting
                h_vals[i] = -1
                h_vals = cas.vertcat(*h_vals)
                assert h_vals.shape == (self.N, 1)
                hi_vec.append(h_vals)  # N, agent i vs everyone (N)
            # Additional constraints for agent i, None here
            h_vec.append(cas.vertcat(*hi_vec))  # N*T

        h_vec = cas.horzcat(*h_vec)
        assert h_vec.shape == (self.n_hi, self.N)
        return h_vec

    def collision_h(self, x_i, x_j):
        """ Collision constraint for x_i, anx x_j agent, h <= 0
        Args:
            x_i: (n,1) State of i at time k
            x_j: (n,1) State of j at time k
        Return:
            h_val: (1,1), h_val <= 0 means no collision

        """
        # car distance larger than 1.2 normalized
        assert x_i.shape == (self.n, 1)
        assert x_j.shape == (self.n, 1)
        collision_radius = self.config.get_param('collision_radius')
        val = -((x_i[0, 0] - x_j[0, 0]) / 1.0)**2 - (
            x_i[1, 0] - x_j[1, 0])**2 + collision_radius**2
        return val


def create_random_game(car_count=3, horizon=20):
    """ Create a CarMergeKinematicBicycle instance with random initial states"""
    default = CarMergeKinematicBicycleCasadiConfig
    T = horizon
    N: int = car_count
    n: int = 4
    m: int = 2

    J_Qr = np.diag([0, 0.1, 0.01, 1.0])
    J_R = np.eye(m) * 0.3

    # multiple car merge, car_count: main_lane_n + merge_lane_n
    main_lane_n = min(int(0.5 * N), N - 1)
    merge_lane_n = N - main_lane_n
    x_pos_main_lane = (
        np.linspace(0, (main_lane_n - 1) * 5.4, main_lane_n)
        + np.random.random(main_lane_n)
    )
    x_pos_merge_lane = (
        2.7
        + (np.random.random() - 0.5) * 2 * 2.5  # overall offset
        + np.linspace(0, (merge_lane_n - 1) * 5.4, merge_lane_n)  # spacing
        + np.random.random(merge_lane_n)  # individual random offset
    )
    v_main_lane = 2.0 + np.random.random(main_lane_n)
    v_merge_lane = 2.0 + np.random.random(merge_lane_n)
    x0_main_lane = np.vstack([
        x_pos_main_lane, default.track_width / 2 * np.ones(main_lane_n),
        v_main_lane,
        np.zeros(main_lane_n)
    ]).T
    x0_merge_lane = np.vstack([
        x_pos_merge_lane, -default.track_width / 2 * np.ones(merge_lane_n),
        v_merge_lane,
        np.zeros(merge_lane_n)
    ]).T
    x0 = np.vstack([x0_main_lane, x0_merge_lane]).T
    target_y = [1] * (main_lane_n + merge_lane_n)
    x_ref = np.zeros((n, N))
    x_ref[2, :] = 2.0  # target speed
    x_ref[1, :] = np.array(target_y)  # target y position

    config = CarMergeKinematicBicycleCasadiConfig(
        T=T,
        dt=default.dt,
        N=N,
        n=n,
        m=m,
        n_hi=N*T,  # Collision constraint only
        track_width=default.track_width,
        collision_radius=default.collision_radius,
        x0=x0.copy(order='F'),
        target_x_ref=x_ref.copy(order='F'),
        J_Qr=J_Qr.copy(order='F'),
        J_R=J_R.copy(order='F')
    )
    return CarMergeKinematicBicycleCasadi(config)
