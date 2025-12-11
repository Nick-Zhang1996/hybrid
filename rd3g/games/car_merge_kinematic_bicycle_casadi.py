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
from rd3g.core.base_casadi_game import CasadiGameConfig
from rd3g.core.base_jax_game import BaseGame

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
    track_width: float = 2.2
    collision_radius: float = 2.0

    x0: Any = None
    """ Initial state for all agents, dim: (N,n)"""
    target_x_ref: Any = None
    """ Target state for all agents, dim: (N,n)"""
    J_Qr: Any = None
    """ Cost matrix for tracking reference state dim: (n,n)"""
    J_Q: Any = None
    """ Cost matrix for penalizing non-zero state dim: (n,n)"""
    J_R: Any = None
    """ Cost matrix for control effort dim: (m,m)"""

    def serialize(self):
        """ Serialize to int_param, double_param for interfacing with CasADi codegen
        Returns:
            int_param: flattened integer array
            double_param: flattened double array
        """
        int_param = np.array([self.T, self.N, self.n, self.m])
        double_param = np.hstack([[self.dt],  # 0
                                  self.x0.flatten(),  # 1:N*n+1
                                  self.target_x_ref.flatten(),  # N*n+1:N*n+1+N*n
                                  self.J_Qr.flatten(),  # 2*N*n+1:3*N*n+1
                                  self.J_R.flatten()])  # 3*N*n+1: 3*N*n+1 + m*m
        assert len(int_param.shape) == 1
        assert len(double_param.shape) == 1
        return int_param, double_param


class CarMergeKinematicBicycleCasadi(BaseGame):
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

        # bounds for visualization
        self.visual_x_lim = [-2.5, 2.5]
        self.visual_y_lim = [-2, 30]
        # animation/visualization related
        self.sprite_visualization = True  # True would use car images instead of boaxes

        self.int_param, self.double_param = config.serialize()

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

    def visualize(self, x0, u, x=None, show=True, save=False):
        if (not show) and (not save):
            return
        if x is None:
            x = np.vstack(
                [self.config.x0[np.newaxis, :, :],
                 self.rollout(self.config.x0, u)])
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
            xx = x[:, i, 0]
            yy = x[:, i, 1]
            ax.plot(-yy, xx, '*-')
        ax.set_aspect('equal', adjustable='box')
        if save:
            filename = resolve_logname(suffix='gif')
            fig.savefig(filename)
            logger.info(f'saved figure to {filename}')
        if show:
            plt.show()

    def animate(self, x0, u, x=None, show=False, save=False):
        if (not show) and (not save):
            return
        if x is None:
            x = np.vstack(
                [self.config.x0[np.newaxis, :, :],
                 self.rollout(self.config.x0, u)])
        fig, ax = plt.subplots()

        if self.sprite_visualization:
            car_scale = self.car_scale
            # draw car sprite
            car_pose_vec = []
            for states in x:
                car_pose_vec.append(
                    [[-states[i][1], states[i][0], states[i][3] + np.pi / 2]
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
                xx = x[:, i, 0] - 1.0
                yy = -(x[:, i, 1]) - 0.5
                angle = x[:, i, 3]
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

        gif_filename = resolve_logname(suffix='gif')
        if save:
            anim.save(gif_filename, writer='pillow')
            logger.info(f'gif saved to {gif_filename}')
        if show:
            plt.show()
        # NOTE save initial, middle, final snapshots
        # update(0)
        # fig.canvas.draw()
        # frame = Image.frombytes('RGB',
        # fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
        # filename = f'./pics/merge_{self.N}car_initial.png'
        # self.print_info(f'saved to {filename}')
        # frame.save(filename)

        # update(self.T//2)
        # fig.canvas.draw()
        # frame = Image.frombytes('RGB',
        # fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
        # filename = f'./pics/merge_{self.N}car_middle.png'
        # self.print_info(f'saved to {filename}')
        # frame.save(filename)

        # update(self.T-1)
        # fig.canvas.draw()
        # frame = Image.frombytes('RGB',
        # fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
        # filename = f'./pics/merge_{self.N}car_final.png'
        # self.print_info(f'saved to {filename}')
        # frame.save(filename)
        return

    def J(self, x_k, u_k_i, i, int_param, double_param):
        '''
        stage cost for an agent, given x,u
        x_k.shape (N,n) x_k_i
        u_k_i.shape (m) u_k_i
        i: agent id
        int_param: integer parameters
        double_param: double parameters
        '''

        x_k_i = x_k[i, :]
        dx = x_k_i - self.config.target_x_ref[i]

        val = dx.T @ self.config.J_Qr @ dx + \
            x_k_i.T @ self.config.J_Q @ x_k_i + u_k_i.T @ self.config.J_R @ u_k_i
        return val

    def Jfi(self, x_T, i):
        return self.J(x_T, cas.SX.zeros(self.m), i)

    def f(self, x, u, i: int):
        ''' Dynamics function x_{t+1} = f(x_t,u,i)
        Args:
            x: (n,) State for agent i
            u: (m,) Control for agent i
        Return:
            (n,) The next state, progressed by self.dt

        this problem has homogeneous agents, so [i] is irrelevant'''
        lf = 1.0
        lr = 1.0
        beta = cas.atan(cas.tan(u[1]) * lr / (lf + lr))
        dx = cas.vertcat(
            x[2] * cas.cos(x[3] + beta), x[2] * cas.sin(x[3] + beta), u[0],
            x[2] / lr * cas.sin(beta)
        )
        val = x + dx * self.config.dt
        return val

    def h(self, x_i, x_j):
        """ Collision constraint for x_i, anx x_j agent, h <= 0
        Args:
            x_i: (n,) State of i at time k
            x_j: (n,) State of j at time k

        """
        # car distance larger than 1.2 normalized
        val = -((x_i[0] - x_j[0]) / 1.0)**2 - (
            x_i[1] - x_j[1])**2 + self.config.collision_radius**2
        return val


def create_random_game(car_count=3, horizon=20):
    """ Create a CarMergeKinematicBicycle instance with random initial states"""
    default = CarMergeKinematicBicycleCasadiConfig()
    T = horizon
    N: int = car_count
    n: int = 4
    m: int = 2

    J_Qr = np.diag([0, 0.1, 0.01, 0])
    J_Q = np.diag([0, 0, 0, 1.0])
    J_R = np.eye(m) * 0.3

    # multiple car merge, car_count: main_lane_n + merge_lane_n
    main_lane_n = min(int(0.67 * N), N - 1)
    merge_lane_n = N - main_lane_n
    x_pos_main_lane = (
        np.linspace(0, (main_lane_n - 1) * 5.4, main_lane_n)
        + np.random.random(main_lane_n)
    )
    x_pos_merge_lane = (
        (np.random.random() - 0.5) * 2 * 2.5  # overall offset
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
    x0 = np.vstack([x0_main_lane, x0_merge_lane])
    target_y = [1] * (main_lane_n + merge_lane_n)
    x_ref = np.zeros((N, n))
    x_ref[:, 2] = 2.0  # target speed
    x_ref[:, 1] = np.array(target_y)  # target y position

    config = CarMergeKinematicBicycleCasadiConfig(
        T=T,
        dt=default.dt,
        N=N,
        n=n,
        m=m,
        track_width=default.track_width,
        collision_radius=default.collision_radius,
        x0=x0,
        target_x_ref=x_ref,
        J_Qr=J_Qr,
        J_Q=J_Q,
        J_R=J_R
    )
    return CarMergeKinematicBicycleCasadi(config)
