import os
import logging
from math import sin, cos, tan, atan, radians, degrees
from dataclasses import dataclass
from typing import Any

import jax.lax
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np
from scipy import interpolate
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.animation import FuncAnimation

from rd3g.utilities.util import cpp_capable, BASEDIR, resolve_logname
# pylint: disable-next=no-name-in-module
from rd3g.src.build.car_merge_kinematic_bicycle import CarMergeKinematicBicycle as cpp_CarMergeKinematicBicycle
from rd3g.core.base_game import BaseGame, BaseGameConfig

logger = logging.getLogger('CarMergeKinematicBicycle')
logger.setLevel(logging.INFO)


@dataclass(frozen=True)
class CarMergeKinematicBicycleConfig(BaseGameConfig):
    """ Base Class for game configuration"""
    T: int = 0
    dt: float = 0.2
    N: int = 3
    n: int = 4
    m: int = 2
    x0: Any = None
    track_width: float = 2.2
    collision_radius: float = 2.0
    target_y: Any = None
    """ Target lateral position, i.e. lane """
    J_Qr: Any = None
    """ Cost matrix for tracking reference state """
    J_Q: Any = None
    """ Cost matrix for penalizing non-zero state """
    J_R: Any = None
    """ Cost matrix for control effort """
    USE_CPP: bool = True


class CarMergeKinematicBicycle(BaseGame):
    ''' Kinematic Bicycle Merging Game
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

    def __init__(self, config: CarMergeKinematicBicycleConfig):
        super().__init__(config)

        # bounds for visualization
        self.visual_x_lim = [-2.5, 2.5]
        self.visual_y_lim = [-2, 30]
        # animation/visualization related
        self.sprite_visualization = True  # True would use car images instead of boaxes

        if (self.sprite_visualization):
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

    def setup_rd3g_cpp(self, solver_config):
        cpp = cpp_CarMergeKinematicBicycle(
            self.config.N,
            self.config.T,
            self.config.dt,
            solver_config.rho_0,
            solver_config.rho_b,
            solver_config.bc_a,
            solver_config.bc_b,
            solver_config.tolerance,
            solver_config.backtracking_max_iter,
            self.config.J_Qr,
            self.config.J_Q,
            self.config.J_R,
            self.h_Qh,
            self.config.target_y,
            self.config.collision_radius,
            solver_config.iterations,
            False)
        cpp.set_x0(self.config.x0)
        self.cpp = cpp
        return cpp

    def visualize(self, x0, u, x=None, show=True, save=False):
        if (not show) and (not save):
            return
        if (x is None):
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

        if (self.sprite_visualization):
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

    def J_x_ref_fun(self, i):
        return np.array([0, self.config.target_y[i], 2.0, 0])

    # --------  math functions and their derivatives ------
    @cpp_capable
    def J(self, x_k, u_k_i, i):
        '''
        step cost for an agent, given x,u
        x_k.shape (N,n) x_k_i
        u_k_i.shape (m) u_k_i
        i: agent id
        '''
        # return (x[2] - 2.0)**2 + (x[1] - self.target_y[i])**2 + 1e-2*x[3]**2 + 1e-2*u.T @ np.eye(self.m) @ u
        val = (x_k[i] - self.J_x_ref_fun(i)).T @ self.config.J_Qr @ (
            x_k[i] - self.J_x_ref_fun(i)
        ) + x_k[i].T @ self.config.J_Q @ x_k[i] + u_k_i.t @ self.config.J_R @ u_k_i
        return val

    def _jax_j(self, x_k, u_k_i, i):
        '''
        step cost for an agent, given x,u
        x_k.shape (N,n) x_k_i
        u_k_i.shape (m) u_k_i
        i: agent id
        '''
        # val = (x_k[i] - self.jax_J_x_ref_fun(i)).T @ self.J_Qr @ (
        #     x_k[i] - self.jax_J_x_ref_fun(i)
        # ) + x_k[i].T @ self.J_Q @ x_k[i] + u_k_i.T @ self.J_R @ u_k_i

        x_k_i = jax.lax.dynamic_index_in_dim(x_k, i, keepdims=False)
        dx = x_k_i - jax.lax.dynamic_index_in_dim(self.jax_target_x_ref, i, keepdims=False)

        val = dx.T @ self.jax_J_Qr @ dx + \
            x_k_i.T @ self.jax_J_Q @ x_k_i + u_k_i.T @ self.jax_J_R @ u_k_i
        return val

    @cpp_capable
    def dJi_dxi(self, x_k, u_k_i, i):
        val = 2 * (x_k[i] -
                   self.J_x_ref_fun(i)).T @ self.config.J_Qr + 2 * x_k[i].T @ self.config.J_Q
        val = val.reshape(1, self.n)
        return val

    @cpp_capable
    def dJi_dxj(self, x_k, u_k_i, i, j):
        val = jnp.zeros((1, self.config.n))
        return val

    @cpp_capable
    def dJi_du(self, x_k, u_k_i, i):
        val = 2 * u_k_i.T @ self.config.J_R
        val = val.reshape(1, 2)
        return val

    # dJ^i / dxi dxi
    @cpp_capable
    def dJi_dxi_dxi(self, x_k, u_k_i, i):
        val = 2 * self.config.J_Qr + 2 * self.config.J_Q
        return val

    @cpp_capable
    def dJi_dxi_dxj(self, x_k, u_k_i, i, j):
        # dJi / dxi dxj
        return 0

    @cpp_capable
    def dJi_dxj_dxj(self, x_k, u_k_i, i, j):
        return 0

    @cpp_capable
    def dJi_dudu(self, x_k, u_k_i, i):
        return 2 * self.config.J_R

    @cpp_capable
    def f(self, x: ArrayLike, u: ArrayLike, i: int):
        ''' Dynamics function x_{t+1} = f(x_t,u,i)
        Args:
            x: (n,) State for agent i
            u: (m,) Control for agent i
        Return:
            (n,) The next state, progressed by self.dt

        this problem has homogeneous agents, so [i] is irrelevant'''
        lf = 1.0
        lr = 1.0
        beta = np.arctan(np.tan(u[1]) * lr / (lf + lr))
        dx = np.array([
            x[2] * np.cos(x[3] + beta), x[2] * np.sin(x[3] + beta), u[0],
            x[2] / lr * np.sin(beta)
        ])
        val = x + dx * self.config.dt
        # NOTE the cpp version return has dimension (n,1), while this is (n,)
        return val

    def jax_f(self, x: ArrayLike, u: ArrayLike, i: int):
        ''' Dynamics function x_{t+1} = f(x_t,u,i)
        Args:
            x: (n,) State for agent i
            u: (m,) Control for agent i
        Return:
            (n,) The next state, progressed by self.dt

        this problem has homogeneous agents, so [i] is irrelevant'''
        lf = 1.0
        lr = 1.0
        beta = jnp.atan(jnp.tan(u[1]) * lr / (lf + lr))
        dx = jnp.array([
            x[2] * jnp.cos(x[3] + beta), x[2] * jnp.sin(x[3] + beta), u[0],
            x[2] / lr * jnp.sin(beta)
        ])
        val = x + dx * self.config.dt
        # NOTE the cpp version return has dimension (n,1), while this is (n,)
        return val

    @cpp_capable
    def df_dx(self, x, u, i):
        beta = atan(tan(u[1]) * 0.5)
        A = jnp.array([[0, 0, cos(x[3] + beta), -x[2] * sin(x[3] + beta)],
                      [0, 0, sin(x[3] + beta), x[2] * cos(x[3] + beta)],
                      [0, 0, 0, 0], [0, 0, sin(beta) / 1.0, 0]])
        val = jnp.eye(4) + A * self.config.dt
        return val

    @cpp_capable
    def df_du(self, x, u, i):
        beta = atan(tan(u[1]) * 0.5)
        dbeta_dst = 0.5 / (((tan(u[1]) * 0.5)**2 + 1) * cos(u[1])**2)
        B = jnp.array([[0, -x[2] * sin(x[3] + beta) * dbeta_dst],
                      [0, x[2] * cos(x[3] + beta) * dbeta_dst], [1, 0],
                      [0, x[2] / 1.0 * cos(beta) * dbeta_dst]])
        val = B * self.config.dt
        return val

    # collision definition is similar to Double Integrator, car is an "ellipsis"
    @cpp_capable
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

    @cpp_capable
    def dh_dxi(self, x_i, x_j):
        val = 2 * (x_i - x_j).T @ self.h_Qh
        val = val.reshape(1, self.config.n)
        return val

    @cpp_capable
    def dh_dxj(self, x_i, x_j):
        val = 2 * (x_j - x_i).T @ self.h_Qh
        val = val.reshape(1, self.config.n)
        return val

    @cpp_capable
    def dh_dxi_dxi(self, x_i, x_j):
        val = 2 * self.h_Qh.T
        return val

    @cpp_capable
    def dh_dxj_dxi(self, x_i, x_j):
        val = -2 * self.h_Qh.T
        return val

    @cpp_capable
    def dh_dxi_dxj(self, x_i, x_j):
        val = -2 * self.h_Qh.T
        return val

    @cpp_capable
    def dh_dxj_dxj(self, x_i, x_j):
        val = 2 * self.h_Qh.T
        return val


def create_random_game(car_count=3, horizon=20):
    """ Create a CarMergeKinematicBicycle instance with random initial states"""
    default = CarMergeKinematicBicycleConfig()
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

    config = CarMergeKinematicBicycleConfig(
        T=T,
        dt=default.dt,
        N=N,
        n=n,
        m=m,
        x0=x0,
        track_width=default.track_width,
        collision_radius=default.collision_radius,
        target_y=target_y,
        J_Qr=J_Qr,
        J_Q=J_Q,
        J_R=J_R
    )
    return CarMergeKinematicBicycle(config)
