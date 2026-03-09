""" Uncoordinated intersection, CasADi version """
import os
import logging
from typing import Any
from math import degrees
from dataclasses import dataclass

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
# TODO: add lane/boundary constraint to h()


@dataclass(frozen=True)
class IntersectionCasadiConfig(CasadiGameConfig):
    """ Base Class for game configuration. 
    Bottom Left of the intersection is the origin. X right, Y up"""
    T: int = 0
    dt: float = 0.2
    N: int = 3
    n: int = 4
    m: int = 2
    n_hi: int = 0  # Total number of constraints for EACH agent, e.g. pairwise collision only: N*T

    lane_width: float = 2.2  # Each lane width
    hori_lanes: int = 2  # Number of horizontal lanes
    vert_lanes: int = 3  # Number of vertical lanes

    collision_radius: float = 2.0  # 1.4
    """ Minimum distance between two cars"""

    x0: Any = None
    """ Initial state for all agents, dim: (n,N)"""
    target_x_ref: Any = None
    """ Target state for all agents, dim: (n,N)"""
    J_Qr_diag_vec: Any = None
    """ Diagonal terms for cost matrices for tracking reference state dim: (n,N)"""
    J_R: Any = None
    """ Cost matrix for control effort dim: (m,m)"""

    def __post_init__(self):
        assert self.x0.shape == (self.n, self.N), (
            'Incorrect self.x0 dimension, '
            f'should be {(self.n, self.N)}, but got {self.x0.shape}')
        assert self.target_x_ref.shape == (self.n, self.N)
        assert self.J_Qr_diag_vec.shape == (self.n, self.N)
        assert self.J_R.shape == (self.m, self.m)
        return super().__post_init__()


class IntersectionCasadi(CasadiGame):
    ''' Kinematic Bicycle Intersection Game, with CasADi
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

    def __init__(self, config: IntersectionCasadiConfig):
        super().__init__(config)

        # n_hi is a new concept
        self.n_hi = config.n_hi
        # bounds for visualization
        self.visual_x_lim = [-10, 10]
        self.visual_y_lim = [-10, 10]

        self.car_scale = 0.0045 / 2
        color_names = [
            'purple', 'yellow', 'red', 'green', 'orange', 'pink', 'cyan',
            'hot_pink'
        ]
        self.car_img_vec = [
            mpimg.imread(
                os.path.join(BASEDIR, 'rd3g', 'resources', f'porsche_{color}.png'))
            for color in color_names
        ]

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

        # Draw lane markings
        # Solid lines:
        c = self.config
        lw = c.lane_width
        tr = (c.vert_lanes*lw, c.hori_lanes*lw)
        # For states: x:right, y:up
        ax.vlines(x=0, ymin=self.visual_y_lim[0], ymax=0)
        ax.vlines(x=0, ymin=tr[1], ymax=self.visual_y_lim[1])
        ax.vlines(x=tr[0], ymin=self.visual_y_lim[0], ymax=0)
        ax.vlines(x=tr[0], ymin=tr[1], ymax=self.visual_y_lim[1])

        ax.hlines(y=0, xmin=self.visual_x_lim[0], xmax=0)
        ax.hlines(y=0, xmin=tr[0], xmax=self.visual_x_lim[1])
        ax.hlines(y=tr[1], xmin=self.visual_x_lim[0], xmax=0)
        ax.hlines(y=tr[1], xmin=tr[0], xmax=self.visual_x_lim[1])
        # Dashed lines
        x_count = (self.visual_x_lim[1] - self.visual_x_lim[0])//1
        for i in np.linspace(self.visual_x_lim[0], self.visual_x_lim[1], x_count):
            for l in range(1, c.hori_lanes):
                ax.hlines(y=l*lw, xmin=i, xmax=i + 0.5)
        y_count = (self.visual_y_lim[1] - self.visual_y_lim[0])//1
        for i in np.linspace(self.visual_y_lim[0], self.visual_y_lim[1], y_count):
            for l in range(1, c.vert_lanes):
                ax.vlines(x=l*lw, ymin=i, ymax=i + 0.5)

        for i in range(self.N):
            xx = x[0, i, :]
            yy = x[1, i, :]
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
        n = self.n
        m = self.m
        T = self.T
        N = self.N
        assert u.shape == (m, N, T)
        assert x.shape == (n, N, T+1)
        fig, ax = plt.subplots()

        car_scale = self.car_scale
        # draw car sprite
        car_pose_vec = []
        for k in range(T+1):
            states = x[:, :, k]
            car_pose_vec.append(
                [[states[0][i], states[1][i], states[3][i]]
                    for i in range(self.config.N)])

        im_vec = []
        for i in range(self.config.N):
            rotated_car_img = np.clip(
                rotate(self.car_img_vec[i % len(self.car_img_vec)],
                       degrees(car_pose_vec[0][i][2]), reshape=True), 0.0, 1.0)
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

        # fine-tune dark background to mimic tarmac
        # Set the background color of the plot (axes background)
        ax.set_facecolor((54 / 255, 69 / 255, 79 / 255))

        # Draw lane markings
        # Solid lines:
        c = self.config
        lw = c.lane_width
        tr = (c.vert_lanes*lw, c.hori_lanes*lw)
        # For states: x:right, y:up
        ax.vlines(x=0, ymin=self.visual_y_lim[0], ymax=0)
        ax.vlines(x=0, ymin=tr[1], ymax=self.visual_y_lim[1], colors='white')
        ax.vlines(x=tr[0], ymin=self.visual_y_lim[0], ymax=0, colors='white')
        ax.vlines(x=tr[0], ymin=tr[1], ymax=self.visual_y_lim[1], colors='white')

        ax.hlines(y=0, xmin=self.visual_x_lim[0], xmax=0, colors='white')
        ax.hlines(y=0, xmin=tr[0], xmax=self.visual_x_lim[1], colors='white')
        ax.hlines(y=tr[1], xmin=self.visual_x_lim[0], xmax=0, colors='white')
        ax.hlines(y=tr[1], xmin=tr[0], xmax=self.visual_x_lim[1], colors='white')
        # Dashed lines
        x_count = (self.visual_x_lim[1] - self.visual_x_lim[0])//1
        for i in np.linspace(self.visual_x_lim[0], self.visual_x_lim[1], x_count):
            for l in range(1, c.hori_lanes):
                ax.hlines(y=l*lw, xmin=i, xmax=i + 0.5, colors='white')
        y_count = (self.visual_y_lim[1] - self.visual_y_lim[0])//1
        for i in np.linspace(self.visual_y_lim[0], self.visual_y_lim[1], y_count):
            for l in range(1, c.vert_lanes):
                ax.vlines(x=l*lw, ymin=i, ymax=i + 0.5, colors='white')

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(*self.visual_x_lim)
        ax.set_ylim(*self.visual_y_lim)

        # Create the animation
        anim = FuncAnimation(fig, update, frames=self.T, blit=False)
        folder = os.path.join(BASEDIR, 'gifs')
        gif_filename = os.path.join(folder, f'intersection_{self.N}car.gif')
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
            filename = os.path.join(folder, f'intersection_{self.N}car_initial.png')
            frame.save(filename)
            logger.info(f'saved snapshots to {filename}')

            update(self.T//2)
            fig.canvas.draw()
            frame = Image.frombytes('RGB',
                                    fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
            filename = os.path.join(folder, f'intersection_{self.N}car_middle.png')
            frame.save(filename)
            logger.info(f'saved snapshots to {filename}')

            update(self.T-1)
            fig.canvas.draw()
            frame = Image.frombytes('RGB',
                                    fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
            filename = os.path.join(folder, f'intersection_{self.N}car_final.png')
            frame.save(filename)
            logger.info(f'saved snapshots to {filename}')
        return

    def F(self, x_k, u_k):
        """ Dynamics for all agents
        Args:
            x_k: (n,N)
            u_k: (m,N)
        Return:
            x_k_next: (n,N)
        """
        x_k_next_vec = []
        for i in range(self.N):
            i_onehot = cas.SX.eye(self.N)[:, i]
            x_k_next_vec.append(self.f(x_k[:, i], u_k[:, i], i_onehot))
        retval = cas.horzcat(*x_k_next_vec)
        assert retval.shape == (self.n, self.N)
        return retval

    def rollout(self, x0, u):
        """ Rollout control to get state trajectory, casadi compatible
        Args:
            x0: (n,N)
            u: (m*N, T), u0..u_T-1
        Return:
            X: (n*N, T) x1..xT
        """
        assert u.shape == (self.m*self.N, self.T)
        assert x0.shape == (self.n, self.N)

        x_k = cas.SX.sym('x_k_', (self.n*self.N))
        u_k = cas.SX.sym('u_k_', (self.m*self.N))
        x_k_next = cas.vec(self.F(cas.reshape(x_k, self.n, self.N),
                                  cas.reshape(u_k, self.m, self.N)))
        config_params = [self.config.get_int_param_sx(), self.config.get_double_param_sx()]
        config_param_repmat = [cas.repmat(param, 1, self.T) for param in config_params]
        accum_fun = cas.Function('accum_fun', [x_k, u_k]+config_params, [x_k_next, 0])
        rollout_fun = accum_fun.mapaccum(self.T)

        X, _ = rollout_fun(cas.vec(x0), u, *config_param_repmat)
        assert X.shape == (self.n*self.N, self.T)
        return X

    # pylint: disable-next=arguments-renamed
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

        target_x_ref = self.config.get_param('target_x_ref')  # n,N
        J_Qr_diag_vec = self.config.get_param('J_Qr_diag_vec')  # n,N
        J_R = self.config.get_param('J_R')
        x_k_i = x_k @ i_onehot  # dim: n,1
        dx = x_k_i - target_x_ref @ i_onehot
        J_Qr_diag = J_Qr_diag_vec @ i_onehot

        val = cas.sum(dx**2 * J_Qr_diag) + u_k_i.T @ J_R @ u_k_i
        return val

    # pylint: disable-next=arguments-renamed
    def Jfi(self, x_T, i_onehot):
        """ Final cost"""
        return self.J(x_T, cas.SX.zeros(self.m), i_onehot)

    # pylint: disable-next=arguments-renamed
    def f(self, x_k_i, u_k_i, i_onehot):
        """ Dynamics function x_{t+1} = f(x_t,u,i)
        Args:
            x_k_i: (n,1) State for agent i [x,y,v,theta]
            u_k_i: (m,1) Control for agent i
            i_onehot: agent id, in one-hot encoding (N), i.e. i=1,N=4 -> [0,1,0,0], column vector
        Return:
            (n,1) The next state, progressed by self.dt

        this problem has homogeneous agents, so [i] is irrelevant"""
        assert x_k_i.shape == (self.config.n, 1)
        assert u_k_i.shape == (self.config.m, 1)
        assert i_onehot.shape == (self.config.N, 1)
        lf = 1.0
        lr = 1.0

        def dynamics(x_, u_):
            beta = cas.atan(cas.tan(u_[1]) * lr / (lf + lr))
            dx = cas.vertcat(
                x_[2] * cas.cos(x_[3] + beta), x_[2] * cas.sin(x_[3] + beta), u_[0],
                x_[2] / lr * cas.sin(beta)
            )
            return dx
        dt = self.config.get_param('dt')

        # RK4 Integration
        k1 = dynamics(x_k_i, u_k_i)
        # k2 = dynamics(x_k_i + 0.5 * dt * k1, u_k_i)
        # k3 = dynamics(x_k_i + 0.5 * dt * k2, u_k_i)
        # k4 = dynamics(x_k_i + dt * k3, u_k_i)
        # val = x_k_i + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # Euler
        val = x_k_i + k1 * dt
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
        For collision checking, each car is modeled as two circles.
        """
        # car distance larger than 1.2 normalized
        assert x_i.shape == (self.n, 1)
        assert x_j.shape == (self.n, 1)
        d = self.config.get_param('collision_radius')
        val = -((x_i[0, 0] - x_j[0, 0]) / 1.0)**2 - (
            x_i[1, 0] - x_j[1, 0])**2 + d**2
        # Need to look into this double circle model
        # offset = 0.7
        # f1x = x_i[0, 0] + offset * cas.cos(x_i[3, 0])
        # f1y = x_i[1, 0] + offset * cas.sin(x_i[3, 0])
        # f2x = x_j[0, 0] + offset * cas.cos(x_j[3, 0])
        # f2y = x_j[1, 0] + offset * cas.sin(x_j[3, 0])
        # r1x = x_i[0, 0] - offset * cas.cos(x_i[3, 0])
        # r1y = x_i[1, 0] - offset * cas.sin(x_i[3, 0])
        # r2x = x_j[0, 0] - offset * cas.cos(x_j[3, 0])
        # r2y = x_j[1, 0] - offset * cas.sin(x_j[3, 0])

        # FF = -(f1x - f2x)**2 - (f1y - f2y)**2 + d**2
        # FR = -(f1x - r2x)**2 - (f1y - r2y)**2 + d**2
        # RF = -(r1x - f2x)**2 - (r1y - f2y)**2 + d**2
        # RR = -(r1x - r2x)**2 - (r1y - r2y)**2 + d**2
        # vals = cas.vertcat(FF, FR, RF, RR)
        # alpha = 10.0  # "sharpness" of the softmax
        # val = (1.0 / alpha) * cas.log(cas.sum1(cas.exp(alpha * vals)))
        return val


def create_random_game(car_count=3, horizon=20):
    """ Create a CarMergeKinematicBicycle instance with random initial states"""
    default = IntersectionCasadiConfig
    T = horizon
    N: int = car_count
    n: int = 4
    m: int = 2

    J_R = np.eye(m) * 0.1  # Control effort

    hori_lane_n = min(int(0.5 * N), N - 1)
    verti_lane_n = N - hori_lane_n

    def make_x0(hori, lane, offset, v, heading):
        """ Make x0 for a single agent.
        hori: bool, on horizontal or vertical lane
        lane: Index of lane
        offset: distance to intersection origin (bottom left)
        v: current speed
        heading: heading w.r.t. lane direction """
        lw = default.lane_width
        if hori:
            y = (0.5 + lane) * lw
            x = offset
        else:
            x = (0.5 + lane) * lw
            y = offset
            heading = heading + np.pi/2
        x0 = [x, y, v, heading]
        return np.array(x0)

    x0_vec = []
    x_ref_vec = []
    # Ji = x.T @ J_Qr[i] @ x
    J_Qr_diag_vec = []

    for i in range(hori_lane_n):
        offset = - i*4.0 - np.random.random()  # 5.4
        v = 2.0+np.random.random()
        x0 = make_x0(True, np.random.randint(0, default.hori_lanes), offset, v, 0.0)
        x0_vec.append(x0)
        x_ref_vec.append(np.array([0, x0[1], x0[2], x0[3]]))
        J_Qr_diag_vec.append(np.array([0, 0.1, 0.01, 10.0]))

    for i in range(verti_lane_n):
        offset = - i*4.0 - 2.7 - np.random.random()
        v = 2.0+np.random.random()
        x0 = make_x0(False, np.random.randint(0, default.vert_lanes), offset, v, 0.0)
        x0_vec.append(x0)
        x_ref_vec.append(np.array([x0[0], 0, x0[2], x0[3]]))
        J_Qr_diag_vec.append(np.array([0.1, 0, 0.01, 10.0]))

    # n * N
    x0 = np.vstack(x0_vec).T
    # n * N
    x_ref = np.vstack(x_ref_vec).T
    J_Qr_diag_vec = np.vstack(J_Qr_diag_vec).T

    config = IntersectionCasadiConfig(
        T=T,
        dt=default.dt,
        N=N,
        n=n,
        m=m,
        n_hi=N*T,  # Collision constraint only
        lane_width=default.lane_width,
        hori_lanes=default.hori_lanes,
        vert_lanes=default.vert_lanes,
        collision_radius=default.collision_radius,
        x0=x0.copy(order='F'),
        target_x_ref=x_ref.copy(order='F'),
        J_Qr_diag_vec=J_Qr_diag_vec.copy(order='F'),
        J_R=J_R.copy(order='F')
    )
    return IntersectionCasadi(config)
