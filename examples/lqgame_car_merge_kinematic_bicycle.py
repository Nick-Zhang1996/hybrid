import numpy as np
from time import time
from math import sin, cos, tan, atan, radians, degrees
from PIL import Image
from scipy import interpolate
import scipy.sparse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

from utilities.util import *
from utilities.time_util import TimeUtil
from LQGame import LQGame

# NOTE: Adjust car count here
car_count = 10


class LQGame_CarMergeKinematicBicycle(LQGame):

    def __init__(self, car_count):
        super().__init__()
        """
        The following notation is used:
        u_i = [throttle, steering]
        x_i = [x, y, v, theta] such that: x:upwards; y:leftwards; v:velocity; theta: ccw (right hand coord)
        Collision constraint: [(xi-xj)/dx]^2 + [(yi-yj)/dy]^2 >= 1
        N: agent count
        T: time step, 1 ... T + 1
        X = concatenated game state, first agent, then by time
            e.g. state p of agent i at time k: X[k, i, p] or X.flatten()[k*N*m + i*m + p]
        U = concatenated control, dim: T * N * m
            e.g. control p of agent i at time k: U[k, i, p] or U.flatten()[k*N*m + i*m + p]
        Q = state cost matrices for each agent and time step. 
            q = state cost vectors ''
        R = control cost matrices ''
            r = control cost vectors ''
            
        x_i_k: 1 ... T, T*N*n NOTE starts from 1
        u_i_k: 0 ... T-1, T*N*m
        Qs = [Q1s, Q2s, ..., QNs] for each of the N agents
            Qis = [Qi_1, Qi_2, ..., Qi_T]
        qs = [q1, q2, ... Q]
            qis = [qi_1, qi_2, ..., qi_T]
        Rs = [R1s, R2s, ..., RNs]
            Ris = [Ri_0, Ri_1, ..., Ri_T-1]
        rs = [r1s, r2s, ..., rNs]
            ris = [ri_0, ri_1, ..., ri_T-1]
        """

        # Decision variables:
        self.N = car_count
        self.T = 20
        self.track_width = 2.2
        self.track_length = 20
        self.dt = dt = 0.2
        self.main_lane_n = min(int(0.65 * car_count), car_count - 1)

        # Dimension of x and u for single agent
        self.n = 4
        self.m = 2

        # Bounds for visualization
        self.visual_x_lim = [-2.5, 2.5]
        self.visual_y_lim = [-2, 30]

        # Collision definition
        self.h_Qh = np.diag([-0.1, -0.1, 0, 0]) * 10

        # Step cost parameters
        #self.J_Qr = np.diag([0, 0.070, 0.01, 0])
        #self.J_Q = np.diag([0,0,0,1.0])
        #self.J_R_merge = np.diag([0.2, 1.5])
        #self.J_R_main = np.diag([1.0, 10.0])
        #self.J_R = np.diag([2.0, 20.0])

        #self.J_Qr = np.diag([0,0.1,0.01,0])
        self.J_Qr = np.diag([0, 0.3, 0.01, 0])
        self.J_Q = np.diag([0, 0, 0, 1.0])
        self.J_R = np.eye(self.m) * 0.3
        self.J_x_ref_fun = lambda i: np.array([0, self.target_y[i], 2.0, 0])

        # multiple car merge, car_count: main_lane_n + merge_lane_n, Dr 650ms
        #np.random.seed(0)
        main_lane_n = min(int(0.65 * car_count), car_count - 1)
        merge_lane_n = car_count - main_lane_n
        x_pos_main_lane = np.linspace(
            0, (main_lane_n - 1) * 5.5,
            main_lane_n) + np.random.random(main_lane_n)
        x_pos_merge_lane = 2.5 + np.linspace(
            0, (merge_lane_n - 1) * 5.5,
            merge_lane_n) + np.random.random(merge_lane_n)
        # NOTE: Starting position: below is greater spacing, can be adjusted. Recommended to keep 2:1 scale
        # x_pos_main_lane = np.linspace(0,(main_lane_n-1)*8,main_lane_n) + np.random.random(main_lane_n)
        # x_pos_merge_lane = 4.0+np.linspace(0,(merge_lane_n-1)*8,merge_lane_n) + np.random.random(merge_lane_n)
        v_main_lane = 2.0 + np.random.random(main_lane_n)
        v_merge_lane = 2.0 + np.random.random(merge_lane_n)
        x0_main_lane = np.vstack([
            x_pos_main_lane, self.track_width / 2 * np.ones(main_lane_n),
            v_main_lane,
            np.zeros(main_lane_n)
        ]).T
        x0_merge_lane = np.vstack([
            x_pos_merge_lane, -self.track_width / 2 * np.ones(merge_lane_n),
            v_merge_lane,
            np.zeros(merge_lane_n)
        ]).T
        self.x0 = np.vstack([x0_main_lane, x0_merge_lane])
        self.target_y = [1] * (main_lane_n + merge_lane_n)
        self.print_debug_enable()

    def setup(self):
        '''
        If implementing cpp, this method will load cpp and eigen module and set x0.
        At the moment, no cpp integration.
        '''
        #raise NotImplementedError
        return

    def _visualize(self, u, x=None):
        """Plots the track and trajectories of the cars and then shows the visualization.

        Args:
            U: Controls of all agents over horizon, [T * N * m]
            X: Game state of all agents over horizon, [T * N * n]. Defaults to None.

        Returns:
            plotted visualization of the simulated road
        """
        if (x is None):
            x = np.vstack(
                [self.x0[np.newaxis, :, :],
                 self.rollout(self.x0, u)])
        fig, ax = plt.subplots()
        ax.vlines(x=-self.track_width,
                  ymin=self.visual_y_lim[0],
                  ymax=self.visual_y_lim[1])
        ax.vlines(x=self.track_width,
                  ymin=self.visual_y_lim[0],
                  ymax=self.visual_y_lim[1])
        # dotted line
        for i in np.linspace(self.visual_y_lim[0], self.visual_y_lim[1], 10):
            ax.vlines(x=0, ymin=i, ymax=i + 0.5)

        for i in range(self.N):
            xx = x[:, i, 0]
            yy = x[:, i, 1]
            plt.plot(-yy, xx, '*-')
        ax.set_aspect('equal', adjustable='box')
        return fig

    def _animation(self, U, X=None, gif_prefix=''):
        """Animates the progression of the game and then saves as a gif.

        Args:
            U: Controls of all agents over horizon, [T * N * m]
            X: Game state of all agents over horizon, [T * N * n]. Defaults to None.
            gif_prefix: Prefix of file generated. Defaults to ''.

        Returns:
            List of cars represented as rectangles
        """
        if X is None:
            X = np.vstack(
                [self.x0[np.newaxis, :, :],
                 self.rollout(self.x0, U)])
        car_pos_vec = []
        car_angle_vec = []
        box_vec = []
        circle_vec = []
        color_vec = ['red', 'green', 'blue', 'black']
        color_vec = [color_vec[i % len(color_vec)] for i in range(self.N)]
        # prepare smoothed animation
        for i, color in zip(range(self.N), color_vec):
            # interpolate for smooth graphics
            #tt = np.linspace(0,self.T*self.dt,50)
            tt = np.linspace(0, self.dt * self.T, self.T + 1)
            # for plt.Rectangle, we offset position so this corresponds to top left corner
            # also flip x axis
            xx = X[:, i, 0] - 1.0
            yy = -(X[:, i, 1]) - 0.5
            angle = X[:, i, 3]
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
                           radius=(7**0.5) / 2,
                           color=color,
                           fill=False))

        fig, ax = plt.subplots()
        ax.set_xlim(*self.visual_x_lim)
        ax.set_ylim(*self.visual_y_lim)

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
        # lane boundary lines
        ax.vlines(x=-self.track_width,
                  ymin=self.visual_y_lim[0],
                  ymax=self.visual_y_lim[1])
        ax.vlines(x=self.track_width,
                  ymin=self.visual_y_lim[0],
                  ymax=self.visual_y_lim[1])
        # dotted line
        for i in np.linspace(self.visual_y_lim[0], self.visual_y_lim[1], 20):
            ax.vlines(x=0, ymin=i, ymax=i + 1)

        ax.set_aspect('equal', adjustable='box')

        # Create the animation
        anim = FuncAnimation(fig,
                             update,
                             frames=len(car_pos_vec[0]),
                             blit=True)
        gif_filename = self.resolveLogname(logPrefix=gif_prefix)
        anim.save(gif_filename, writer='pillow')
        plt.show()

    def f(self, x, u, i):
        """Advances the dynamics by one time step according to the kinematic bicycle model.

        Args:
            x: Car state at time step, [n * 1]
            u: Car controls at time step, [m * 1]
            i: Agent number

        Returns:
            Dynamics of next time step
        """
        lf = 1.0
        lr = 1.0
        beta = atan(tan(u[1]) * lr / (lf + lr))
        dx = np.array([
            x[2] * cos(x[3] + beta), x[2] * sin(x[3] + beta), u[0],
            x[2] / lr * sin(beta)
        ])
        return x + dx * self.dt

    def df_dx(self, x, u, i):
        """Jacobian of dynamics with respect to x.

        Args:
            x: Car state at time step, [n * 1]
            u: Car controls at time step, [m * 1]
            i: Agent number

        Returns:
            First time derivative of dynamics with respect to x
        """
        beta = atan(tan(u[1]) * 0.5)
        A = np.array([[0, 0, cos(x[3] + beta), -x[2] * sin(x[3] + beta)],
                      [0, 0, sin(x[3] + beta), x[2] * cos(x[3] + beta)],
                      [0, 0, 0, 0], [0, 0, sin(beta) / 1.0, 0]])
        val = np.eye(4) + A * self.dt
        return val

    def df_du(self, x, u, i):
        """Jacobian of dynamics with respect to u.

        Args:
            x: Car state at time step, [n * 1]
            u: Car controls at time step, [m * 1]
            i: Agent number

        Returns:
            First time derivative of dynamics with respect to u
        """
        beta = atan(tan(u[1]) * 0.5)
        dbeta_dst = 0.5 / (((tan(u[1]) * 0.5)**2 + 1) * cos(u[1])**2)
        B = np.array([[0, -x[2] * sin(x[3] + beta) * dbeta_dst],
                      [0, x[2] * cos(x[3] + beta) * dbeta_dst], [1, 0],
                      [0, x[2] / 1.0 * cos(beta) * dbeta_dst]])
        val = B * self.dt
        return val

    def J(self, x_k, u_k_i, i):
        '''
        step cost for an agent, given x,u
        x_k.shape (N*n) x_k_i = [x,y,vx,vy]
        u_k_i.shape (m) u_k_i = [ax, ay]
        i: agent id
        '''
        #return (x[2] - 2.0)**2 + (x[1] - self.target_y[i])**2 + 1e-2*x[3]**2 + 1e-2*u.T @ np.eye(self.m) @ u
        if (self.config.USE_CPP):
            return self.cpp.J(x_k, u_k_i, i)
        val = (x_k[i] - self.J_x_ref_fun(i)).T @ self.J_Qr @ (
            x_k[i] - self.J_x_ref_fun(i)
        ) + x_k[i].T @ self.J_Q @ x_k[i] + u_k_i.T @ self.J_R @ u_k_i
        if (self.config.CPP_DEBUG):
            alt = self.cpp.J(x_k, u_k_i, i)
            if (np.linalg.norm(alt - val) > 1e-4):
                breakpoint()
        return val

    # dJi dxi
    def dJi_dxi(self, x_k, u_k_i, i):
        if (self.config.USE_CPP):
            return self.cpp.dJi_dxi(x_k, u_k_i, i)
        val = 2 * (x_k[i] -
                   self.J_x_ref_fun(i)).T @ self.J_Qr + 2 * x_k[i].T @ self.J_Q
        # if (i == 1):
        #     print('x_k[i]: ', x_k[i])
        #     print('y diff: ', x_k[i][1] - self.J_x_ref_fun(i)[1])
        #     print('val: ', val)
        if (self.config.CPP_DEBUG):
            alt = self.cpp.dJi_dxi(x_k, u_k_i, i)
            if (np.linalg.norm(alt - val) > 1e-4):
                breakpoint()
        return val

    # dJi dxj
    def dJi_dxj(self, x_k, u_k_i, i, j):
        if (self.config.USE_CPP):
            return self.cpp.dJi_dxj(x_k, u_k_i, i, j)
        val = 0
        if (self.config.CPP_DEBUG):
            alt = self.cpp.dJi_dxj(x_k, u_k_i, i, j)
            if (np.linalg.norm(alt - val) > 1e-4):
                breakpoint()
        return val

    def dJi_du(self, x_k, u_k_i, i):
        if (self.config.USE_CPP):
            return self.cpp.dJi_du(x_k, u_k_i, i)
        val = 2 * u_k_i.T @ self.J_R
        if (self.config.CPP_DEBUG):
            alt = self.cpp.dJi_du(x_k, u_k_i, i)
            if (np.linalg.norm(alt - val) > 1e-4):
                breakpoint()
        return val

    # dJ^i / dxi dxi
    def dJi_dxi_dxi(self, x_k, u, i):
        if (self.config.USE_CPP):
            return self.cpp.dJi_dxi_dxi(x_k, u, i)
        val = 2 * self.J_Qr + 2 * self.J_Q
        if (self.config.CPP_DEBUG):
            alt = self.cpp.dJi_dxi_dxi(x_k, u, i)
            if (np.linalg.norm(alt - val) > 1e-4):
                breakpoint()
        return val

    # dJi / dxi dxj
    def dJi_dxi_dxj(self, x_k, u_k_i, i, j):
        return 0

    def dJi_dxj_dxj(self, x_k, u_k_i, i, j):
        return 0

    def dJi_dudu(self, x_k, u_k_i, i):
        return 2 * self.J_R


if __name__ == "__main__":
    main = LQGame_CarMergeKinematicBicycle(car_count)
    main.solve(save_gif=False, visualize=True, animate=True)
    main.final()
