import numpy as np
from time import time
from math import sin, cos, tan, atan, radians, degrees
from PIL import Image
from scipy import interpolate
import scipy.sparse  # sparse matrix operations
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import matplotlib.image as mpimg
from scipy.ndimage import rotate

from utilities.util import *
#from utilities.TimeUtil import TimeUtil
from src.build.car_merge_kinematic_bicycle import CarMergeKinematicBicycle as cpp_CarMergeKinematicBicycle
from residual_game import ResidualGame, ResidualGameConfig


# Still Rocket Landing, but without step cost on state, only control cost, only terminal cost contains state cost
class RocketLanding(ResidualGame):

    def __init__(self, config: ResidualGameConfig):
        super().__init__(config)

        # agent 0: rocket
        # x: [x,y,theta, vx,vy,omega]
        # u: [Tx,Ty], thrust with respect to body, Tx is nozzle direction, Ty is sideways to right:
        # unit:g
        # agent 1: ship
        # x: [y,vy]
        # u: [ay]: unit:g
        # to make implementation easier, we use n=5 for both agents, we just ignore state 2-4 for agent 1
        # Entries from unused state will be eliminated when we strip Dr of zero row/cols

        # agent count: N, time step: 1..T+1
        # X (game state) = concatenated state, first by agent, then by time)
        # state p of agent i at time k: X[k,i,p] or X.flatten()[k*N*m + i*m + p]
        # U (control) = concatenated control  dim: T*N*m
        # control p of agent i at time k: U[k,i,p] or U.flatten()[k*N*m + i*m + p]
        '''
        x_i_k: 1..T, T*N*n  NOTE starts from 1
        u_i_k: 0..T-1, T*N*m
        lamda_i_k: 0..T-1 T*N*n
        mu_k_i_j: 1..T T*N*N NOTE starts from 1
        '''

        # Problem formulation
        self.N = 2
        self.T = 20
        self.dt = dt = 0.2

        #self.x0 = np.array([[-12,1,radians(0), -0.3, 0.2, 0],[1,0.2,0,0,0,0]])
        #self.x0 = np.array([[-10,-1.0,radians(15), -0.3, 2, 0],[-3,-1.0,0,0,0,0]])
        self.x0 = np.array([[-12, 1, radians(30), -0.3, 0.2, 0],
                            [2, -3, 0, 0, 0, 0]])

        # dimension of x and u for single agent
        # max(n^i)
        self.n = 6
        self.m = 2

        # bounds for visualization
        self.visual_x_lim = [-20, 20]
        self.visual_y_lim = [-2, 30]
        rocket_path = os.path.join(BASEDIR, f'resources/rocket_alpha.png')
        self.rocket_img = mpimg.imread(rocket_path)
        ship_path = os.path.join(BASEDIR, f'resources/ship_alpha.png')
        self.ship_img = mpimg.imread(ship_path)

        # cost functions R for rocket, S for ship
        self.Q_R = np.diag([1, 0, 1, 1, 1, 1])
        self.R_R = np.diag([1, 1]) * 1e-3

        self.Q_S = np.diag([0, 0, 0, 0, 0, 0])
        self.R_S = np.diag([1, 1]) * 1e-3

        self.Q_D = np.diag([1, 1])  # penalize dy, dvy
        self.P_R = np.zeros((self.m, self.n))
        self.P_R[0, 1] = 1
        self.P_R[1, 4] = 1
        self.P_S = np.zeros((self.m, self.n))
        self.P_S[0, 0] = 1
        self.P_S[1, 1] = 1
        self.print_debug_enable()
        # NOTE this is not implemented in cpp
        self.dynamics_residual_weight = 1.0

        u_ref = np.zeros((self.T, self.N, self.m))
        #u_ref[:,0,0] = 0.1
        #u_ref[:,1,0] = 1.0
        self.guess = u_ref

    def setup(self):
        # subclass responsible for loading cpp/eigen module
        # and setting x0
        if (self.config.USE_CPP or self.config.CPP_DEBUG):
            raise RuntimeError
            self.cpp = cpp_CarMergeKinematicBicycle(self.N, self.T, self.dt,
                                                    self.rho, self.rho_b,
                                                    self.bc_a, self.bc_b,
                                                    self.J_Qr, self.J_Q,
                                                    self.J_R, self.h_Qh,
                                                    self.target_y)
            self.cpp.set_x0(self.x0)

    def _visualize(self, u, x=None):
        rocket_scale = 0.01 / 2  # for rocket size
        ship_scale = 0.01 / 2  # for ship size

        if (x is None):
            x = np.vstack(
                [self.x0[np.newaxis, :, :],
                 self.rollout(self.x0, u)])
        rollout_X = np.vstack(
            [self.x0[np.newaxis, :, :],
             self.rollout(self.x0, u)])
        fig, ax = plt.subplots()

        # agent 0, rocket
        xx_0 = x[:, 0, 0]
        yy_0 = x[:, 0, 1]
        plt.plot(yy_0, -xx_0, '*-')
        # plot initial pose
        pose_0 = x[0, 0]
        rotated_rocket_img = rotate(self.rocket_img,
                                    degrees(pose_0[2]),
                                    reshape=True)
        L, W, _ = rotated_rocket_img.shape
        ax.imshow(rotated_rocket_img,
                  extent=[
                      pose_0[1] - W * rocket_scale,
                      pose_0[1] + W * rocket_scale,
                      -pose_0[0] - L * rocket_scale,
                      -pose_0[0] + L * rocket_scale
                  ])
        # plot final pose
        pose_f = x[-1, 0]
        rotated_rocket_img = rotate(self.rocket_img,
                                    degrees(pose_f[2]),
                                    reshape=True)
        L, W, _ = rotated_rocket_img.shape
        ax.imshow(rotated_rocket_img,
                  extent=[
                      pose_f[1] - W * rocket_scale,
                      pose_f[1] + W * rocket_scale,
                      -pose_f[0] - L * rocket_scale,
                      -pose_f[0] + L * rocket_scale
                  ])

        # agent 1: ship
        vertical_offset = -0.8
        pose_0 = x[:, 1, 0]
        plt.plot(pose_0, np.zeros_like(pose_0), '*-')
        # plot initial pose
        L, W, _ = self.ship_img.shape
        ax.imshow(self.ship_img,
                  extent=[
                      pose_0[0] - W * ship_scale, pose_0[0] + W * ship_scale,
                      vertical_offset - L * ship_scale,
                      vertical_offset + L * ship_scale
                  ])
        # plot final pose
        pose_f = x[-1, 1]
        L, W, _ = self.ship_img.shape
        ax.imshow(self.ship_img,
                  extent=[
                      pose_f[0] - W * ship_scale, pose_f[0] + W * ship_scale,
                      vertical_offset - L * ship_scale,
                      vertical_offset + L * ship_scale
                  ])
        ax.hlines(y=0, xmin=self.visual_x_lim[0], xmax=self.visual_x_lim[1])

        # DEBUG, plot rollout trajectory
        plt.plot(rollout_X[:, 0, 1], -rollout_X[:, 0, 0], 'o-')
        plt.plot(rollout_X[:, 1, 0], np.zeros_like(pose_0), 'o-')

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(*self.visual_x_lim)
        ax.set_ylim(*self.visual_y_lim)
        return fig

    # TODO
    def _animation(self, U, X=None, gif_prefix=''):
        return
        ''' build a gif animation'''
        if X is None:
            X = np.vstack(
                [self.x0[np.newaxis, :, :],
                 self.rollout(self.x0, U)])
        car_pos_vec = []
        car_angle_vec = []
        box_vec = []
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
                              color=color))

        fig, ax = plt.subplots()
        ax.set_xlim(*self.visual_x_lim)
        ax.set_ylim(*self.visual_y_lim)

        def update(frame):
            for i in range(self.N):
                box_vec[i].set_xy(car_pos_vec[i][frame])
                box_vec[i].set_angle(car_angle_vec[i][frame])

        # Add the boxes to the plot
        for box in box_vec:
            ax.add_patch(box)
        # lane boundary lines

        ax.set_aspect('equal', adjustable='box')

        # Create the animation
        anim = FuncAnimation(fig,
                             update,
                             frames=len(car_pos_vec[0]),
                             blit=True)
        gif_filename = self.resolveLogname(logPrefix=gif_prefix)
        anim.save(gif_filename, writer='pillow')
        plt.show()

    ''' --------  math functions and their derivatives ------ '''

    def J(self, x_k, u_k_i, i):
        '''
        step cost for an agent, given x,u
        x_k.shape (N*n) x_k_i = [x,y,vx,vy]
        u_k_i.shape (m) u_k_i = [ax, ay]
        i: agent id
        '''
        if (self.config.USE_CPP):
            return self.cpp.J(x_k, u_k_i, i)
        if (i == 0):
            val = u_k_i.T @ self.R_R @ u_k_i
        elif (i == 1):
            val = u_k_i.T @ self.R_S @ u_k_i

        if (self.config.CPP_DEBUG):
            alt = self.cpp.J(x_k, u_k_i, i)
            if (np.linalg.norm(alt - val) > 1e-4):
                breakpoint()
        return val

    # dJi dxi
    def dJi_dxi(self, x_k, u_k_i, i):
        if (self.config.USE_CPP):
            return self.cpp.dJi_dxi(x_k, u_k_i, i)
        if (self.config.CPP_DEBUG):
            alt = self.cpp.dJi_dxi(x_k, u_k_i, i)
            if (np.linalg.norm(alt - val) > 1e-4):
                breakpoint()
        val = np.zeros((1, self.n))
        if (self.config.DEBUG):
            if (i == 0):
                num = jacobianNumerical(
                    lambda xi: self.J(np.vstack([xi, x_k[1]]), u_k_i, i),
                    x_k[0])
            elif (i == 1):
                num = jacobianNumerical(
                    lambda xi: self.J(np.vstack([x_k[0], xi]), u_k_i, i),
                    x_k[1])
            assert (np.linalg.norm(num - val) < 1e-4)
        return val

    # dJi dxj
    def dJi_dxj(self, x_k, u_k_i, i, j):
        if (self.config.USE_CPP):
            return self.cpp.dJi_dxj(x_k, u_k_i, i, j)
        val = np.zeros((1, self.n))
        if (self.config.CPP_DEBUG):
            alt = self.cpp.dJi_dxj(x_k, u_k_i, i, j)
            if (np.linalg.norm(alt - val) > 1e-4):
                breakpoint()
        if (self.config.DEBUG):
            if (i == 0 and j == 1):
                num = jacobianNumerical(
                    lambda xj: self.J(np.vstack([x_k[0], xj]), u_k_i, i),
                    x_k[1].flatten())
            elif (i == 1 and j == 0):
                num = jacobianNumerical(
                    lambda xj: self.J(np.vstack([xj, x_k[1]]), u_k_i, i),
                    x_k[0].flatten())
            assert (np.linalg.norm(num - val) < 1e-4)
        return val

    def dJi_du(self, x_k, u_k_i, i):
        if (self.config.USE_CPP):
            return self.cpp.dJi_du(x_k, u_k_i, i)
        if (i == 0):
            val = 2 * u_k_i.T @ self.R_R
        elif (i == 1):
            val = 2 * u_k_i.T @ self.R_S

        if (self.config.CPP_DEBUG):
            alt = self.cpp.dJi_du(x_k, u_k_i, i)
            if (np.linalg.norm(alt - val) > 1e-4):
                breakpoint()
        if (self.config.DEBUG):
            num = jacobianNumerical(lambda xi: self.J(x_k, u_k_i, i), u_k_i)
            assert (np.linalg.norm(num - val) < 1e-4)
        return val

    def dJi_dudu(self, x_k, u_k_i, i):
        if (i == 0):
            val = 2 * self.R_R
        elif (i == 1):
            val = 2 * self.R_S
        return val

    def dJi_dxi_dxi(self, x_k, u_k_i, i):
        return np.zeros((self.n, self.n))

    def dJi_dxi_dxj(self, x_k, u_k_i, i, j):
        return np.zeros((self.n, self.n))

    def dJi_dxj_dxj(self, x_k, u_k_i, i, j):
        return np.zeros((self.n, self.n))

    # terminal cost
    def Jfi(self, x_k, i):
        '''
        step cost for an agent, given x,u
        x_k.shape (N*n) x_k_i = [x,y,vx,vy]
        u_k_i.shape (m) u_k_i = [ax, ay]
        i: agent id
        '''
        if (self.config.USE_CPP):
            return self.cpp.Jfi(x_k, i)
        if (i == 0):
            dx = self.P_R @ x_k[0] - self.P_S @ x_k[1]
            val = x_k[0].T @ self.Q_R @ x_k[0] + dx.T @ self.Q_D @ dx
        elif (i == 1):
            dx = self.P_R @ x_k[0] - self.P_S @ x_k[1]
            val = x_k[1].T @ self.Q_S @ x_k[1] + dx.T @ self.Q_D @ dx

        if (self.config.CPP_DEBUG):
            alt = self.cpp.Jfi(x_k, i)
            if (np.linalg.norm(alt - val) > 1e-4):
                breakpoint()
        if (len(val.shape) > 1):
            breakpoint()
        return val

    def dJfi_dxi(self, x_k, i):
        if (self.config.USE_CPP):
            return self.cpp.dJfi_dxi(x_k, i)
        if (i == 0):
            dx = self.P_R @ x_k[0] - self.P_S @ x_k[1]
            val = 2 * x_k[0] @ self.Q_R + 2 * dx.T @ self.Q_D @ self.P_R
        elif (i == 1):
            dx = self.P_R @ x_k[0] - self.P_S @ x_k[1]
            val = 2 * x_k[1] @ self.Q_S - 2 * dx.T @ self.Q_D @ self.P_S

        if (self.config.CPP_DEBUG):
            alt = self.cpp.dJfi_dxi(x_k, i)
            if (np.linalg.norm(alt - val) > 1e-4):
                breakpoint()
        if (self.config.DEBUG):
            if (i == 0):
                num = jacobianNumerical(
                    lambda xi: self.Jfi(np.vstack([xi, x_k[1]]), i), x_k[0])
            elif (i == 1):
                num = jacobianNumerical(
                    lambda xi: self.Jfi(np.vstack([x_k[0], xi]), i), x_k[1])
            assert (np.linalg.norm(num - val) < 1e-4)
        return val

    def dJfi_dxj(self, x_k, i, j):
        if (self.config.USE_CPP):
            return self.cpp.dJfi_dxj(x_k, i, j)
        if (i == 0 and j == 1):
            dx = self.P_R @ x_k[0] - self.P_S @ x_k[1]
            val = -2 * dx.T @ self.Q_D @ self.P_S
        elif (i == 1 and j == 0):
            dx = self.P_R @ x_k[0] - self.P_S @ x_k[1]
            val = 2 * dx.T @ self.Q_D @ self.P_R

        if (self.config.CPP_DEBUG):
            alt = self.cpp.dJfi_dxj(x_k, i, j)
            if (np.linalg.norm(alt - val) > 1e-4):
                breakpoint()
        if (self.config.DEBUG):
            if (i == 0 and j == 1):
                num = jacobianNumerical(
                    lambda xj: self.Jfi(np.vstack([x_k[0], xj]), i),
                    x_k[1].flatten())
            elif (i == 1 and j == 0):
                num = jacobianNumerical(
                    lambda xj: self.Jfi(np.vstack([xj, x_k[1]]), i),
                    x_k[0].flatten())
            assert (np.linalg.norm(num - val) < 1e-4)
        return val

    def dJfi_dxi_dxi(self, x_k, i):
        if (self.config.USE_CPP):
            return self.cpp.dJfi_dxi_dxi(x_k, i)
        if (i == 0):
            val = 2 * self.Q_R + 2 * self.P_R.T @ self.Q_D @ self.P_R
        elif (i == 1):
            val = 2 * self.Q_S + 2 * self.P_S.T @ self.Q_D @ self.P_S
        if (self.config.CPP_DEBUG):
            alt = self.cpp.dJfi_dxi_dxi(x_k, i)
            if (np.linalg.norm(alt - val) > 1e-4):
                breakpoint()
        return val

    def dJfi_dxi_dxj(self, x_k, i, j):
        if (i == 0 and j == 1):
            val = -2 * self.P_R.T @ self.Q_D @ self.P_S
        elif (i == 1 and j == 0):
            val = -2 * self.P_S.T @ self.Q_D @ self.P_R
        return val

    def dJfi_dxj_dxj(self, x_k, i, j):
        if (i == 0 and j == 1):
            val = 2 * self.P_S.T @ self.Q_D @ self.P_S
        elif (i == 1 and j == 0):
            val = 2 * self.P_R.T @ self.Q_D @ self.P_R
        return val

    # --- dynamics ---
    def f(self, x, u, i):
        if (i == 0):
            I = 1.0
            L = 1.0
            dx = np.array([
                x[3], x[4], x[5], u[0] * cos(x[2]) - u[1] * sin(x[2]) + 1.0,
                u[1] * cos(x[2]) + u[0] * sin(x[2]), u[1] * L / I
            ])
        elif (i == 1):
            dx = np.array([x[1], u[0], 0, 0, 0, 0])
        return x + dx * self.dt

    def df_dx(self, x, u, i):
        if (i == 0):
            A = np.zeros((self.n, self.n))
            A[0, 3] = 1
            A[1, 4] = 1
            A[2, 5] = 1
            A[3, 2] = -u[0] * sin(x[2]) - u[1] * cos(x[2])
            A[4, 2] = u[0] * cos(x[2]) - u[1] * sin(x[2])
        elif (i == 1):
            A = np.zeros((self.n, self.n))
            A[0, 1] = 1

        val = np.eye(self.n) + A * self.dt
        if (self.config.DEBUG):
            num = jacobianNumerical(lambda xx: self.f(xx, u, i), x, dim=self.n)
            assert (np.linalg.norm(num - val) < 1e-4)
        return val

    def df_du(self, x, u, i):
        if (i == 0):
            L = 1.0
            I = 1.0
            B = np.zeros((self.n, self.m))
            B[3, 0] = cos(x[2])
            B[3, 1] = -sin(x[2])
            B[4, 0] = sin(x[2])
            B[4, 1] = cos(x[2])
            B[5, 1] = L / I
        elif (i == 1):
            B = np.zeros((self.n, self.m))
            B[1, 0] = 1
        val = B * self.dt
        if (self.config.DEBUG):
            num = jacobianNumerical(lambda uu: self.f(x, uu, i), u, dim=self.n)
            assert (np.linalg.norm(num - val) < 1e-4)
        return val

    # handle collision
    def h(self, x_i, x_j):
        return -1

    def dh_dxi(self, x_i, x_j):
        return np.zeros(self.n)

    def dh_dxj(self, x_i, x_j):
        return np.zeros(self.n)

    def dh_dxi_dxi(self, x_i, x_j):
        return np.zeros((self.n, self.n))

    def dh_dxj_dxi(self, x_i, x_j):
        return np.zeros((self.n, self.n))

    def dh_dxi_dxj(self, x_i, x_j):
        return np.zeros((self.n, self.n))

    def dh_dxj_dxj(self, x_i, x_j):
        return np.zeros((self.n, self.n))

    def testAnimation(self):
        u_ref = np.zeros((self.T, self.N, self.m))
        x_ref = self.rollout(self.x0, u_ref)
        full_x_ref = np.vstack([self.x0[np.newaxis, :, :], x_ref])
        #self._animation(u_ref,full_x_ref)
        self._visualize(u_ref, full_x_ref)
        plt.show()


if __name__ == "__main__":
    main = RocketLanding(ResidualGameConfig())
    main.setup()
    main.solve(save_gif=False, visualize=True, animate=True)
    main.final()
    #main.testAnimation()
