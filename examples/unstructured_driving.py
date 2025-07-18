import numpy as np
from time import time
from PIL import Image
from scipy import interpolate
import scipy.sparse  # sparse matrix operations
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

from utilities.util import *
from utilities.time_util import TimeUtil
from src.build.unstructured_driving import UnstructuredDriving as cpp_UnstructuredDriving
from residual_game import ResidualGame, ResidualGameConfig


# example: unstructured lane change
# uses double integrator dynamics
# this version use U as decision variable only
class UnstructuredDriving(ResidualGame):

    def __init__(self, config: ResidualGameConfig, car_count=3):
        super().__init__()

        # u_i = [ax,ay] longitudinal, lateral acceleration
        # x_i = [x,y,vx,vy]
        # x+_i = f(x_i,u_i) = [x + vx*dt + 0.5*ax*dt*dt, y + vy*dt + 0.5*ay*dt*dt ]
        # s.t. [(xi-xj)/dx]**2 + [(yi-yj)/dy]**2 >= 1
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
        # decision variables:
        self.N = car_count
        self.track_width = 5
        self.track_length = 20
        self.dt = dt = 0.25
        #self.dt = dt = 0.1

        # dimension of x and u for single agent
        self.n = 4
        self.m = 2

        # bounds for visualization
        self.visual_x_lim = [-2.5, 2.5]
        self.visual_y_lim = [-2, 30]

        # initial state, stated in unit of car size
        self.x0 = np.array([[0, -0.9, 1.5, 0.5], [0, 0.4, 2.5, 0.2],
                            [0, 2.0, 1.7, -0.3]])
        self.target_y = [-1.0, 1.0, 1.0]
        #self.x0 = x0 = np.array([[0,-0.9,1.5,0.5],[0,0.4,2.5,0.2],[0,2.0,1.7,-0.3],[0.3,-0.3,1.3,0.3],[-0.4,0.8,0.4,1.0],[1.0,0.0,0.1,0.3]])
        #self.target_y = [-2.0,-1.0,1.0,1.4,1.7,2.0]

        # step cost parameters
        # NOTE this lambda fun needs to be implemented in c++
        self.J_x_ref_fun = lambda i: np.array([0, self.target_y[i], 2.0, 0])
        self.J_Qr = np.diag([0, 1, 1, 0])
        self.J_Q = np.diag([0, 0, 0, 1e-2])
        self.J_R = np.eye(self.m) * 1e-2

        # dynamics parameters
        self.A = np.eye(self.n)
        self.A[0, 2] = dt
        self.A[1, 3] = dt
        self.B = np.array([[0.5 * dt**2, 0], [0, 0.5 * dt**2], [dt, 0],
                           [0, dt]])

        # collision definition
        self.h_Qh = np.diag([-1, -1, 0, 0])

    def setup(self):
        # subclass responsible for loading cpp/eigen module
        # and setting x0
        if (self.config.USE_CPP or self.config.CPP_DEBUG):
            self.cpp = cpp_UnstructuredDriving(
                self.N, self.T, self.dt, self.rho, self.rho_b, self.bc_a,
                self.bc_b, self.config.tolerance, self.backtracking_max_iter,
                self.J_Qr, self.J_Q, self.J_R, self.A, self.B, self.h_Qh,
                self.target_y, self.config.iterations, False)
            self.cpp.set_x0(self.x0)

    def _visualize(self, u, x=None):
        if (x is None):
            x = np.vstack(
                [self.x0[np.newaxis, :, :],
                 self.rollout(self.x0, u)])
        fig, ax = plt.subplots()
        ax.vlines(x=-self.track_width / 2, ymin=-1, ymax=self.track_length)
        ax.vlines(x=self.track_width / 2, ymin=-1, ymax=self.track_length)
        for i in range(self.N):
            xx = x[:, i, 0]
            yy = x[:, i, 1]
            plt.plot(yy, xx, '*-')
        ax.set_aspect('equal', adjustable='box')
        return fig

    def _animation(self, U, X=None, gif_prefix=''):
        ''' build a gif animation'''
        if X is None:
            X = np.vstack(
                [self.x0[np.newaxis, :, :],
                 self.rollout(self.x0, U)])
        car_pos_vec = []
        box_vec = []
        color_vec = ['red', 'green', 'blue', 'black']
        color_vec = [color_vec[i % len(color_vec)] for i in range(self.N)]
        # prepare smoothed animation
        for i, color in zip(range(self.N), color_vec):
            tt = np.linspace(0, self.dt * self.T, self.T + 1)
            xx = X[:, i, 0]
            yy = X[:, i, 1]
            xx_fun = interpolate.interp1d(tt, xx)
            yy_fun = interpolate.interp1d(tt, yy)

            # interpolate for smooth graphics
            #tt = np.linspace(0,self.T*self.dt,50)
            pos_vec = np.vstack([yy_fun(tt), xx_fun(tt)]).T
            car_pos_vec.append(pos_vec)
            box_vec.append(plt.Rectangle(pos_vec[0], 1, 1, color=color))

        fig, ax = plt.subplots()
        ax.set_xlim(*self.visual_x_lim)
        ax.set_ylim(*self.visual_y_lim)

        def update(frame):
            for i in range(self.N):
                box_vec[i].set_xy(car_pos_vec[i][frame])

        # Add the boxes to the plot
        for box in box_vec:
            ax.add_patch(box)
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

    # this problem has homogeneous agents, so [i] is irrelevant
    def f(self, x, u, i):
        '''
        return np.array([x[0] + x[2]*self.dt + 0.5*self.dt*self.dt*u[0]/10,
        x[1] + x[3]*self.dt + 0.5*self.dt*self.dt*u[1]/10,
        x[2] + self.dt*u[0]/10,
        x[3] + self.dt*u[1]/10])
        '''
        return self.A @ x + self.B @ u

    def df_dx(self, x, u, i):
        return self.A

    def df_du(self, x, u, i):
        return self.B

    def h(self, x_i, x_j):
        ''' car distance larger than 1.0 '''
        if (self.config.USE_CPP):
            return self.cpp.h(x_i, x_j)
        #return (x_i-x_j).T @ self.h_Qh @ (x_i-x_j) + 1.0**2
        # below is faster
        #return -(x_i[0]-x_j[0])**2 - (x_i[1]-x_j[1])**2 + 1.0**2
        val = -(x_i[0] - x_j[0])**2 - (x_i[1] - x_j[1])**2 + 1.2**2
        if (self.config.CPP_DEBUG):
            alt = self.cpp.h(x_i, x_j)
            if (np.linalg.norm(alt - val) > 1e-4):
                breakpoint()
        return val

    def dh_dxi(self, x_i, x_j):
        if (self.config.USE_CPP):
            return self.cpp.dh_dxi(x_i, x_j)
        val = 2 * (x_i - x_j).T @ self.h_Qh
        if (self.config.CPP_DEBUG):
            alt = self.cpp.dh_dxi(x_i, x_j)
            if (np.linalg.norm(alt - val) > 1e-4):
                breakpoint()
        return val

    def dh_dxj(self, x_i, x_j):
        if (self.config.USE_CPP):
            return self.cpp.dh_dxj(x_i, x_j)
        val = 2 * (x_j - x_i).T @ self.h_Qh
        if (self.config.CPP_DEBUG):
            alt = self.cpp.dh_dxj(x_i, x_j)
            if (np.linalg.norm(alt - val) > 1e-4):
                breakpoint()
        return val

    def dh_dxi_dxi(self, x_i, x_j):
        if (self.config.USE_CPP):
            return self.cpp.dh_dxi_dxi(x_i, x_j)
        val = 2 * self.h_Qh.T
        if (self.config.CPP_DEBUG):
            alt = self.cpp.dh_dxi_dxi(x_i, x_j)
            if (np.linalg.norm(alt - val) > 1e-4):
                breakpoint()
        return val

    def dh_dxj_dxi(self, x_i, x_j):
        if (self.config.USE_CPP):
            return self.cpp.dh_dxj_dxi(x_i, x_j)
        val = -2 * self.h_Qh.T
        if (self.config.CPP_DEBUG):
            alt = self.cpp.dh_dxj_dxi(x_i, x_j)
            if (np.linalg.norm(alt - val) > 1e-4):
                breakpoint()
        return val

    def dh_dxi_dxj(self, x_i, x_j):
        if (self.config.USE_CPP):
            return self.cpp.dh_dxi_dxj(x_i, x_j)
        val = -2 * self.h_Qh.T
        if (self.config.CPP_DEBUG):
            alt = self.cpp.dh_dxi_dxj(x_i, x_j)
            if (np.linalg.norm(alt - val) > 1e-4):
                breakpoint()
        return val

    def dh_dxj_dxj(self, x_i, x_j):
        if (self.config.USE_CPP):
            return self.cpp.dh_dxj_dxj(x_i, x_j)
        val = 2 * self.h_Qh.T
        if (self.config.CPP_DEBUG):
            alt = self.cpp.dh_dxj_dxj(x_i, x_j)
            if (np.linalg.norm(alt - val) > 1e-4):
                breakpoint()
        return val


if __name__ == "__main__":
    main = UnstructuredDriving(ResidualGameConfig())
    main.setup()
    main.solve(save_gif=False, visualize=True)
    main.final()
