# sanity check example for trivial game on double integrator
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
from residual_game import ResidualGame, ResidualGameConfig


class DoubleIntegrator(ResidualGame):

    def __init__(self, config: ResidualGameConfig):
        super().__init__(config)

        # agent 0:
        # x: [y,vy]
        # u: [ay]: unit:g

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
        self.N = 1
        self.T = 20
        self.dt = dt = 0.1

        self.x0 = np.array([[1, 0]])

        # dimension of x and u for single agent
        # max(n^i)
        self.n = 2
        self.m = 1

        # bounds for visualization
        self.visual_x_lim = [-20, 20]
        self.visual_y_lim = [-2, 30]

        # cost functions R for rocket, S for ship
        self.Q = np.diag([1.0, 1.0])
        self.R = np.diag([1]) * 1e-2

        self.print_debug_enable()
        # NOTE this is not implemented in cpp
        self.dynamics_residual_weight = 1.0

        u_ref = np.zeros((self.T, self.N, self.m))
        self.guess = u_ref

    def setup(self):
        # subclass responsible for loading cpp/eigen module
        # and setting x0
        if (self.config.USE_CPP or self.config.CPP_DEBUG):
            raise RuntimeError

    def _visualize(self, u, x=None):

        if (x is None):
            x = np.vstack(
                [self.x0[np.newaxis, :, :],
                 self.rollout(self.x0, u)])
        rollout_X = np.vstack(
            [self.x0[np.newaxis, :, :],
             self.rollout(self.x0, u)])
        fig, ax = plt.subplots()

        # plot rollout trajectory
        plt.plot(x[:, 0, 0], 'r-')
        plt.plot(x[:, 0, 1], 'r-')
        plt.plot(rollout_X[:, 0, 0], 'bo-')
        plt.plot(rollout_X[:, 0, 1], 'bo-')
        return fig

    def _animation(self, U, X=None, gif_prefix=''):
        return

    ''' --------  math functions and their derivatives ------ '''

    def J(self, x_k, u_k_i, i):
        '''
        step cost for an agent, given x,u
        i: agent id
        '''
        val = x_k[i].T @ self.Q @ x_k[i] + u_k_i.T @ self.R @ u_k_i
        return val

    # dJi dxi
    def dJi_dxi(self, x_k, u_k_i, i):
        val = 2 * x_k[i] @ self.Q
        return val

    # dJi dxj
    def dJi_dxj(self, x_k, u_k_i, i, j):
        return np.zeros((1, self.n))

    def dJi_du(self, x_k, u_k_i, i):
        val = 2 * u_k_i.T @ self.R
        return val

    # dJ^i / dxi dxi
    def dJi_dxi_dxi(self, x_k, u, i):
        return 2 * self.Q

    # dJi / dxi dxj
    def dJi_dxi_dxj(self, x_k, u_k_i, i, j):
        return np.zeros((self.n, self.n))

    def dJi_dxj_dxj(self, x_k, u_k_i, i, j):
        return np.zeros((self.n, self.n))

    def dJi_dudu(self, x_k, u_k_i, i):
        val = 2 * self.R
        return val

    def f(self, x, u, i):
        dx = np.array([x[1], u[0]])
        return x + dx * self.dt

    def df_dx(self, x, u, i):
        A = np.zeros((self.n, self.n))
        A[0, 1] = 1
        val = np.eye(self.n) + A * self.dt
        return val

    def df_du(self, x, u, i):
        B = np.zeros((self.n, self.m))
        B[1, 0] = 1
        val = B * self.dt
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
    main = DoubleIntegrator(ResidualGameConfig())
    main.setup()
    main.solve()
    main.final()
    #main.testAnimation()
