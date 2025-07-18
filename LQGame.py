# LQGame for comparison

import os
import sys
import pickle
from time import time
from itertools import chain

import numpy as np
from math import sin, cos, tan, radians, degrees, pi, atan
from scipy import interpolate
from scipy.linalg import block_diag
import scipy.sparse

from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

from utilities.util import *
from utilities.time_util import TimeUtil
from LQGameSolver import my_solve_lq_game
from typing import NamedTuple


class LQGameConfig(NamedTuple):
    tolerance: float = 5e-4
    iterations: int = 100


class LQGame(PrintObject):

    def __init__(self):
        # penalty on dx
        self.normalization_cost = 100

        # NOTE: These parameters depend on the actual problem and will be overridden in subclass
        self.N = None
        self.dt = dt = None
        # Dimension of x and u for single agent
        self.n = None
        self.m = None
        self.x0 = None

        # Solver variables
        self.frame_vec = []
        self.residuals = None
        self.violations = None
        self.profiler = TimeUtil(False)
        self.alpha = 1

    def solve(self, save_gif=False, visualize=False, animate=False):
        """Runs main iteration loop, sets up variables and handles visualization.

        Args:
            save_gif: Boolean to toggle feature. Defaults to False.
            visualize: Boolean to toggle feature. Defaults to False.
            animate: Boolean to toggle feature. Defaults to False.
        """
        N = self.N
        T = self.T
        n = self.n
        m = self.m

        Ps = [np.array([np.zeros((m, n * N))] * T) for i in range(N)]
        alphas = [np.array([np.zeros((m, 1))] * T) for i in range(N)]
        u_ref = np.array([np.zeros((N, m, 1)) for i in range(T)])  # T, N,m,1
        x_ref = np.array(self.rollout(self.x0, np.array(u_ref)))
        '''
        with open('u_ref.p','rb') as f:
            u_ref = pickle.load(f)
        '''

        full_x = []

        self.visualize(np.array(u_ref),
                       visualize=visualize,
                       animate=animate,
                       gif_prefix='before')
        t0 = time()
        t = self.profiler
        i = 0
        has_converged = False

        while True:
            t.s()
            if i == self.config.iterations:  # NOTE: Adjust to whatever max iteration desired
                self.print_ok('iteration limit reached, now stopping')
                break
            try:
                x_ref, u_ref, Ps, alphas, full_x = self.solve_iteration(
                    x_ref, u_ref, Ps, alphas, i, full_x)
                '''
                with open('u_ref.p','wb') as f:
                    pickle.dump(u_ref,f)
                '''
            except StopIteration:
                self.print_ok('Stopping criterion met!')
                has_converged = True
                break
            t.e()
            self.print_info(f'------ {N} agents, iter {i}------')
            #self.visualize(np.array(u_ref), x_ref, visualize, save_gif, gif_prefix = 'after')
            i += 1

        t_solve = time() - t0
        self.print_info(f'total solve time: {t_solve}')
        #full_x_ref = np.vstack([self.x0[np.newaxis, :, :], x_ref])
        self.visualize(np.array(u_ref),
                       x_ref,
                       visualize,
                       save_gif,
                       animate,
                       gif_prefix='after')
        return u_ref, x_ref, has_converged

    def solve_iteration(self, x_ref, u_ref, Ps, alphas, iteration, full_x):
        """Solves a single iteration of LQGame.

        Args:
            x_ref: Reference trajectory over time frame, [T * N * n]
            u_ref: Reference controls over time frame, [T * N * m]
            Ps: Optimal strategy from ARE solution, [T * N * n * m]
            alphas: Affine terms from optimal strategy, [T * N * m]
            iteration: iteration number
            full_x: full list of trajectories for each iteration

        Raises:
            StopIteration: Stops loop once convergence condition has been reached

        Returns:
            Reference trajectory and controls of horizon, optimal solution to ARE, and updated trajectory list for each
                iteration
        """
        N = self.N
        T = horizon = self.T
        n = self.n
        m = self.m
        xx = []
        xx.append([self.x0[i] for i in range(N)])  # T,N
        uu = [[] for i in range(T)]  # T,N
        As_agents = [[] for i in range(N)]  # N,T
        Bs_agents = [[] for i in range(N)]
        # 1. Find new control based on optimal solution and then roll out new trajectory + linearize
        for t in range(T):
            dx = np.hstack([xx[-1][i] - x_ref[t][i] for i in range(N)])
            xx_t = []
            for i in range(N):
                new_u_i = u_ref[t][i] - Ps[i][t] @ dx.reshape(-1,
                                                              1) + alphas[i][t]
                new_x_i = self.f(xx[-1][i], new_u_i.T[0], i)
                A = self.df_dx(new_x_i, new_u_i, i) * self.dt
                B = self.df_du(new_x_i, new_u_i, i) * self.dt
                xx_t.append(new_x_i)
                uu[t].append(new_u_i)
                As_agents[i].append(A)
                Bs_agents[i].append(B)
            xx.append(xx_t)
        x_ref = np.array(xx)
        u_ref = np.array(uu)

        As = [
            block_diag(*[As_agents[i][t] for i in range(N)]) for t in range(T)
        ]
        Bs = [[
            np.vstack([
                Bs_agents[i][t] if i == j else np.zeros((n, m))
                for i in range(N)
            ]) for t in range(T)
        ] for j in range(N)]

        # 2. Find cost matrices using new trajectory and controls
        Qs, qs, Rs, rs, has_collision = self.getCostMatrices(
            x_ref, u_ref, iteration)

        # 3. Find optimal solution
        new_Ps, new_alphas = my_solve_lq_game(As, Bs, Qs, qs, Rs, rs,
                                              self.profiler)
        new_Ps = np.array(new_Ps)
        new_alphas = np.array(new_alphas)

        # DEBUG test which option gives lower cost
        # only test feedforward
        # original cost = 0
        '''
        cost_vec = [0]*N
        dx = np.zeros((n*N,1))
        for k in range(T):
            self.print_debug(f'step {k}')
            dx = As[k] @ dx
            for i in range(N):
                du = new_alphas[i][k] + new_Ps[i][k] @ dx
                dx += Bs[i][k] @ du
            for i in range(N):
                du = new_alphas[i][k]
                step_cost = 0.5*dx.T @ Qs[i][k] @ dx + qs[i][k].T @ dx + 0.5*du.T @ Rs[i][k] @ du + rs[i][k].T @ du
                cost_vec[i] += step_cost
                self.print_debug(f'\t agent {i}cost {step_cost}')
        for i in range(N):
            self.print_debug(f'agent {i}, d_cost = {cost_vec[i]}')
        '''

        alpha = 1.0  # NOTE: This determines "how much of" the new solution we want to be applied to next iter
        if (iteration == 0):
            alphas = alpha * new_alphas
            Ps = alpha * new_Ps
        else:
            alphas = alphas * (1 - alpha) + alpha * new_alphas
            Ps = Ps * (1 - alpha) + alpha * new_Ps

        # Backtracking line search implementation
        '''
        cost = 0
        for t in range(T):
            for i in range(N):
                cost += self.J(x_ref[t], u_ref[t][i], i)

        # NOTE: Testing backtracking line search:
        step_size = 1.0
        for j in range(10):
            x_ls = [self.x0]
            u_ls = []
            cost_ls = 0
            for t in range(T):
                u_ls_t = []
                x_ls_t = []
                dx_ls = np.hstack([x_ls[-1][i] - x_ref[t][i] for i in range(N)])
                for i in range(N):
                    u_ls_k_i = u_ref[t][i] + step_size * alphas[i][t] + Ps[i][t] @ (dx_ls).reshape(-1, 1)
                    x_next = self.f(x_ls[-1][i], u_ls_k_i.T[0], i)
                    cost_ls += self.J(x_ls[-1], u_ls_k_i.T[0], i)
                    u_ls_t.append(u_ls_k_i)
                    x_ls_t.append(x_next)
                x_ls.append(x_ls_t)
                u_ls.append(u_ls_t)
            if np.linalg.norm(cost_ls) <= np.linalg.norm(cost) * 0.75:
                x_new = self.rollout(self.x0, np.array(u_ls))
                u_ref = u_ls
                break
            else:
                step_size *= 0.50
        '''

        # We want to stop the loop early if converged upon a viable solution. This is defined when the last
        # 3 iterations have produced solutions that are close enough to each other (norm < 0.1). This tolerance can be adjusted.
        full_x.append(x_ref)
        if not has_collision:
            if iteration >= 3:
                norm_a = np.linalg.norm(np.array(x_ref) - np.array(full_x[-1]))
                norm_b = np.linalg.norm(np.array(x_ref) - np.array(full_x[-2]))
                if norm_a <= self.config.tolerance and norm_b <= self.config.tolerance:
                    raise StopIteration

        return x_ref, u_ref, Ps, alphas, full_x

    def getCostMatrices(self, xx, uu, iteration):
        """Generates cost matrices for each time step based on Jacobians and Hessians of cost function.

        Args:
            xx: States over time horizon
            uu: Controls over time horizon
            iteration: Iteration count

        Returns:
            Quadratic and linear cost terms with respect to state and controls
        """
        global scalar
        N = self.N
        T = horizon = self.T
        m = self.m
        n = self.n
        # first we formulate cost on x,u, and later transform it to cost on dx, du
        Qs = [[] for i in range(N)]
        qs = [[] for i in range(N)]
        Rs = [[] for i in range(N)]
        rs = [[] for i in range(N)]

        II = np.hstack([np.eye(n), -np.eye(n)])
        R0 = np.zeros((m, m))

        has_collision = False

        for t in range(horizon):
            for i in range(N):
                # NOTE: q, r will be reshaped into column vectors
                Q_i = np.zeros((N * n, N * n))
                q_i = np.zeros((1, N * n))
                R_i = np.zeros((m, m))
                r_i = np.zeros((m, 1))

                # Check collision violation + apply cost
                for j in range(i + 1, N):
                    h = -((xx[t][i][0] - xx[t][j][0]) / 1.0)**2 - (
                        xx[t][i][1] - xx[t][j][1])**2 + 7
                    if h >= 0:
                        has_collision = True
                        # for agent i
                        # Hessian dh/dxdx
                        Q_i_col = 2 * self.h_Qh.T
                        Q_i[i * n:(i + 1) * n, i * n:(i + 1) * n] += Q_i_col

                        #Gradient dh/dx
                        q_i_col = 2 * (xx[t][i] - xx[t][j]).T @ self.h_Qh
                        q_i[0, i * n:(i + 1) * n] += q_i_col

                        # for agent j
                        # Hessian dh/dxdx
                        Q_j_col = 2 * self.h_Qh.T
                        Q_i[j * n:(j + 1) * n, j * n:(j + 1) * n] += Q_j_col

                        #Gradient dh/dx
                        q_j_col = 2 * (xx[t][j] - xx[t][i]).T @ self.h_Qh
                        q_i[0, j * n:(j + 1) * n] += q_j_col

                # Step cost

                # Hessian dJ/dxdx
                Q_i_step = 2 * (self.J_Qr +
                                self.J_Q) + self.normalization_cost * np.eye(n)
                Q_i[i * n:(i + 1) * n, i * n:(i + 1) * n] += Q_i_step

                # Gradient dJ/dx
                q_i_step = 2 * (xx[t][i] - self.J_x_ref_fun(i)
                                ).T @ self.J_Qr + 2 * xx[t][i].T @ self.J_Q
                q_i[0, i * n:(i + 1) * n] += q_i_step

                # We want to dampen the swerve reactions from the main lane cars, so we have control cost specific for these cars
                # Hessian dJ/dudu
                '''
                if (i < self.main_lane_n):
                    R_i = 2*self.J_R_main
                else:
                    R_i = 2*self.J_R_merge
                # Gradient dJ/du
                if (i < self.main_lane_n):
                    r_i = 2 * uu[t][i].T @ self.J_R_main
                else:
                    r_i = 2 * uu[t][i].T @ self.J_R_merge
                '''
                R_i = 2 * self.J_R
                r_i = 2 * uu[t][i].T @ self.J_R

                Qs[i].append(Q_i)
                qs[i].append(q_i.reshape(-1, 1))
                Rs[i].append(R_i)
                rs[i].append(r_i.reshape(-1, 1))
        return Qs, qs, Rs, rs, has_collision

    def J(self, x_k, u_k_i, i):
        """_summary_

        Args:
            xx: _description_
            uu: _description_
            i: _description_
        """
        if (i < self.main_lane_n):
            J_R = self.J_R_main
        else:
            J_R = self.J_R_merge
        cost = (x_k[i] - self.J_x_ref_fun(i)).T @ self.J_Qr @ (
            x_k[i] - self.J_x_ref_fun(i)
        ) + x_k[i].T @ self.J_Q @ x_k[i] + u_k_i.T @ J_R @ u_k_i
        return cost

    def rollout(self, x0, U):
        """ Given initial state and controls for time horizon, find state for every time step based on controls

        Args:
            x0: Initial state, (N * n)
            U: Controls over time horizon, (T * N * m)

        Returns:
            Vehicle states for each time step
        """
        U = U.reshape(self.T, self.N, self.m)
        X = np.zeros((self.T + 1, self.N, self.n))
        X[0, :, :] = x0.reshape(self.N, self.n)
        # x+ = x + vx*dt + 0.5*ax*dt*dt
        # vx+ = vx + ax*dt
        for i in range(self.N):
            for k in range(1, self.T + 1):
                X[k, i] = self.f(X[k - 1, i], U[k - 1, i], i)
        return X[1:, :, :]

    def visualize(self,
                  U,
                  X=None,
                  visualize=False,
                  save_gif=False,
                  animate=False,
                  gif_prefix='run'):
        """Handles visualization of the simulation.

        Args:
            U: Controls over time horizon
            X: States over time horizon. Defaults to None.
            visualize: Boolean toggling this feature. Defaults to False.
            save_gif: Boolean toggling this feature. Defaults to False.
            animate: Boolean toggling this feature. Defaults to False.
            gif_prefix: Prefix for saved file. Defaults to 'run'.
        """
        if (visualize or save_gif):
            fig = self._visualize(U, X)
            if (save_gif):
                fig.canvas.draw()
                frame = Image.frombytes('RGB', fig.canvas.get_width_height(),
                                        fig.canvas.tostring_rgb())
                self.frame_vec.append(frame)
            if (visualize):
                plt.show()
        if (animate):
            self._animation(U, X, gif_prefix=gif_prefix)

        return

    def resolveLogname(self, logPrefix='run'):
        """Saves gif file.

        Args:
            logPrefix: Prefix for file name. Defaults to 'run'.

        Returns:
            Name of file
        """
        # setup log file
        # log file will record state of the vehicle for later analysis
        logFolder = os.path.abspath(
            os.path.join(os.path.dirname(__file__), 'gifs'))
        logSuffix = ".gif"
        no = 1
        while os.path.isfile(logFolder + logPrefix + str(no) + logSuffix):
            no += 1

        log_no = no
        logFilename = logFolder + logPrefix + str(no) + logSuffix
        return logFilename

    def final(self):
        """Print final profiler stats, saves gif and confirms file save
        """
        self.profiler.summary()
        if (len(self.frame_vec) > 0):
            gif_filename = self.resolveLogname()
            self.frame_vec[0].save(fp=gif_filename,
                                   format='GIF',
                                   append_images=self.frame_vec,
                                   save_all=True,
                                   duration=200,
                                   loop=0)
            self.print_debug(f'GIf saved to {gif_filename}')
