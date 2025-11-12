import numpy as np
from time import time
from math import sin, cos, tan, atan, radians, degrees
from PIL import Image
from scipy import interpolate
import scipy.sparse  # sparse matrix operations
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from matplotlib.animation import FuncAnimation
import matplotlib.image as mpimg
from scipy.ndimage import rotate

from utilities.util import *
from utilities.time_util import TimeUtil
from src.build.car_merge_kinematic_bicycle import CarMergeKinematicBicycle as cpp_CarMergeKinematicBicycle
from residual_game import ResidualGame, ResidualGameConfig


# example: Merging
# uses kinematic bicycle model
class CarMergeKinematicBicycle(ResidualGame):

    def __init__(self, config: ResidualGameConfig, car_count: int, T: int):
        super().__init__(config)
        # u_i = [throttle, steering]
        # x_i = [x,y,v,theta]: x: upwards, y:leftward, theta: ccw (right hand coord)
        # collision constraint: [(xi-xj)/dx]**2 + [(yi-yj)/dy]**2 >= 1
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
        self.print_debug_enable()

        # Problem formulation
        # decision variables:
        self.N = car_count
        self.T = T
        self.track_width = 2.2
        self.track_length = 20
        self.collision_radius = 2.0
        self.dt = dt = 0.2
        # NOTE this is not implemented in cpp
        self.dynamics_residual_weight = 1.0

        # dimension of x and u for single agent
        self.n = 4
        self.m = 2

        # stein sampling prior
        self.dim_theta = self.T * self.N * self.m
        self.covariance_mtx = np.diag([2 * 0.5] * self.dim_theta)

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
                    os.path.join(BASEDIR, f'resources/porsche_{color}.png'))
                for color in color_names
            ]

        # collision definition
        self.h_Qh = np.diag([-1.0, -1, 0, 0])
        '''
        # initial state,
        # NOTE this is 3 car simple case, it will be overridden
        self.x0 = np.array([[0,0.9,1.5,0.0],[4,1.1,1.5,0.0],[2.1,-1.3,1.7,0.0]])
        self.target_y = [1.0,1.0,1.0]
        '''

        # step cost parameters
        # NOTE this lambda fun needs to be implemented in c++
        self.J_x_ref_fun = lambda i: np.array([0, self.target_y[i], 2.0, 0])
        self.J_Qr = np.diag([0, 0.1, 0.01, 0])
        self.J_Q = np.diag([0, 0, 0, 1.0])
        self.J_R = np.eye(self.m) * 0.3
        self.guess = np.zeros((self.T, self.N, self.m))

        # multiple car merge, car_count: main_lane_n + merge_lane_n
        main_lane_n = min(int(0.67 * car_count), car_count - 1)
        merge_lane_n = car_count - main_lane_n
        x_pos_main_lane = np.linspace(
            0, (main_lane_n - 1) * 5.4,
            main_lane_n) + np.random.random(main_lane_n)
        # x_pos_merge_lane = 2.5+np.linspace(0,(merge_lane_n-1)*5.4,merge_lane_n) + np.random.random(merge_lane_n)
        x_pos_merge_lane = (np.random.random() - 0.5) * 2 * 2.5 + np.linspace(
            0, (merge_lane_n - 1) * 5.4,
            merge_lane_n) + np.random.random(merge_lane_n)
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

        # DEBUG print Dr dimension
        T = self.T
        N = self.N
        m = self.m
        n = self.n
        dim_y = T * N * n + T * N * m + T * N * n + T * N * N
        # self.print_debug(f"dim_y = {dim_y}, Dr memory: {(dim_y**2)*8/1024}KB")

    def setup(self):
        # subclass responsible for loading cpp/eigen module
        # and setting x0
        if (self.config.USE_CPP or self.config.CPP_DEBUG):
            self.cpp = cpp_CarMergeKinematicBicycle(
                self.N, self.T, self.dt, self.rho, self.rho_b, self.bc_a,
                self.bc_b, self.config.tolerance, self.backtracking_max_iter,
                self.J_Qr, self.J_Q, self.J_R, self.h_Qh, self.target_y,
                self.collision_radius, self.config.iterations, False)
            self.cpp.set_x0(self.x0)

    def _visualize(self, u, x=None):
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
        ''' build a gif animation'''
        if X is None:
            X = np.vstack(
                [self.x0[np.newaxis, :, :],
                 self.rollout(self.x0, U)])
        fig, ax = plt.subplots()

        if (self.sprite_visualization):
            car_scale = self.car_scale
            # draw car sprite
            car_pose_vec = []
            for states in X:
                car_pose_vec.append(
                    [[-states[i][1], states[i][0], states[i][3] + np.pi / 2]
                     for i in range(self.N)])

            im_vec = []
            for i in range(self.N):
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
                for i in range(self.N):
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
            color_vec = [color_vec[i % len(color_vec)] for i in range(self.N)]
            # prepare smoothed animation
            for i, color in zip(range(self.N), color_vec):
                # interpolate for smooth graphics
                # tt = np.linspace(0,self.T*self.dt,50)
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
                               radius=(self.collision_radius) / 2,
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
        ax.vlines(x=-self.track_width,
                  ymin=self.visual_y_lim[0],
                  ymax=self.visual_y_lim[1],
                  colors='white')
        ax.vlines(x=self.track_width,
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
        anim = FuncAnimation(fig, update, frames=self.T, blit=True)

        gif_filename = self.resolveLogname(logPrefix=gif_prefix)
        anim.save(gif_filename, writer='pillow')
        self.print_info(f'gif saved to {gif_filename}')
        plt.show()
        '''
        # NOTE save initial, middle, final snapshots
        update(0)
        fig.canvas.draw()
        frame = Image.frombytes('RGB',
        fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
        filename = f'./pics/merge_{self.N}car_initial.png'
        self.print_info(f'saved to {filename}')
        frame.save(filename)

        update(self.T//2)
        fig.canvas.draw()
        frame = Image.frombytes('RGB',
        fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
        filename = f'./pics/merge_{self.N}car_middle.png'
        self.print_info(f'saved to {filename}')
        frame.save(filename)

        update(self.T-1)
        fig.canvas.draw()
        frame = Image.frombytes('RGB',
        fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
        filename = f'./pics/merge_{self.N}car_final.png'
        self.print_info(f'saved to {filename}')
        frame.save(filename)
        '''

    ''' --------  math functions and their derivatives ------ '''

    def J(self, x_k, u_k_i, i):
        '''
        step cost for an agent, given x,u
        x_k.shape (N*n) x_k_i
        u_k_i.shape (m) u_k_i
        i: agent id
        '''
        # return (x[2] - 2.0)**2 + (x[1] - self.target_y[i])**2 + 1e-2*x[3]**2 + 1e-2*u.T @ np.eye(self.m) @ u
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
        lf = 1.0
        lr = 1.0
        beta = atan(tan(u[1]) * lr / (lf + lr))
        dx = np.array([
            x[2] * cos(x[3] + beta), x[2] * sin(x[3] + beta), u[0],
            x[2] / lr * sin(beta)
        ])
        return x + dx * self.dt

    def df_dx(self, x, u, i):
        beta = atan(tan(u[1]) * 0.5)
        A = np.array([[0, 0, cos(x[3] + beta), -x[2] * sin(x[3] + beta)],
                      [0, 0, sin(x[3] + beta), x[2] * cos(x[3] + beta)],
                      [0, 0, 0, 0], [0, 0, sin(beta) / 1.0, 0]])
        val = np.eye(4) + A * self.dt
        if (self.config.DEBUG):
            num = jacobianNumerical(lambda xx: self.f(xx, u, i), x, dim=self.n)
            assert (np.linalg.norm(num - val) < 1e-4)
        return val

    def df_du(self, x, u, i):
        beta = atan(tan(u[1]) * 0.5)
        dbeta_dst = 0.5 / (((tan(u[1]) * 0.5)**2 + 1) * cos(u[1])**2)
        B = np.array([[0, -x[2] * sin(x[3] + beta) * dbeta_dst],
                      [0, x[2] * cos(x[3] + beta) * dbeta_dst], [1, 0],
                      [0, x[2] / 1.0 * cos(beta) * dbeta_dst]])
        val = B * self.dt
        if (self.config.DEBUG):
            num = jacobianNumerical(lambda uu: self.f(x, uu, i), u, dim=self.n)
            assert (np.linalg.norm(num - val) < 1e-4)
        return val

    # collision definition is similar to Double Integrator, car is an "ellipsis"
    def h(self, x_i, x_j):
        """ Collision constraint for x_i, anx x_j agent, h <= 0"""
        # car distance larger than 1.2 normalized
        if (self.config.USE_CPP):
            return self.cpp.h(x_i, x_j)
        val = -((x_i[0] - x_j[0]) / 1.0)**2 - (
            x_i[1] - x_j[1])**2 + self.collision_radius**2
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

    def testAnimation(self):
        u_ref = np.zeros((self.T, self.N, self.m))
        u_ref[:, 0, 1] = radians(20)
        x_ref = self.rollout(self.x0, u_ref)
        full_x_ref = np.vstack([self.x0[np.newaxis, :, :], x_ref])
        self._animation(u_ref, full_x_ref)
        # self._visualize(u_ref,full_x_ref)
        # plt.show()


if __name__ == "__main__":
    # np.random.seed(0)
    _config = ResidualGameConfig(USE_CPP=False)
    main = CarMergeKinematicBicycle(_config, car_count=5, T=20)
    main.setup()
    u_ref, full_x_ref, has_converged = main.solve()
    main.final()
    print(
        f'u_ref mean {np.mean(u_ref.flatten())} std {np.std(u_ref.flatten())}')
    # main.testAnimation()
