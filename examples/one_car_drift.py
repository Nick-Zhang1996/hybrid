import numpy as np
from time import time
from PIL import Image
from scipy import interpolate
import scipy.sparse  # sparse matrix operations
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from math import cos, sin, pi, atan2, radians, degrees, tan, atan
from scipy.interpolate import splprep, splev, CubicSpline, interp1d
from scipy.optimize import fsolve
import matplotlib.image as mpimg
from scipy.ndimage import rotate

from utilities.util import *
from utilities.time_util import TimeUtil
from src.build.car_merge_kinematic_bicycle import CarMergeKinematicBicycle as cpp_CarMergeKinematicBicycle
from residual_game import ResidualGame, ResidualGameConfig
from track.Skidpad import Skidpad

from utilities.symbolic_dynamics import SymbolicDynamics
import sympy


# example: Car drifting (1/2 car)
# uses dynamic bicycle model, defined on Frenet frame
class OneCarDrift(ResidualGame):

    def __init__(self, config: ResidualGameConfig):
        super().__init__(config)

        # u_i = [ds, db] time derivative of steering angle, and rear tire slip angle
        # x_i = [s,n,mu,vx,vy,r,theta,beta_r] ref:
        # collision constraint: None
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
        self.N = 1
        self.T = 100
        self.dt = dt = 0.05

        # dimension of x and u for single agent
        self.n = 8
        self.m = 2

        # dynamics parameters
        self.mass = 1.0
        self.Iz = 1.0
        self.lf = 1.0
        self.lr = 1.0
        self.Tmax = 0.174

        # initial state,
        #self.x0 = np.array([[0,0,radians(10),1,0.2,0.1, radians(10),10]])
        vx = 1.0
        vy = -0.38
        r = 0.1333
        theta = -radians(10.45)
        Br = radians(3.82)
        self.guess = np.zeros((self.T, self.N, self.m))
        self.guess[:, 0, 0] = -radians(0)

        self.mu_ref = radians(17)
        self.vx_ref = vx
        self.x0 = np.array([[0, 0, self.mu_ref, vx, vy, r, theta, Br]])

        self.control_cost = 1e-2
        self.n_cost = 0.3
        self.vx_cost = 0.1

        self.print_debug_enable()

        self.track = Skidpad()
        img_dir = os.path.join(BASEDIR, 'resources/porsche_orange.png')
        self.car_img = mpimg.imread(img_dir)
        self.car_scale = 0.004 / 2

        # bounds for visualization
        self.visual_x_lim = [0, 25]
        self.visual_y_lim = [0, 25]

        #self.testAnimation(self.guess)

    def setup(self):
        # find an appropriate equilibrium point
        data = self.findSaddlePoint(plot=False)
        #saddle_point_vec.append( (vx,vy,r, theta, Br, ds, k_s, mu) )
        k = lambda s: splev(s, self.track.curvature)[0].item()
        k_s = k(0)
        best_idx_vec = np.argsort(np.abs(data[:, 6] - k_s))[:5]
        for idx in best_idx_vec:
            vx = data[idx, 0]
            vy = data[idx, 1]
            r = data[idx, 2]
            theta = data[idx, 3]
            Br = data[idx, 4]
            ds = data[idx, 5]
            k_s = data[idx, 6]
            mu = data[idx, 7]
            print(
                f' candidate saddle: vx = {vx:.2f}, vy = {vy:.2f}, r = {r/np.pi*180}deg/s, theta = {degrees(theta):.2f}deg, Br = {degrees(Br):.2f}deg, radius = {1/k_s:.2f}m, mu = {degrees(mu):.2f}deg'
            )
        # select the eq point with max slip angle
        idx = best_idx_vec[np.argmax(data[best_idx_vec, 7])]
        vx = data[idx, 0]
        vy = data[idx, 1]
        r = data[idx, 2]
        theta = data[idx, 3]
        Br = data[idx, 4]
        ds = data[idx, 5]
        k_s = data[idx, 6]
        mu = data[idx, 7]
        print(
            f' selected saddle: vx = {vx:.2f}, vy = {vy:.2f}, r = {r/np.pi*180}deg/s, theta = {degrees(theta):.2f}deg, Br = {degrees(Br):.2f}deg, radius = {1/k_s:.2f}m, mu = {degrees(mu):.2f}deg'
        )

        self.mu_ref = mu
        self.vx_ref = vx
        self.x0 = np.array([[0, 0, self.mu_ref, vx, vy, r, theta, Br]])
        return

    def getCartesianFromFrenet(self, states):
        A = np.array([[0, -1], [1, 0]])
        ss, nn, mu, vx, vy, r, theta, Br = states
        rr = np.array(splev(ss, self.track.raceline_s))
        dr = np.array(splev(ss, self.track.raceline_s, der=1))
        normal_dir = A @ dr / np.linalg.norm(dr, axis=0)
        heading = np.arctan2(dr[1], dr[0]) + mu
        r = normal_dir * nn + rr
        # x,y, cartesian heading
        #breakpoint()
        return (r[0], r[1], heading)

    def _visualize(self, u, x=None, snapshots=5):
        car_scale = self.car_scale
        if (x is None):
            x = np.vstack(
                [self.x0[np.newaxis, :, :],
                 self.rollout(self.x0, u)])
        fig, ax = plt.subplots()

        for index in range(0, len(x), len(x) // snapshots):
            pose = self.getCartesianFromFrenet(x[index, 0])
            rotated_car_img = np.clip(
                rotate(self.car_img, degrees(pose[2]), reshape=True), 0.0, 1.0)
            L, W, _ = rotated_car_img.shape
            ax.imshow(rotated_car_img,
                      extent=[
                          pose[0] - W * car_scale, pose[0] + W * car_scale,
                          pose[1] - L * car_scale, pose[1] + L * car_scale
                      ])
        '''
        # draw car initial and final pose
        pose = self.getCartesianFromFrenet(X[0,0])
        # plot initial pose
        rotated_car_img = np.clip(rotate(self.car_img,degrees(pose[2]),reshape=True), 0.0, 1.0)
        L,W,_ = rotated_car_img.shape
        ax.imshow(rotated_car_img, extent=[pose[0]-W*car_scale, pose[0]+W*car_scale, pose[1]-L*car_scale, pose[1]+L*car_scale])

        # plot final pose
        pose = self.getCartesianFromFrenet(X[-1,0])
        rotated_car_img = np.clip(rotate(self.car_img,degrees(pose[2]),reshape=True), 0.0, 1.0)
        L,W,_ = rotated_car_img.shape
        ax.imshow(rotated_car_img, extent=[pose[0]-W*car_scale, pose[0]+W*car_scale, pose[1]-L*car_scale, pose[1]+L*car_scale])
        '''

        # draw raceline
        ss = np.linspace(0, self.track.raceline_len_m, 1000)
        rr = np.array(splev(ss, self.track.raceline_s))
        ax.plot(rr[0], rr[1])
        A = np.array([[0, -1], [1, 0]])

        # draw car trajectory
        for i in range(self.N):
            ss = x[:, i, 0]
            nn = x[:, i, 1]
            rr = np.array(splev(ss, self.track.raceline_s))
            dr = np.array(splev(ss, self.track.raceline_s, der=1))
            normal_dir = A @ dr / np.linalg.norm(dr, axis=0)
            rr = normal_dir * nn + rr
            ax.plot(rr[0], rr[1], '*-')

        ax.set_aspect('equal', adjustable='box')
        return fig

    def _animation(self, U, X=None, gif_prefix=''):
        return  # FIXME
        ''' build a gif animation'''
        car_scale = self.car_scale
        if X is None:
            X = np.vstack(
                [self.x0[np.newaxis, :, :],
                 self.rollout(self.x0, U)])
        fig, ax = plt.subplots()

        # draw raceline
        ss = np.linspace(0, self.track.raceline_len_m, 1000)
        rr = np.array(splev(ss, self.track.raceline_s))
        ax.plot(rr[0], rr[1])
        A = np.array([[0, -1], [1, 0]])

        # draw car trajectory
        for i in range(self.N):
            ss = X[:, i, 0]
            nn = X[:, i, 1]
            rr = np.array(splev(ss, self.track.raceline_s))
            dr = np.array(splev(ss, self.track.raceline_s, der=1))
            normal_dir = A @ dr / np.linalg.norm(dr, axis=0)
            rr = normal_dir * nn + rr
            ax.plot(rr[0], rr[1], '-')

        # draw car sprite
        car_pose_vec = []
        for states in X:
            pose = self.getCartesianFromFrenet(states[0])
            car_pose_vec.append(pose)

        # plot initial pose
        rotated_car_img = np.clip(
            rotate(self.car_img, degrees(car_pose_vec[0][2]), reshape=True),
            0.0, 1.0)
        L, W, _ = rotated_car_img.shape
        im = ax.imshow(rotated_car_img,
                       extent=[
                           car_pose_vec[0][0] - W * car_scale,
                           car_pose_vec[0][0] + W * car_scale,
                           car_pose_vec[0][1] - L * car_scale,
                           car_pose_vec[0][1] + L * car_scale
                       ])

        def update(frame):
            rotated_car_img = np.clip(
                rotate(self.car_img,
                       degrees(car_pose_vec[frame][2]),
                       reshape=True), 0.0, 1.0)
            L, W, _ = rotated_car_img.shape
            im.set_data(rotated_car_img)
            im.set_extent((car_pose_vec[frame][0] - W * car_scale,
                           car_pose_vec[frame][0] + W * car_scale,
                           car_pose_vec[frame][1] - L * car_scale,
                           car_pose_vec[frame][1] + L * car_scale))
            return [im]

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(*self.visual_x_lim)
        ax.set_ylim(*self.visual_y_lim)

        # Create the animation
        anim = FuncAnimation(fig,
                             update,
                             frames=len(car_pose_vec),
                             blit=True,
                             interval=10)
        gif_filename = self.resolveLogname(logPrefix=gif_prefix)
        anim.save(gif_filename, writer='pillow')
        plt.show()

    ''' --------  math functions and their derivatives ------ '''

    # TODO
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
        s, n, mu, vx, vy, r, theta, Br = x_k[i]
        dsteer, dB = u_k_i
        val = (mu - self.mu_ref)**2 + self.vx_cost * (
            vx - self.vx_ref)**2 + self.n_cost * n**2 + self.control_cost * (
                dsteer**2 + dB**2)
        if (self.config.CPP_DEBUG):
            alt = self.cpp.J(x_k, u_k_i, i)
            if (np.linalg.norm(alt - val) > 1e-4):
                breakpoint()
        return val

    # dJi dxi
    def dJi_dxi(self, x_k, u_k_i, i):
        if (self.config.USE_CPP):
            return self.cpp.dJi_dxi(x_k, u_k_i, i)
        x0, x1, x2, x3, x4, x5, x6, x7 = x_k[i]
        u0, u1 = u_k_i
        val = np.array([[
            0, 2 * self.n_cost * x1, -2 * self.mu_ref + 2 * x2,
            self.vx_cost * (-2 * self.vx_ref + 2 * x3), 0, 0, 0, 0
        ]])
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
        x0, x1, x2, x3, x4, x5, x6, x7 = x_k[i]
        u0, u1 = u_k_i
        val = np.array([[self.control_cost * u0, self.control_cost * u1]])

        if (self.config.CPP_DEBUG):
            alt = self.cpp.dJi_du(x_k, u_k_i, i)
            if (np.linalg.norm(alt - val) > 1e-4):
                breakpoint()
        return val

    # dJ^i / dxi dxi
    def dJi_dxi_dxi(self, x_k, u, i):
        if (self.config.USE_CPP):
            return self.cpp.dJi_dxi_dxi(x_k, u, i)
        val = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 2 * self.n_cost, 0, 0, 0, 0, 0, 0],
                        [0, 0, 2, 0, 0, 0, 0, 0],
                        [0, 0, 0, 2 * self.vx_cost, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]])
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
        return np.eye(self.m) * self.control_cost

    # this problem has homogeneous agents, so [i] is irrelevant
    def f(self, x, u, i):
        s, n, mu, vx, vy, r, theta, Br = x
        dsteer, dB = u
        k = lambda s: splev(s, self.track.curvature)[0].item()
        k_s = k(s)
        vy_sign = (1 if vy > 0 else -1)
        m = self.mass
        Iz = self.Iz
        lf = self.lf
        lr = self.lr
        Tmax = self.Tmax

        Fry = Tmax * sin(Br) * -vy_sign
        Frx = Tmax * cos(Br)
        # NOTE different from ref paper (opposite sign)
        Bf = -atan((vy + lf * r) / vx) + theta
        # TODO use pacejka F = A sin(B atan(C beta) )
        Ffy = Bf
        ds = (vx * cos(mu) - vy * sin(mu)) / (1 - n * k_s)
        dn = vx * sin(mu) + vy * cos(mu)
        dmu = r - k_s * ds
        dvx = 1 / m * (Frx - Fry * sin(theta) + m * vy * r)
        dvy = 1 / m * (Fry + Ffy * cos(theta) - m * vx * r)
        dr = 1 / Iz * (Ffy * cos(theta) * lf - Fry * lr)
        dx = np.array([ds, dn, dmu, dvx, dvy, dr, dsteer, dB])
        #print(f'x = {x}')
        #print(f'ds = {ds}, dn = {dn}, dmu = {dmu}, dvx = {dvx}, dvy = {dvy}, dr = {dr}')
        return x + dx * self.dt

    # TODO test
    def df_dx(self, x, u, i):
        m = self.mass
        Iz = self.Iz
        lf = self.lf
        lr = self.lr
        Tmax = self.Tmax
        s, n, mu, vx, vy, r, theta, Br = x
        dsteer, dB = u
        x0, x1, x2, x3, x4, x5, x6, x7 = x
        u0, u1 = u

        k = lambda s: splev(s, self.track.curvature)[0].item()
        k_s = k(x0)
        vy_sign = (1 if vy > 0 else -1)

        dfdx = np.array(
            [[
                0, k_s * (x3 * cos(x2) - x4 * sin(x2)) / (-k_s * x1 + 1)**2,
                (-x3 * sin(x2) - x4 * cos(x2)) / (-k_s * x1 + 1),
                cos(x2) / (-k_s * x1 + 1), -sin(x2) / (-k_s * x1 + 1), 0, 0, 0
            ], [0, 0, x3 * cos(x2) - x4 * sin(x2),
                sin(x2),
                cos(x2), 0, 0, 0],
             [
                 0,
                 -k_s**2 * (x3 * cos(x2) - x4 * sin(x2)) / (-k_s * x1 + 1)**2,
                 -k_s * (-x3 * sin(x2) - x4 * cos(x2)) / (-k_s * x1 + 1),
                 -k_s * cos(x2) / (-k_s * x1 + 1),
                 k_s * sin(x2) / (-k_s * x1 + 1), 1, 0, 0
             ],
             [
                 0, 0, 0, 0, 1.0 * x5, 1.0 * x4,
                 1.0 * vy_sign * sin(x7) * cos(x6),
                 1.0 * vy_sign * sin(x6) * cos(x7) - 1.0 * sin(x7)
             ],
             [
                 0, 0, 0, -1.0 * x5 + 1.0 * (x4 + 1.0 * x5) * cos(x6) /
                 (x3**2 * (1 + (x4 + 1.0 * x5)**2 / x3**2)),
                 -1.0 * cos(x6) / (x3 * (1 + (x4 + 1.0 * x5)**2 / x3**2)),
                 -1.0 * x3 - 1.0 * cos(x6) /
                 (x3 * (1 + (x4 + 1.0 * x5)**2 / x3**2)), -1.0 * (x6 - atan(
                     (x4 + 1.0 * x5) / x3)) * sin(x6) + 1.0 * cos(x6),
                 -1.0 * vy_sign * cos(x7)
             ],
             [
                 0, 0, 0, 1.0 * (x4 + 1.0 * x5) * cos(x6) /
                 (x3**2 * (1 + (x4 + 1.0 * x5)**2 / x3**2)),
                 -1.0 * cos(x6) / (x3 * (1 + (x4 + 1.0 * x5)**2 / x3**2)),
                 -1.0 * cos(x6) / (x3 * (1 + (x4 + 1.0 * x5)**2 / x3**2)),
                 -1.0 * (x6 - atan(
                     (x4 + 1.0 * x5) / x3)) * sin(x6) + 1.0 * cos(x6),
                 1.0 * vy_sign * cos(x7)
             ], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]])

        val = np.eye(self.n) + dfdx * self.dt
        if (self.config.DEBUG):
            num = jacobianNumerical(lambda xx: self.f(xx, u, i), x, dim=self.n)
            assert (np.linalg.norm(num - val) < 1e-4)
        return val

    # TODO test
    def df_du(self, x, u, i):
        B = np.zeros((self.n, self.m))
        B[6, 0] = 1
        B[7, 1] = 1
        val = B * self.dt
        if (self.config.DEBUG):
            num = jacobianNumerical(lambda uu: self.f(x, uu, i), u, dim=self.n)
            assert (np.linalg.norm(num - val) < 1e-4)
        return val

    # no collision in this problem
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

    def testAnimation(self, U=None):
        if (U is None):
            u_ref = np.zeros((self.T, self.N, self.m))
            u_ref[:, 0, 0] = -radians(2)
        else:
            u_ref = U
        x_ref = self.rollout(self.x0, u_ref)
        full_x_ref = np.vstack([self.x0[np.newaxis, :, :], x_ref])
        #self._animation(u_ref,full_x_ref)
        self._visualize(u_ref, full_x_ref)
        plt.show()

    def phasePortrait_vx_r(self):
        m = self.mass
        Iz = self.Iz
        lf = self.lf
        lr = self.lr
        Tmax = self.Tmax
        # setpoints, vy>0
        Br = radians(10)
        theta = radians(20)
        # variables
        vx_range = np.linspace(0.05, 0.4)
        r_range = np.linspace(-1.3, 0.5)
        vx, r = np.meshgrid(vx_range, r_range)

        Fry = Tmax * np.sin(Br) * (-1)
        Frx = Tmax * np.cos(Br)
        vy = np.tan(-(
            (m * vx * r - Fry) / np.cos(theta) - theta)) * vx - lf * r

        # NOTE different from ref paper (opposite sign)
        Bf = -np.arctan((vy + lf * r) / vx) + theta
        # TODO use pacejka F = A np.sin(B atan(C beta) )
        Ffy = Bf
        dvx = 1 / m * (Frx - Fry * np.sin(theta) + m * vy * r)
        dr = 1 / Iz * (Ffy * np.cos(theta) * lf - Fry * lr)

        dvy = 1 / m * (Fry + Ffy * np.cos(theta) - m * vx * r)  # == 0
        print(f' dvy = {np.linalg.norm(dvy)} = 0?')

        fig, ax = plt.subplots()
        ax.plot(0.152, -0.397, 'ro')
        ax.streamplot(vx, r, dvx, dr, color='C0')
        ax.set_xlabel('vx')
        ax.set_ylabel('r')

        def fun(x):
            vx = x[0]
            r = x[1]
            vy = np.tan(-(
                (m * vx * r - Fry) / np.cos(theta) - theta)) * vx - lf * r
            Ffy = Bf = -np.arctan((vy + lf * r) / vx) + theta
            dvx = 1 / m * (Frx - Fry * np.sin(theta) + m * vy * r)
            dr = 1 / Iz * (Ffy * np.cos(theta) * lf - Fry * lr)
            return [dvx, dr]

        #root = fsolve(fun, (1.0,-0.1)) # -0.38, -0.91
        root = fsolve(fun, (1.0, 0.2))  # -0.38, -0.91
        print(f'vx,r = {root}')
        vx = root[0]
        r = root[1]
        vy = np.tan(-(
            (m * vx * r - Fry) / np.cos(theta) - theta)) * vx - lf * r

        Fry = Tmax * np.sin(Br)
        Frx = Tmax * np.cos(Br)
        Ffy = Bf = -np.arctan((vy + lf * r) / vx) + theta
        dvx = 1 / m * (Frx - Fry * np.sin(theta) + m * vy * r)
        dvy = 1 / m * (Fry + Ffy * np.cos(theta) - m * vx * r)
        dr = 1 / Iz * (Ffy * np.cos(theta) * lf - Fry * lr)
        print(dvx, dvy, dr)
        breakpoint()

        print(f'vy={vy}')
        plt.show()

    def phasePortrait_vy_r(self):
        m = self.mass
        Iz = self.Iz
        lf = self.lf
        lr = self.lr
        Tmax = self.Tmax
        # setpoints, vy>0
        # vx = 0.38, vy = 1.14, r = -0.91
        Br = radians(10)
        theta = radians(20)
        vx = 0.38
        # variables
        vy_range = np.linspace(0.8, 1.4)
        r_range = np.linspace(-1.3, 0.5)
        vy, r = np.meshgrid(vy_range, r_range)

        Fry = Tmax * np.sin(Br) * (-1)
        Frx = Tmax * np.cos(Br)

        # NOTE different from ref paper (opposite sign)
        Bf = -np.arctan((vy + lf * r) / vx) + theta
        # TODO use pacejka F = A np.sin(B atan(C beta) )
        Ffy = Bf
        dvx = 1 / m * (Frx - Fry * np.sin(theta) + m * vy * r)
        dr = 1 / Iz * (Ffy * np.cos(theta) * lf - Fry * lr)
        dvy = 1 / m * (Fry + Ffy * np.cos(theta) - m * vx * r)

        fig, ax = plt.subplots()
        ax.plot(1.14, -0.916, 'or')
        ax.streamplot(vy, r, dvy, dr, color='C0')
        ax.set_xlabel('vy')
        ax.set_ylabel('r')

        def fun(x):
            vy = x[0]
            r = x[1]
            vy = np.tan(-(
                (m * vx * r - Fry) / np.cos(theta) - theta)) * vx - lf * r
            Ffy = Bf = -np.arctan((vy + lf * r) / vx) + theta
            dvx = 1 / m * (Frx - Fry * np.sin(theta) + m * vy * r)
            dr = 1 / Iz * (Ffy * np.cos(theta) * lf - Fry * lr)
            dvy = 1 / m * (Fry + Ffy * np.cos(theta) - m * vx * r)
            return [dvy, dr]

        root = fsolve(fun, (1.14, -0.91))
        print(f'vy,r = {root}')
        vy = root[0]
        r = root[1]
        Ffy = Bf = -np.arctan((vy + lf * r) / vx) + theta
        dvx = 1 / m * (Frx - Fry * np.sin(theta) + m * vy * r)
        print(f'dvx = {dvx}')
        plt.show()

        # more testing
        vx = 0.379
        r = -0.916
        vy = 1.14
        Ffy = Bf = -np.arctan((vy + lf * r) / vx) + theta
        dvx = 1 / m * (Frx - Fry * np.sin(theta) + m * vy * r)
        dr = 1 / Iz * (Ffy * np.cos(theta) * lf - Fry * lr)
        dvy = 1 / m * (Fry + Ffy * np.cos(theta) - m * vx * r)
        print(f'dvx = {dvx}, dr = {dr}, dvy = {dvy}')
        return

    def findSaddlePoint(self, plot=True):
        ''' find saddle point given vx,
        the result should satisfy: dmu, dvx, dvy, dr = 0
        vx > 0, vy < 0, r > 0, theta < 0, 0 < Br < pi/2 (ccw drifting)
        '''

        m = self.mass
        Iz = self.Iz
        lf = self.lf
        lr = self.lr
        Tmax = self.Tmax
        #Br = radians(10); theta = -radians(20);
        saddle_point_vec = []
        for theta in np.linspace(-radians(40), radians(0)):
            for Br in np.linspace(radians(0), radians(90)):
                vy_sign = -1
                Fry = Tmax * np.sin(Br) * (-vy_sign)
                Frx = Tmax * np.cos(Br)

                def dvxdr_fun(x):
                    vx = x[0]
                    r = x[1]
                    vy = np.tan(-((m * vx * r - Fry) / np.cos(theta) -
                                  theta)) * vx - lf * r
                    Ffy = Bf = -np.arctan((vy + lf * r) / vx) + theta
                    dvx = 1 / m * (Frx - Fry * np.sin(theta) + m * vy * r)
                    dr = 1 / Iz * (Ffy * np.cos(theta) * lf - Fry * lr)
                    return [dvx, dr]

                root = fsolve(dvxdr_fun, (1.0, 0.2))
                vx = root[0]
                r = root[1]
                vy = np.tan(-(
                    (m * vx * r - Fry) / np.cos(theta) - theta)) * vx - lf * r

                vy_sign = (1 if vy > 0 else -1)
                Fry = Tmax * np.sin(Br) * (-vy_sign)
                Frx = Tmax * np.cos(Br)
                Ffy = Bf = -np.arctan((vy + lf * r) / vx) + theta
                dvx = 1 / m * (Frx - Fry * np.sin(theta) + m * vy * r)
                dvy = 1 / m * (Fry + Ffy * np.cos(theta) - m * vx * r)
                dr = 1 / Iz * (Ffy * np.cos(theta) * lf - Fry * lr)

                if (vx > 0 and vy < 0 and theta < 0 and r > 0 and Br > 0
                        and Br < np.pi - np.arctan(-vy / vx)
                        and np.linalg.norm(dvx) < 1e-3
                        and np.linalg.norm(dvy) < 1e-3 and np.linalg.norm(dr)):
                    # find ds, mu, k_s that suits this saddle point
                    ds = (vx**2 + vy**2)**0.5
                    k_s = r / ds
                    mu = -np.arctan(vy / vx)
                    saddle_point_vec.append(
                        (vx, vy, r, theta, Br, ds, k_s, mu))
                    if (plot):
                        print(
                            f' new saddle point: vx = {vx:.2f}, vy = {vy:.2f}, r = {r/np.pi*180}deg/s, theta = {degrees(theta):.2f}deg, Br = {degrees(Br):.2f}deg, radius = {1/k_s:.2f}m, mu = {degrees(mu):.2f}deg'
                        )
                else:
                    '''
                    # plot for diagnosis
                    print(f' BAD saddle point: vx = {vx:.2f}, vy = {vy:.2f},  r = {r/np.pi*180}deg/s, theta = {degrees(theta):.2f}deg, Br = {degrees(Br):.2f}deg')
                    print(f'\t {dvx, dvy, dr} = 0?')
                    vx_range = np.linspace(0.05,2.0)
                    r_range = np.linspace(-1.3,0.5)
                    vx, r = np.meshgrid(vx_range, r_range)
                    vy = np.tan( - ( (m*vx*r - Fry)/np.cos(theta) - theta ) )*vx - lf*r

                    vy_sign = np.ones_like(vy)
                    vy_sign[vy<0] = -1
                    Fry = Tmax * np.sin(Br) * (-vy_sign)
                    Frx = Tmax * np.cos(Br)
                    Ffy = Bf = -np.arctan( (vy+lf*r)/vx ) + theta
                    dvx = 1/m*(Frx - Fry*np.sin(theta) + m*vy*r)
                    dr = 1/Iz*(Ffy*np.cos(theta)*lf - Fry * lr)
                    fig,ax = plt.subplots()
                    ax.plot(root[0], root[1],'ro')
                    ax.streamplot(vx, r, dvx, dr,color='C0')
                    ax.set_xlabel('vx')
                    ax.set_ylabel('r')
                    plt.show()
                    breakpoint()
                    '''
                    continue

        # plotting for debug
        data = np.array(saddle_point_vec)
        if (plot):
            vx_vec = data[:, 0]
            vy_vec = data[:, 1]
            ds = data[:, 5]
            k_s = data[:, 6]
            fig, ax = plt.subplots()
            ax.plot(ds, 1 / k_s, 'o')
            ax.set_xlabel('speed (m/s)')
            ax.set_ylabel('radius')
            ax.set_xlim([0, 8])
            ax.set_ylim([0, 100])
            plt.show()
            breakpoint()
        return data
        '''
        fig,ax = plt.subplots()
        #ax.plot(valid_theta_vec, valid_r_vec, 'o-',label=f'slip: {slip_angle:.1f}deg')
        #ax.annotate(f'Br={valid_Br_vec[i]:.2f}deg', (valid_theta_vec[i], valid_r_vec[i]))
        ax.set_xlabel('steering/theta (deg)')
        ax.set_ylabel('angular speed (deg/s)')
        ax.set_title(f'vx = {vx}')
        ax.legend()
        plt.show()
        '''

    def buildDynamicsJacobian(self):
        ''' find dfdx, dfdu with symbolic math, note this finds df/dx, not dx+/dx '''
        m = self.mass
        Iz = self.Iz
        lf = self.lf
        lr = self.lr
        Tmax = self.Tmax
        dyn = SymbolicDynamics(self.n, self.m)

        # almost verbatim copy of f(x,u,i)
        s, n, mu, vx, vy, r, theta, Br = dyn.x
        dsteer, dB = dyn.u

        k_s = sympy.symbols(f'k_s')  # NOTE external variable
        #k = lambda s: splev(s,self.track.curvature)[0].item()
        #k_s = k(s)

        vy_sign = sympy.symbols(f'vy_sign')  # NOTE external variable
        #vy_sign = (1 if vy>0 else -1)

        Fry = Tmax * sympy.sin(Br) * -vy_sign
        Frx = Tmax * sympy.cos(Br)
        Bf = -sympy.atan((vy + lf * r) / vx) + theta
        Ffy = Bf
        ds = (vx * sympy.cos(mu) - vy * sympy.sin(mu)) / (1 - n * k_s)
        dn = vx * sympy.sin(mu) + vy * sympy.cos(mu)
        dmu = r - k_s * ds
        dvx = 1 / m * (Frx - Fry * sympy.sin(theta) + m * vy * r)
        dvy = 1 / m * (Fry + Ffy * sympy.cos(theta) - m * vx * r)
        dr = 1 / Iz * (Ffy * sympy.cos(theta) * lf - Fry * lr)
        dyn.f = [ds, dn, dmu, dvx, dvy, dr, dsteer, dB]
        dyn.symDerF()
        print(f'dfdx = {dyn.dfdx}')
        print(f'dfdu = {dyn.dfdu}')

        # almost verbatim copy of L
        mu_ref = sympy.symbols(f'mu_ref')
        #mu_ref = radians(10);
        vx_ref = sympy.symbols(f'vx_ref')
        #vx_ref = 1.0

        dyn.l = (mu - mu_ref)**2 + (
            vx - vx_ref)**2 + n**2 + 1e-2 * (dsteer**2 + dB**2)
        dyn.symDerL()
        print(f'lx = {dyn.lx}')
        print(f'lxx = {dyn.lxx}')

        print(f'lu = {dyn.lu}')
        print(f'luu = {dyn.luu}')
        print(f'lux = {dyn.lux}')
        return


if __name__ == "__main__":
    main = OneCarDrift(ResidualGameConfig())
    #main.findSaddlePoint()
    #main.testAnimation()
    #main.buildDynamicsJacobian()
    #main.phasePortrait_vx_r()
    #main.phasePortrait_vy_r()
    main.setup()
    main.solve()
    main.final()
