import os
import numpy as np
from time import time
from PIL import Image
from scipy import interpolate
import scipy.sparse # sparse matrix operations
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from math import cos,sin,pi,atan2,radians,degrees,tan,atan
from scipy.interpolate import splprep, splev,CubicSpline,interp1d

from util import *
from TimeUtil import TimeUtil
from src.build.car_merge_kinematic_bicycle import CarMergeKinematicBicycle as cpp_CarMergeKinematicBicycle
from ResidualGame import ResidualGame
from track.Skidpad import Skidpad

# example: Car drifting (1/2 car)
# uses dynamic bicycle model, defined on Frenet frame
class CarDrift(ResidualGame):
    USE_CPP = False
    FORCE_PYTHON_SOLVER = False
    def __init__(self):
        super().__init__()

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
        self.dt = dt = 0.2

        # dimension of x and u for single agent
        self.n = 8
        self.m = 2

        # bounds for visualization
        self.visual_x_lim = [-10,10]
        self.visual_y_lim = [-10,10]

        # initial state,
        #self.x0 = np.array([[0,0,radians(10),1,0.2,0.1, radians(10),10]])
        self.x0 = np.array([[0,0,0,1,0,0,atan(10),0]])
        
        self.print_debug_enable()

        self.track = Skidpad()

    def setup(self):
        return

    def _visualize(self,U,X=None):
        if (X is None):
            X = np.vstack([self.x0[np.newaxis,:,:],self.rollout(self.x0,U)])
        fig, ax = plt.subplots()

        # draw raceline
        ss = np.linspace(0,self.track.raceline_len_m,1000)
        rr = np.array(splev(ss,self.track.raceline_s))
        ax.plot(rr[0],rr[1])
        A = np.array([[0,-1],[1,0]]) 

        # draw car trajectory
        for i in range(self.N):
            ss = X[:,i,0]
            nn = X[:,i,1]
            rr = np.array(splev(ss,self.track.raceline_s))
            dr = np.array(splev(ss,self.track.raceline_s,der=1))
            normal_dir = A @ dr/np.linalg.norm(dr,axis=0)
            rr = normal_dir * nn + rr
            ax.plot(rr[0], rr[1],'*-')
            breakpoint()

        ax.set_aspect('equal', adjustable='box')
        return fig

    def _animation(self,U,X=None,gif_prefix=''):
        return
        ''' build a gif animation'''
        if X is None:
            X = np.vstack([self.x0[np.newaxis,:,:],self.rollout(self.x0,U)])
        car_pos_vec = []
        car_angle_vec = []
        box_vec = []
        circle_vec = []
        color_vec = ['red','green','blue','black']
        color_vec = [color_vec[i%len(color_vec)] for i in range(self.N)]
        # prepare smoothed animation
        for i,color in zip(range(self.N),color_vec):
            # interpolate for smooth graphics
            #tt = np.linspace(0,self.T*self.dt,50)
            tt = np.linspace(0,self.dt*self.T,self.T+1)
            # for plt.Rectangle, we offset position so this corresponds to top left corner
            # also flip x axis
            xx = X[:,i,0] - 1.0
            yy = -(X[:,i,1]) - 0.5
            angle = X[:,i,3]
            xx_fun = interpolate.interp1d(tt,xx)
            yy_fun = interpolate.interp1d(tt,yy)
            angle_fun = interpolate.interp1d(tt,angle)

            pos_vec = np.vstack([yy_fun(tt),xx_fun(tt)]).T
            angle_vec = angle_fun(tt)/np.pi*180.0
            car_angle_vec.append(angle_vec)
            car_pos_vec.append(pos_vec)
            box_vec.append(plt.Rectangle(pos_vec[0], 1, 2,angle=angle_vec[0], color=color))
            circle_vec.append(plt.Circle(pos_vec[0]+np.array([0.5,1.0]), radius=(7**0.5)/2,  color=color, fill=False))

        fig, ax = plt.subplots()
        ax.set_xlim(*self.visual_x_lim)
        ax.set_ylim(*self.visual_y_lim)
        def update(frame):
            for i in range(self.N):
                box_vec[i].set_xy(car_pos_vec[i][frame])
                box_vec[i].set_angle(car_angle_vec[i][frame])
                circle_vec[i].set_center(car_pos_vec[i][frame]+np.array([0.5,1.0]))
            return box_vec
        # Add the boxes to the plot
        for box in box_vec:
            ax.add_patch(box)
        for circ in circle_vec:
            ax.add_patch(circ)
        # lane boundary lines
        ax.vlines(x=-self.track_width,ymin=self.visual_y_lim[0],ymax=self.visual_y_lim[1])
        ax.vlines(x=self.track_width, ymin=self.visual_y_lim[0],ymax=self.visual_y_lim[1])
        # dotted line
        for i in np.linspace(self.visual_y_lim[0], self.visual_y_lim[1], 20):
            ax.vlines(x=0, ymin=i,ymax=i+1)

        ax.set_aspect('equal', adjustable='box')

        # Create the animation
        anim = FuncAnimation(fig, update, frames=len(car_pos_vec[0]), blit=True)
        gif_filename = self.resolveLogname(logPrefix=gif_prefix)
        anim.save(gif_filename, writer='pillow')
        plt.show()


    ''' --------  math functions and their derivatives ------ '''
    # TODO
    def J(self,x_k,u_k_i,i):
        '''
        step cost for an agent, given x,u
        x_k.shape (N*n) x_k_i = [x,y,vx,vy]
        u_k_i.shape (m) u_k_i = [ax, ay]
        i: agent id
        '''
        #return (x[2] - 2.0)**2 + (x[1] - self.target_y[i])**2 + 1e-2*x[3]**2 + 1e-2*u.T @ np.eye(self.m) @ u
        if (self.USE_CPP):
            return self.cpp.J(x_k,u_k_i,i)
        val = (x_k[i]-self.J_x_ref_fun(i)).T @ self.J_Qr @ (x_k[i]-self.J_x_ref_fun(i)) + x_k[i].T @ self.J_Q @ x_k[i] + u_k_i.T @ self.J_R @ u_k_i
        if (self.CPP_DEBUG):
            alt = self.cpp.J(x_k,u_k_i,i)
            if (np.linalg.norm(alt-val)>1e-4):
                breakpoint()
        return val

    # dJi dxi
    def dJi_dxi(self,x_k,u_k_i,i):
        if (self.USE_CPP):
            return self.cpp.dJi_dxi(x_k,u_k_i,i)
        val = 2* (x_k[i]-self.J_x_ref_fun(i)).T @ self.J_Qr + 2*x_k[i].T @ self.J_Q
        if (self.CPP_DEBUG):
            alt = self.cpp.dJi_dxi(x_k,u_k_i,i)
            if (np.linalg.norm(alt-val)>1e-4):
                breakpoint()
        return val

    # dJi dxj
    def dJi_dxj(self,x_k,u_k_i,i,j):
        if (self.USE_CPP):
            return self.cpp.dJi_dxj(x_k,u_k_i,i,j)
        val = 0
        if (self.CPP_DEBUG):
            alt = self.cpp.dJi_dxj(x_k,u_k_i,i,j)
            if (np.linalg.norm(alt-val)>1e-4):
                breakpoint()
        return val

    def dJi_du(self,x_k,u_k_i,i):
        if (self.USE_CPP):
            return self.cpp.dJi_du(x_k,u_k_i,i)
        val = 2* u_k_i.T @ self.J_R
        if (self.CPP_DEBUG):
            alt = self.cpp.dJi_du(x_k,u_k_i,i)
            if (np.linalg.norm(alt-val)>1e-4):
                breakpoint()
        return val

    # dJ^i / dxi dxi
    def dJi_dxi_dxi(self,x_k,u,i):
        if (self.USE_CPP):
            return self.cpp.dJi_dxi_dxi(x_k,u,i)
        val = 2*self.J_Qr + 2*self.J_Q
        if (self.CPP_DEBUG):
            alt = self.cpp.dJi_dxi_dxi(x_k,u,i)
            if (np.linalg.norm(alt-val)>1e-4):
                breakpoint()
        return val

    # dJi / dxi dxj
    def dJi_dxi_dxj(self, x_k, u_k_i, i, j):
        return 0
    def dJi_dxj_dxj(self, x_k, u_k_i, i, j):
        return 0
    def dJi_dudu(self, x_k, u_k_i, i):
        return 2*self.J_R

    # this problem has homogeneous agents, so [i] is irrelevant
    def f(self,x,u,i):
        s,n,mu,vx,vy,r,theta,Br = x
        dsteer,dB = u
        k = lambda s: splev(s,self.track.curvature)[0].item()
        m = 1.0; Iz = 1.0; lf = 1.0; lr = 1.0;
        Tmax = 1.0;

        Fry = Tmax * sin(Br)* (1 if vy>0 else -1)
        Frx = Tmax * cos(Br)*0 # FIXME
        # NOTE different from ref paper (opposite sign)
        Bf = -atan( (vy+lf*r)/vx ) + theta
        # TODO use pacejka F = A sin(B atan(C beta) )
        Ffy = Bf
        ds = (vx*cos(mu) - vy*sin(mu))/(1-n*k(s))
        dn = vx*sin(mu) + vy*cos(mu)
        dmu = r - k(s)*ds
        dvx = 1/m*(Frx - Fry*sin(theta) + m*vy*r)
        dvy = 1/m*(Fry + Ffy*cos(theta) - m*vx*r)
        dr = 1/Iz*(Ffy*cos(theta)*lf - Fry * lr)
        dx = np.array([ds, dn, dmu, dvx, dvy, dr, dsteer, dB])
        return x+dx*self.dt

    # TODO
    def df_dx(self,x,u,i):
        beta = atan(tan(u[1])*0.5)
        A = np.array([[0,0,cos(x[3]+beta), -x[2]*sin(x[3]+beta)],
            [0,0, sin(x[3]+beta), x[2]*cos(x[3]+beta)],
            [0,0,0,0],
            [0,0,sin(beta)/1.0,0]])
        val = np.eye(4) + A*self.dt
        if (self.DEBUG):
            num = jacobianNumerical(lambda xx:self.f(xx,u,i), x,dim=self.n)
            assert (np.linalg.norm(num-val)<1e-4)
        return val

    # TODO
    def df_du(self,x,u,i):
        beta = atan(tan(u[1])*0.5)
        dbeta_dst = 0.5/(  ((tan(u[1])*0.5)**2+1) * cos(u[1])**2 )
        B = np.array([[0,-x[2]*sin(x[3]+beta)*dbeta_dst],
            [0,x[2]*cos(x[3]+beta)*dbeta_dst],
            [1,0],
            [0,x[2]/1.0*cos(beta)*dbeta_dst]])
        val = B*self.dt
        if (self.DEBUG):
            num = jacobianNumerical(lambda uu:self.f(x,uu,i), u,dim=self.n)
            assert (np.linalg.norm(num-val)<1e-4)
        return val

    # no collision in this problem
    def h(self, x_i, x_j):
        return -1

    def dh_dxi(self,x_i,x_j):
        return np.zeros(self.n)
    def dh_dxj(self,x_i,x_j):
        return np.zeros(self.n)
    def dh_dxi_dxi(self,x_i,x_j):
        return np.zeros((self.n,self.n))
    def dh_dxj_dxi(self,x_i,x_j):
        return np.zeros((self.n,self.n))
    def dh_dxi_dxj(self,x_i,x_j):
        return np.zeros((self.n,self.n))
    def dh_dxj_dxj(self,x_i,x_j):
        return np.zeros((self.n,self.n))

    def testAnimation(self):
        u_ref = np.zeros((self.T,self.N,self.m))
        x_ref = self.rollout(self.x0,u_ref)
        full_x_ref = np.vstack([self.x0[np.newaxis,:,:],x_ref])
        #self._animation(u_ref,full_x_ref)
        self._visualize(u_ref,full_x_ref)
        plt.show()



if __name__=="__main__":
    main = CarDrift()
    #main.setup()
    #main.solve(save_gif=False,visualize=True,animate=True)
    #main.final()
    main.testAnimation()

