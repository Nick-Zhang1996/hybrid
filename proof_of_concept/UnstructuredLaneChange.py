import os
import numpy as np
from time import time
from PIL import Image
from scipy import interpolate
import scipy.sparse # sparse matrix operations
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

from util import *
from TimeUtil import TimeUtil

DEBUG = False
t = TimeUtil(True)

# example: unstructured lane change
# this version use U as decision variable only
class UnstructuredLaneChange():
    def __init__(self,car_count=3):
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
        self.T = 8
        self.track_width = 5
        self.track_length = 20
        self.dt = dt = 0.25

        self.rho = 10.0
        self.rho_b = 1.5

        # dimension of u and x for single agent
        self.m = 2
        self.n = 4

        # backtracking line search param
        self.bc_a = 0.1 #alpha
        self.bc_b = 0.5 #beta

        # initial state, stated in unit of car size
        self.x0 = np.array([[0,-0.9,1.5,0.5],[0,0.4,2.5,0.2],[0,2.0,1.7,-0.3]])
        self.target_y = [-1.0,1.0,1.0]
        #self.x0 = x0 = np.array([[0,-0.9,1.5,0.5],[0,0.4,2.5,0.2],[0,2.0,1.7,-0.3],[0.3,-0.3,1.3,0.3],[-0.4,0.8,0.4,1.0],[1.0,0.0,0.1,0.3]])
        #self.target_y = [-2.0,-1.0,1.0,1.4,1.7,2.0]
        self.frame_vec = []

        # step cost parameters
        self.J_x_ref_fun = lambda i:np.array([0,self.target_y[i],2.0,0])
        self.J_Qr = np.diag([0,1,1,0])
        self.J_Q = np.diag([0,0,0,1e-2])
        self.J_R = np.eye(self.m)*1e-2

        # dynamics parameters
        self.A = np.eye(self.n)
        self.A[0,2] = dt
        self.A[1,3] = dt
        self.B = np.array([[0.5*dt**2,0],[0,0.5*dt**2],[dt,0],[0,dt]])

        # collision definition
        self.h_Qh = np.diag([-1,-1,0,0])

    def solve(self,save_gif=False,visualize=False,animate=False):
        N = self.N; T = self.T; n = self.n; m = self.m
        # y: x(T*N*n) ,u(T*N*m), lambda(T,N,n),mu(T,N,N)
        print(f'dim y: {(T*N*n) +(T*N*m)+ (N*T*n)+(T*N*N)}')

        u_ref = np.zeros((T,N,self.m))
        # x_1 .. x_T, NOTE the array index is offset from the math notation
        x_ref = self.rollout(self.x0,u_ref)
        lambda_ref = np.zeros((T,N,self.n))
        # defined for all h_k_i_j, but all values may not be used
        mu_ref = np.zeros((T,N,N))
        self.visualize(u_ref,animate=animate,gif_prefix='before')
        t0 = time()
        for i in range(10):
            x_ref, u_ref, lambda_ref, mu_ref, stopping = self.step(x_ref,u_ref,lambda_ref,mu_ref)
            print(f'after iter {i}')
            if stopping:
                break
        t_solve = time()-t0
        print(f'total solve time: {t_solve}')
        #print(u_ref)
        #print(x_ref)
        full_x_ref = np.vstack([self.x0[np.newaxis,:,:],x_ref])
        self.visualize(u_ref,full_x_ref,visualize,save_gif,animate,gif_prefix='after')

    def step(self,x_ref,u_ref,lambda_ref,mu_ref):
        global t
        t.s()
        t.s('setup')
        N = self.N; T = self.T; n = self.n; m = self.m
        # r0 + Dr*dr = 0
        h_plus_mask = self.getHplusMask(x_ref)
        r0 = self.r(x_ref,u_ref,lambda_ref,mu_ref,h_plus_mask)
        y0 = np.hstack([x_ref.flatten(), u_ref.flatten(), lambda_ref.flatten(), mu_ref.flatten()])
        # x,u,lamda,mu = split_y(y)
        split_y = lambda y: (y[:T*N*n].reshape(T,N,n), y[T*N*n:T*N*n + T*N*m].reshape(T,N,m), y[T*N*n + T*N*m:T*N*n + T*N*m + N*T*n].reshape(T,N,n), y[T*N*n + T*N*m + N*T*n:].reshape(T,N,N))
        r_y_fun = lambda y: self.r(*split_y(y),h_plus_mask)

        t.e('setup')
        t0 = time()
        Dr = self.dr_dy( x_ref, u_ref, lambda_ref, mu_ref, h_plus_mask)
        print(f't: Dr analytical {(time()-t0)*1000:.2f}ms')

        if (DEBUG):
            t0 = time()
            Dr_alt = jacobianNumerical(r_y_fun,y0,dim=r0.shape[0])
            print(f't: Dr numerical {time()-t0}')
            print(np.linalg.norm(Dr-Dr_alt))
            assert(np.linalg.norm(Dr-Dr_alt)<1e-4)


        '''
        t.s('lstsq')
        dy, residuals, rank, s = np.linalg.lstsq(Dr,-r0)
        t.e('lstsq')
        '''
        t.s('sparse-lstsq')
        sparse_Dr = scipy.sparse.csc_matrix(Dr, dtype=float)
        dy, istop, itn, normr = scipy.sparse.linalg.lsqr(sparse_Dr,-r0)[:4]
        t.e('sparse-lstsq')
        '''
        # DEBUG
        print(f'sparse solution diff {np.linalg.norm(sparse_dy-dy)}')
        print(f' dy residual {np.linalg.norm(Dr @ dy + r0)}')
        print(f' sparse dy residual {np.linalg.norm(sparse_Dr @ sparse_dy + r0)}')
        '''


        total_entries = Dr.shape[0]*Dr.shape[1]
        nonzero_entries = len(np.nonzero(Dr.flatten())[0])
        print(f' nonzero entries:  {nonzero_entries/total_entries}')
        # Newton direction
        # line search

        t.s('line search')
        # backtracking line search
        step = 1.0 # step size
        dy = dy.flatten()
        r0_norm = np.linalg.norm(r0)
        for i in range(10):
            r_t = r_y_fun(y0+step*dy)
            r_t_norm = np.linalg.norm(r_t)
            if (r_t_norm > (1-self.bc_a*step)*r0_norm):
                step *= self.bc_b
            else:
                break
        t.e('line search')


        print(f'r0_norm {r0_norm} rt_norm {r_t_norm}')

        # dynamics residual
        # TODO check this
        '''
        x, u, lamda, mu = split_y(y0)
        dyn_res = self.getDynamicsResiduals(x,u)
        print(f'old dyn residual {dyn_res:.4f}')
        x, u, lamda, mu = split_y(y0+step*dy)
        dyn_res = self.getDynamicsResiduals(x,u)
        print(f'new dyn residual {dyn_res:.4f}')
        #self.visualize(u_ref+du)
        print(f'x_ref {x_ref}')
        print(f'u_ref {u_ref}')
        '''
        t.e()
        h_plus_violations = 0
        # check residuals
        dim_x = T*N*n
        dim_u = T*N*m
        index = 0
        for i in range(self.N):
            print(f'agent {i}')
            dLL_dx_res = np.linalg.norm(r_t[index:index+dim_x])
            index += dim_x
            dLL_du_res = np.linalg.norm(r_t[index:index+dim_u])
            index += dim_u
            fx_res = np.linalg.norm(r_t[index:index+n*T])
            index += n*T
            h_res = np.linalg.norm(r_t[index:index+np.sum(h_plus_mask[:,i])])
            h_plus_violations += h_res
            index += np.sum(h_plus_mask[:,i])
            print(f'dLL_dx {dLL_dx_res:.2f}, dLL_du {dLL_du_res:.2f}, fx {fx_res:.2f}, h_res {h_res:.2f}, h_plus {np.sum(h_plus_mask[:,i])}')

        if (np.abs(r_t_norm - r0_norm)<5e-4 and h_plus_violations<1e-3):
            stopping = True
        else:
            stopping = False

        return split_y(y0+step*dy) + (stopping,)

    def getDynamicsResiduals(self,x, u):
        r = 0
        for i in range(self.N):
            dual_fx = self.f(self.x0[i], u[0,i]) - x[0,i]
            r += np.sum(dual_fx**2)
            for k in range(1,self.T):
                dual_fx = self.f(x[k-1,i], u[k,i]) - x[k,i]
                r += np.sum(dual_fx**2)
        return r

    def rollout(self,x0,U):
        ''' given x0 and u0..u_T-1 (T*N*m), find x1..xT '''
        U = U.reshape(self.T,self.N,self.m)
        X = np.zeros((self.T+1,self.N,self.n))
        X[0,:,:] = x0.reshape(self.N,self.n)
        # x+ = x + vx*dt + 0.5*ax*dt*dt
        # vx+ = vx + ax*dt
        for i in range(self.N):
            for k in range(1,self.T+1):
                X[k,i] = self.f(X[k-1,i], U[k-1,i])
        return X[1:,:,:]

    def visualize(self,U,X=None,visualize=True,save_gif=False,animate=False,gif_prefix='run'):
        if (visualize or save_gif):
            fig = self._visualize(U)
            if (save_gif):
                fig.canvas.draw()
                frame = Image.frombytes('RGB',
                fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
                self.frame_vec.append(frame)
            if (visualize):
                plt.show()
        if (animate):
            self._animation(U,X,gif_prefix=gif_prefix)

        return

    def final(self):
        t.summary()
        if (len(self.frame_vec)>0):
            gif_filename = self.resolveLogname()
            self.frame_vec[0].save(fp=gif_filename,format='GIF',append_images=self.frame_vec,save_all=True,duration = 200,loop=0)
            print(f'GIf saved to {gif_filename}')

    def _visualize(self,U):
        X = np.vstack([self.x0[np.newaxis,:,:],self.rollout(self.x0,U)])
        fig, ax = plt.subplots()
        ax.vlines(x=-self.track_width/2,ymin=-1,ymax=self.track_length)
        ax.vlines(x=self.track_width/2,ymin=-1,ymax=self.track_length)
        for i in range(self.N):
            xx = X[:,i,0]
            yy = X[:,i,1]
            plt.plot(yy,xx,'*-')
        ax.set_aspect('equal', adjustable='box')
        return fig

    def _animation(self,U,X=None,gif_prefix=''):
        ''' build a gif animation'''
        if X is None:
            X = np.vstack([self.x0[np.newaxis,:,:],self.rollout(self.x0,U)])
        car_pos_vec = []
        box_vec = []
        color_vec = ['red','green','blue','black']
        color_vec = [color_vec[i%len(color_vec)] for i in range(self.N)]
        # prepare smoothed animation
        for i,color in zip(range(self.N),color_vec):
            tt = np.linspace(0,self.dt*self.T,self.T+1)
            xx = X[:,i,0]
            yy = X[:,i,1]
            xx_fun = interpolate.interp1d(tt,xx)
            yy_fun = interpolate.interp1d(tt,yy)

            # interpolate for smooth graphics
            #tt = np.linspace(0,self.T*self.dt,50)
            pos_vec = np.vstack([yy_fun(tt),xx_fun(tt)]).T
            car_pos_vec.append(pos_vec)
            box_vec.append(plt.Rectangle(pos_vec[0], 1, 1, color=color))
            print(pos_vec[0], color)

        fig, ax = plt.subplots()
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2, 30)
        def update(frame):
            for i in range(self.N):
                box_vec[i].set_xy(car_pos_vec[i][frame])
        # Add the boxes to the plot
        for box in box_vec:
            ax.add_patch(box)
        ax.set_aspect('equal', adjustable='box')

        # Create the animation
        anim = FuncAnimation(fig, update, frames=len(car_pos_vec[0]), blit=True)
        gif_filename = self.resolveLogname(logPrefix=gif_prefix)
        anim.save(gif_filename, writer='pillow')
        plt.show()


    def resolveLogname(self,logPrefix='run'):
        # setup log file
        # log file will record state of the vehicle for later analysis
        logFolder = "./gifs/"
        logSuffix = ".gif"
        no = 1
        while os.path.isfile(logFolder+logPrefix+str(no)+logSuffix):
            no += 1

        log_no = no
        logFilename = logFolder+logPrefix+str(no)+logSuffix
        return logFilename

    ''' --------  math functions and their derivatives ------ '''
    def J(self,x,u,i):
        ''' 
        step cost for an agent, given x,u 
        x.shape (n) x = [x,y,vx,vy]
        u.shape (m) u = [ax, ay]
        i: agent id
        '''
        #return (x[2] - 2.0)**2 + (x[1] - self.target_y[i])**2 + 1e-2*x[3]**2 + 1e-2*u.T @ np.eye(self.m) @ u
        return (x-self.J_x_ref_fun(i)).T @ self.J_Qr @ (x-self.J_x_ref_fun(i)) + x.T @ self.J_Q @ x + u.T @ self.J_R @ u

    def dJ_dx(self,x,u,i):
        return  2* (x-self.J_x_ref_fun(i)).T @ self.J_Qr + 2*x.T @ self.J_Q
    def dJ_du(self,x,u,i):
        return  2* u.T @ self.J_R
    def dJ_dxdx(self,x,u,i):
        return  2*self.J_Qr + 2*self.J_Q

    def f(self,x,u):
        '''
        return np.array([x[0] + x[2]*self.dt + 0.5*self.dt*self.dt*u[0]/10,
        x[1] + x[3]*self.dt + 0.5*self.dt*self.dt*u[1]/10,
        x[2] + self.dt*u[0]/10,
        x[3] + self.dt*u[1]/10])
        '''
        return self.A @ x + self.B @ u
    def df_dx(self,x,u):
        return self.A
    def df_du(self,x,u):
        return self.B


    def h(self, x_i, x_j):
        ''' car distance larger than 1.0 '''
        #return (x_i-x_j).T @ self.h_Qh @ (x_i-x_j) + 1.0**2
        # below is faster
        return -(x_i[0]-x_j[0])**2 - (x_i[1]-x_j[1])**2 + 1.0**2

    # TODO rewrite this to be faster
    def dh_dxi(self,x_i,x_j):
        return 2*(x_i-x_j).T @ self.h_Qh
    def dh_dxj(self,x_i,x_j):
        return 2*(x_j-x_i).T @ self.h_Qh
    def dh_dxi_dxi(self,x_i,x_j):
        return 2*self.h_Qh.T
    def dh_dxj_dxi(self,x_i,x_j):
        return -2* self.h_Qh.T
    def dh_dxi_dxj(self,x_i,x_j):
        return -2*self.h_Qh.T
    def dh_dxj_dxj(self,x_i,x_j):
        return 2*self.h_Qh.T

    def L(self,x_k, u_k_i, x_k1_i, h_k_plus_mask,lamda_k, mu_k,i):
        # feasibility for h>0
        h_plus = np.sum( [ mu_k[i,j.item()] * ( self.h(x_k[i], x_k[j.item()]) ) for j in np.nonzero(h_k_plus_mask[i])[0] ],axis=0)
        # barrier for h < 0
        h_minus = -1/self.rho*np.sum([np.log(-min(self.h(x_k[i], x_k[j.item()]),-1e-100)) if j.item() != i else 0 for j in np.nonzero(~h_k_plus_mask[i])[0] ])
        dynamics = lamda_k[i].T @ ( self.f(x_k[i],u_k_i) - x_k1_i)
        return self.J(x_k[i], u_k_i, i) + h_plus + h_minus + dynamics
    def dL_dx_ik(self,x_k, u_k_i, x_k1_i, h_k_plus_mask,lamda_k, mu_k,i):
        # NOTE the behavior of barrier function near boundary may need tuning
        if (DEBUG):
            # dJ_dx -- passed
            num = jacobianNumerical(lambda xx:self.J(xx,u_k_i,i), x_k[i])
            ana = self.dJ_dx(x_k[i], u_k_i, i)
            assert (np.linalg.norm(num-ana) < 1e-4)
            # df_dx -- inconclusive
            num = jacobianNumerical(lambda uu:self.J(x_k[i],uu,i), u_k_i)
            ana = self.dJ_du(x_k[i], u_k_i, i)
            assert (np.linalg.norm(num-ana) < 1e-4)
            # dh_dxi -- inconclusive
            for j in np.nonzero(h_k_plus_mask[i])[0]:
                if i == j:
                    continue
                ana = self.dh_dxi(x_k[i], x_k[j])
                num = jacobianNumerical(lambda xx:self.h(xx,x_k[j]),x_k[i])
                assert (np.linalg.norm(num-ana) < 1e-4)

        val =  self.dJ_dx(x_k[i],u_k_i,i) + lamda_k[i].T @ self.df_dx(x_k[i],u_k_i)
        val += np.sum( [ mu_k[i,j.item()] * ( self.dh_dxi(x_k[i], x_k[j.item()]) ) for j in np.nonzero(h_k_plus_mask[i])[0] ], axis=0)
        val += -1/self.rho*np.sum([min(1/self.h(x_k[i], x_k[j.item()]),1e10) * self.dh_dxi(x_k[i], x_k[j.item()]) * (j.item() != i) for j in np.nonzero(~h_k_plus_mask[i])[0] ],axis=0)
        return val

    def dL_dx_ik1(self,x_k, u_k_i, x_k1_i, h_k_plus_mask,lamda_k, mu_k,i):
        ''' dL/dx_i_k+1 '''
        return -lamda_k[i].T
    def dL_dx_jk(self,x_k, u_k_i, x_k1_i, h_k_plus_mask,lamda_k, mu_k,i,j):
        assert (i!=j)
        return  mu_k[i,j] * ( self.dh_dxj(x_k[i], x_k[j]) ) if h_k_plus_mask[i,j] else \
            -1/self.rho*min(1/self.h(x_k[i], x_k[j]),1e10) * self.dh_dxi(x_k[i], x_k[j])
    def dL_du(self,x_k, u_k_i, x_k1_i, h_k_plus_mask,lamda_k, mu_k,i):
        return self.dJ_du(x_k[i],u_k_i,i) + lamda_k[i].T @ self.df_du(x_k[i], u_k_i)


    def LLi(self,x,u,h_plus_mask,lamda,mu,i):
        T = self.T
        LLi_val = np.sum( [self.L(x[k-1],u[k,i], x[k,i],h_plus_mask[k-1], lamda[k], mu[k-1],i) for k in range(1,T)] ,axis=0)
        # x0 related terms
        LLi_val += self.J(self.x0[i],u[0,i],i) + lamda[0,i].T @ ( self.f(self.x0[i],u[0,i]) - x[0,i])
        # x_T related terms
        LLi_val += self.J(x[T-1,i],np.zeros_like(u[0,i]),i)
        h_plus = np.sum( [ mu[T-1,i,j.item()] * ( self.h(x[T-1,i], x[T-1,j.item()]) ) for j in np.nonzero(h_plus_mask[T-1,i])[0] ],axis=0)
        h_minus = -1/self.rho*np.sum([np.log(-min(self.h(x[T-1,i], x[T-1,j.item()]),-1e-100)) * (j.item() != i) for j in np.nonzero(~h_plus_mask[T-1,i])[0] ],axis=0)
        LLi_val += h_plus + h_minus
        return LLi_val

    def dLLi_dx(self,x,u,h_plus_mask,lamda,mu,i):
        ''' return: 1*dim(x) = 1*(T*N*n) , Note index of x starts with 1'''
        T = self.T; N = self.N; n = self.n; m = self.m
        der = np.zeros(T*N*n)
        submtx_i_k = lambda i,k:der[(k-1)*N*n+i*n:(k-1)*N*n+(i+1)*n]
        # dLLi_dxi
        for k in range(1,T):
            sub = submtx_i_k(i,k)
            sub[:] = self.dL_dx_ik(x[k-1],u[k,i],x[k,i],h_plus_mask[k-1],lamda[k],mu[k-1],i) -lamda[k-1,i].T
            if (DEBUG):
                num = jacobianNumerical(lambda xx:self.L(xx.reshape(N,n), u[k,i], x[k,i], h_plus_mask[k-1],lamda[k], mu[k-1],i),x[k-1].flatten())
                num = num[0,i*n:(i+1)*n] - lamda[k-1,i].T
                assert( np.linalg.norm(num-sub) < 1e-4)

        # dLLi_dxi_T
        sub = submtx_i_k(i,T)
        sub[:] = -lamda[T-1,i].T + self.dJ_dx(x[T-1,i],np.zeros(m),i) \
            + np.sum( [ mu[T-1,i,j.item()] * ( self.dh_dxi(x[T-1,i], x[T-1,j.item()]) ) for j in np.nonzero(h_plus_mask[T-1,i])[0] ],axis=0) \
            -1/self.rho*np.sum([min(1/self.h(x[T-1,i], x[T-1,j.item()]),1e10)*self.dh_dxi(x[T-1,i],x[T-1,j.item()]) *(j.item() != i) for j in np.nonzero(~h_plus_mask[T-1,i])[0] ],axis=0)

        # TODO optimize
        # dLLi_dxj
        for j in range(0,N):
            if i==j:
                continue
            for k in range(1,T+1):
                sub = submtx_i_k(j,k)
                sub[:] =  mu[k-1,i,j] * self.dh_dxj(x[k-1,i], x[k-1,j]) if h_plus_mask[k-1,i,j] else \
                    -1/self.rho*min(1/self.h(x[k-1,i], x[k-1,j]),1e10)*self.dh_dxj(x[k-1,i], x[k-1,j])
        return der.reshape(1,-1)

    # TODO check
    def dLLi_du(self,x,u,h_plus_mask,lamda,mu,i):
        ''' return: 1*dim(u) = 1*(T*N*m) '''
        T = self.T; N = self.N; n = self.n; m = self.m
        der = np.zeros(T*N*m)
        submtx_i_k = lambda i,k:der[k*N*m+i*m:k*N*m+(i+1)*m]
        # dLLi_dui_0
        sub = submtx_i_k(i,0)
        sub[:] = self.dJ_du(self.x0[i],u[0,i],i) + lamda[0,i].T @ self.df_du(self.x0[i],u[0,i])
        # dLLi_dui_k
        for k in range(1,T):
            sub = submtx_i_k(i,k)
            # dL_du
            sub[:] = self.dJ_du(x[k-1,i],u[k,i],i) + lamda[k,i].T @ self.df_du(x[k-1,i],u[k,i])
        return der


    def r_old(self, x, u, lamda, mu, h_plus_mask):
        T = self.T
        try:
            r = np.zeros(0)
            for i in range(self.N):
                dLL_dx = jacobianNumerical(lambda xx:self.LLi(xx.reshape(x.shape),u,h_plus_mask,lamda,mu,i), x.flatten())
                dLL_du = jacobianNumerical(lambda uu:self.LLi(x,uu.reshape(u.shape),h_plus_mask,lamda,mu,i), u.flatten())
                r = np.hstack([r,dLL_dx.flatten(), dLL_du.flatten()])
                # dynamics for f(x0,u0) = x1
                r = np.hstack([r,self.f(self.x0[i], u[0,i]) - x[0,i]])
                for k in range(1,self.T):
                    r = np.hstack([r,self.f(x[k-1,i], u[k,i]) - x[k,i]]) # dual for dynamics
                    r = np.hstack([r]+[ self.h(x[k-1,i], x[k-1,j.item()]) for j in np.nonzero(h_plus_mask[k-1,i])[0] ])
                # h(x_T_i, x_T_j)
                r = np.hstack([r]+[ self.h(x[T-1,i], x[T-1,j.item()]) for j in np.nonzero(h_plus_mask[T-1,i])[0] ])
        except ValueError as e:
            raise e
            breakpoint()
        return r

    def r(self, x, u, lamda, mu, h_plus_mask):
        T = self.T
        try:
            r = np.zeros(0)
            for i in range(self.N):
                dLL_dx = self.dLLi_dx(x,u,h_plus_mask,lamda,mu,i)
                dLL_du = self.dLLi_du(x,u,h_plus_mask,lamda,mu,i)
                if (DEBUG):
                    dLL_du_num = jacobianNumerical(lambda uu:self.LLi(x,uu.reshape(u.shape),h_plus_mask,lamda,mu,i), u.flatten())
                    assert(np.linalg.norm(dLL_du-dLL_du_num)<1e-4)
                    dLL_dx_num = jacobianNumerical(lambda xx:self.LLi(xx.reshape(x.shape),u,h_plus_mask,lamda,mu,i), x.flatten())
                    assert(np.linalg.norm(dLL_dx-dLL_dx_num)<1e-4)
                r = np.hstack([r,dLL_dx.flatten(), dLL_du.flatten()])
                # dynamics for f(x0,u0) = x1
                r = np.hstack([r,self.f(self.x0[i], u[0,i]) - x[0,i]])
                for k in range(1,self.T):
                    r = np.hstack([r,self.f(x[k-1,i], u[k,i]) - x[k,i]]) # dual for dynamics
                for k in range(1,self.T):
                    r = np.hstack([r]+[ self.h(x[k-1,i], x[k-1,j.item()]) for j in np.nonzero(h_plus_mask[k-1,i])[0] ])
                # h(x_T_i, x_T_j)
                r = np.hstack([r]+[ self.h(x[T-1,i], x[T-1,j.item()]) for j in np.nonzero(h_plus_mask[T-1,i])[0] ])
        except ValueError as e:
            raise e
            breakpoint()
        return r

    def Bh(self,x_i,x_j):
        return -1/self.rho * np.log(-min(self.h(x_i, x_j),-1e-100))

    def dBh_dxi(self,x_i,x_j):
        # B(h) = -rho^-1 log(-h)
        #dB(h)/dx = -rho^-1 h^-1 dhdx
        val = -1/(self.rho*self.h(x_i, x_j))* self.dh_dxi(x_i,x_j)
        if (DEBUG):
            val_num = jacobianNumerical(lambda xx:self.Bh(xx.reshape(x_i.shape),x_j), x_i.flatten())
            assert(np.linalg.norm(val-val_num)<1e-4)
        return val

    def dBh_dxj(self,x_i,x_j):
        # B(h) = -rho^-1 log(-h)
        #dB(h)/dx = -rho^-1 h^-1 dhdx
        val = -1/(self.rho*self.h(x_i, x_j))* self.dh_dxj(x_i,x_j)
        if (DEBUG):
            val_num = jacobianNumerical(lambda xx:self.Bh(x_i,xx.reshape(x_j.shape)), x_j.flatten())
            assert(np.linalg.norm(val-val_num)<1e-4)
        return val

    def dBh_dxi_dxi(self,x_i,x_j):
        h = self.h(x_i,x_j)
        dhdxi = self.dh_dxi(x_i,x_j).reshape(1,self.n)
        val = 1/(self.rho * h) * (-self.dh_dxi_dxi(x_i,x_j) + 1/h * dhdxi.T @ dhdxi )
        if (DEBUG):
            val_num = jacobianNumerical(lambda xx:self.dBh_dxi(xx.reshape(x_i.shape),x_j), x_i.flatten(),dim=self.n)
            assert(np.linalg.norm(val-val_num)<1e-4)
        return val

    def dBh_dxi_dxj(self,x_i,x_j):
        h = self.h(x_i,x_j)
        dhdxi = self.dh_dxi(x_i,x_j).reshape(1,self.n)
        dhdxj = self.dh_dxj(x_i,x_j).reshape(1,self.n)
        val = 1/(self.rho * h) * (-self.dh_dxi_dxj(x_i,x_j) + 1/h * dhdxi.T @ dhdxj )
        if (DEBUG):
            val_num = jacobianNumerical(lambda xx:self.dBh_dxi(x_i, xx.reshape(x_j.shape)), x_j.flatten(),dim=self.n)
            assert(np.linalg.norm(val-val_num)<1e-4)
        return val

    def dBh_dxj_dxj(self,x_i,x_j):
        h = self.h(x_i,x_j)
        dhdxi = self.dh_dxi(x_i,x_j).reshape(1,self.n)
        dhdxj = self.dh_dxj(x_i,x_j).reshape(1,self.n)
        val = 1/(self.rho * h) * (-self.dh_dxj_dxj(x_i,x_j) + 1/h * dhdxj.T @ dhdxj )
        if (DEBUG):
            val_num = jacobianNumerical(lambda xx:self.dBh_dxj(x_i, xx.reshape(x_j.shape)), x_j.flatten(),dim=self.n)
            assert(np.linalg.norm(val-val_num)<1e-4)
        return val


    def dLLi_dxdx(self,x,u,h_plus_mask,lamda,mu,i):
        T = self.T; N = self.N; n = self.n; m = self.m; dim_x = T*N*n


        dLL_dxdx = np.zeros((dim_x,dim_x))
        submtx = lambda k,i,j: dLL_dxdx[(k-1)*N*n+i*n:(k-1)*N*n+(i+1)*n,(k-1)*N*n+j*n:(k-1)*N*n+(j+1)*n]
        submtx_num = lambda k,i,j: dLL_dxdx_num[(k-1)*N*n+i*n:(k-1)*N*n+(i+1)*n,(k-1)*N*n+j*n:(k-1)*N*n+(j+1)*n]
        for k in range(1,T):
            #dLLi_dxki_dxki
            mtx = submtx(k,i,i)
            val1 = self.dJ_dxdx(x[k-1,i],u[k,i],i)
            val2 = np.sum( [ mu[k-1,i,j.item()] * ( self.dh_dxi_dxi(x[k-1,i], x[k-1,j.item()]) ) for j in np.nonzero(h_plus_mask[k-1,i])[0] ],axis=0)
            val3 = np.sum([self.dBh_dxi_dxi(x[k-1,i], x[k-1,j.item()]) * (j.item() != i) for j in np.nonzero(~h_plus_mask[k-1,i])[0] ],axis=0)
            mtx[:,:] = val1 + val2 + val3

        #dLLi_dxki_dxki, k=T, u_T is undefined, use 0 to penalize J(x) only
        mtx = submtx(T,i,i)
        mtx[:,:] = self.dJ_dxdx(x[T-1,i],np.zeros(m),i) \
                + np.sum( [ mu[T-1,i,j.item()] * ( self.dh_dxi_dxi(x[T-1,i], x[T-1,j.item()]) ) for j in np.nonzero(h_plus_mask[T-1,i])[0] ],axis=0) \
                + np.sum([self.dBh_dxi_dxi(x[T-1,i], x[T-1,j.item()]) * (j.item() != i) for j in np.nonzero(~h_plus_mask[T-1,i])[0] ],axis=0)
        for k in range(1,T+1):
            for j in range(N):
                if i==j:
                    continue
                if (j in np.nonzero(h_plus_mask[k-1,i])[0]):
                    #dLLi_dxki_dxkj
                    val = mu[k-1,i,j] * self.dh_dxi_dxj(x[k-1,i], x[k-1,j])
                    mtx = submtx(k,i,j)
                    mtx[:,:] = val
                    mtx = submtx(k,j,i)
                    mtx[:,:] = val.T
                    #dLLi_dxkj_dxkj
                    mtx = submtx(k,j,j)
                    mtx[:,:] = mu[k-1,i,j] * self.dh_dxj_dxj(x[k-1,i], x[k-1,j])
                else:
                    #dLLi_dxki_dxkj
                    val = self.dBh_dxi_dxj(x[k-1,i], x[k-1,j])
                    mtx = submtx(k,i,j)
                    mtx[:,:] = val
                    mtx = submtx(k,j,i)
                    mtx[:,:] = val.T
                    #dLLi_dxkj_dxkj
                    mtx = submtx(k,j,j)
                    mtx[:,:] = self.dBh_dxj_dxj(x[k-1,i], x[k-1,j])


        if (DEBUG):
            '''
            for k in range(1,T+1):
                for ii in range(N):
                    for j in range(N):
                        mtx_num = submtx_num(k,ii,j)
                        mtx = submtx(k,ii,j)
                        if(not np.linalg.norm(mtx-mtx_num)<1e-4):
                            print(f'i = {i} ii={ii},j={j},k={k}')
                            #breakpoint()
            '''
            dLL_dxdx_num = jacobianNumerical(lambda xx:self.dLLi_dx(xx.reshape(x.shape),u,h_plus_mask,lamda,mu,i), x.flatten(),dim=dim_x)
            print(f'dLL_dxdx err {np.linalg.norm(dLL_dxdx_num - dLL_dxdx)}')
            assert(np.linalg.norm(dLL_dxdx_num - dLL_dxdx)<1e-4)
        return dLL_dxdx

    '''
    # this derivative is identically zero
    def dLLi_dudx(self,x,u,h_plus_mask,lamda,mu,i):
        T = self.T; N = self.N; n = self.n; m = self.m; dim_x = T*N*n; dim_u = T*N*m
        return jacobianNumerical(lambda xx:self.dLLi_du(xx.reshape(x.shape),u,h_plus_mask,lamda,mu,i), x.flatten(),dim=dim_u)
    '''

    def dF_dx(self,x,u,i,k):
        ''' F(x,u) = f(x_k_i,u_k_i)-x_k+1_i, find dF_dx, note x here is of dim(T*N*n) '''
        T = self.T; N = self.N; n = self.n; m = self.m; dim_x = T*N*n
        dFdx = np.zeros((n,dim_x))
        dFdx[:,(k-1)*N*n+i*n:(k-1)*N*n+(i+1)*n] = self.df_dx(x[k-1,i],u[k,i])
        dFdx[:,k*N*n+i*n:k*N*n+(i+1)*n] = -np.eye(n)
        return dFdx

    def dF0_dx(self,x,u,i):
        ''' F0(x,u) = f(x_0_i,u_0_i)-x_1_i, find dF_dx note x here is of dim(T*N*n)
            A specialization for dF_dx when k=0, since we need x0
        '''
        T = self.T; N = self.N; n = self.n; m = self.m; dim_x = T*N*n
        dFdx = np.zeros((n,dim_x))
        dFdx[:,i*n:(i+1)*n] = -np.eye(n)
        return dFdx

    def dh_dx(self,x,k,i,j):
        ''' find d h(x_i,x_j)/ d x note x here is of dim(T*N*n) '''
        T = self.T; N = self.N; n = self.n; m = self.m
        dim_x = T*N*n
        dhdx = np.zeros((1,dim_x))
        dhdx[:,(k-1)*N*n+i*n:(k-1)*N*n+(i+1)*n] = self.dh_dxi(x[k-1,i],x[k-1,j])
        dhdx[:,(k-1)*N*n+j*n:(k-1)*N*n+(j+1)*n] = self.dh_dxj(x[k-1,i],x[k-1,j])
        return dhdx

    def dr_dx(self, x, u, lamda, mu, h_plus_mask):
        ''' return: dim(r)*dim(x) '''
        T = self.T; N = self.N; n = self.n; m = self.m
        dim_x = N*T*n; dim_u = N*T*m
        dim_r = N*(dim_x+dim_u+T*n)+np.sum(h_plus_mask)
        drdx = np.zeros((dim_r,dim_x))
        index = 0
        for i in range(self.N):
            dLL_dxdx = self.dLLi_dxdx(x,u,h_plus_mask,lamda,mu,i)
            # this item is identically zero
            #dLL_dudx = np.zeros((dim_u,dim_x))
            dF0dx = self.dF0_dx(x,u,i)
            # dynamics for f(x0,u0) = x1
            drdx[index:index+dim_x,:] = dLL_dxdx
            index += dim_x + dim_u
            drdx[index:index+n,:] = dF0dx
            for k in range(1,self.T):
                dFdx = self.dF_dx(x,u,i,k)
                drdx[index+k*n:index+(k+1)*n,:] = dFdx
            index += n*T
            for k in range(1,self.T+1):
                indices = np.nonzero(h_plus_mask[k-1,i])[0]
                if (len(indices) == 0):
                    continue
                dhdx = np.vstack([ self.dh_dx(x,k,i,j.item()) for j in indices ])
                drdx[index:index+dhdx.shape[0],:] = dhdx
            index += np.sum(h_plus_mask[:,i])

        if (DEBUG):
            drdx_num = jacobianNumerical(lambda xx:self.r(xx.reshape(x.shape),u,lamda,mu,h_plus_mask), x.flatten(),dim=dim_r)
            print(f'drdx err {np.linalg.norm(drdx-drdx_num)}')
            assert(np.linalg.norm(drdx-drdx_num)<1e-4)
        return drdx

    def dr_dx_old(self, x, u, lamda, mu, h_plus_mask):
        ''' return: dim(r)*dim(x) '''
        T = self.T; N = self.N; n = self.n; m = self.m
        dim_x = N*T*n; dim_u = N*T*m
        dim_r = N*(dim_x+dim_u+T*n)+np.sum(h_plus_mask)
        #TODO change to fill-in style
        drdx = np.zeros((0,dim_x))
        for i in range(self.N):
            dLL_dxdx = self.dLLi_dxdx(x,u,h_plus_mask,lamda,mu,i)
            # this item is identically zero
            dLL_dudx = np.zeros((dim_u,dim_x))
            dF0dx = self.dF0_dx(x,u,i)
            # dynamics for f(x0,u0) = x1
            drdx = np.vstack([drdx,dLL_dxdx, dLL_dudx, dF0dx])
            for k in range(1,self.T):
                dFdx = self.dF_dx(x,u,i,k)
                drdx = np.vstack([drdx,dFdx])
            for k in range(1,self.T):
                drdx = np.vstack([drdx]+[ self.dh_dx(x,k,i,j.item()) for j in np.nonzero(h_plus_mask[k-1,i])[0] ])
            # h(x_T_i, x_T_j)
            drdx = np.vstack([drdx]+[ self.dh_dx(x,T,i,j.item()) for j in np.nonzero(h_plus_mask[T-1,i])[0] ])

        if (DEBUG):
            drdx_num = jacobianNumerical(lambda xx:self.r(xx.reshape(x.shape),u,lamda,mu,h_plus_mask), x.flatten(),dim=dim_r)
            print(f'drdx err {np.linalg.norm(drdx-drdx_num)}')
            assert(np.linalg.norm(drdx-drdx_num)<1e-4)
        return drdx

    def dr_du(self, x, u, lamda, mu, h_plus_mask):
        ''' return: dim(r)*dim(u) '''
        T = self.T; N = self.N; n = self.n; m = self.m
        dim_x = T*N*n; dim_u = T*N*m
        dim_r = N*(dim_x+dim_u+T*n)+np.sum(h_plus_mask)

        drdu = np.zeros((dim_r,dim_u))
        index = 0
        for i in range(self.N):
            index += dim_x
            for k in range(self.T):
                dLL_duik_duik = 2*self.J_R
                drdu[index+k*N*m+i*m:index+k*N*m+(i+1)*m, k*N*m+i*m:k*N*m+(i+1)*m] = dLL_duik_duik
            index += dim_u
            k = 0
            drdu[index+k*n:index+(k+1)*n, k*N*m+i*m:k*N*m+(i+1)*m] = self.df_du(self.x0[i],u[k,i])
            for k in range(1,self.T):
                drdu[index+k*n:index+(k+1)*n, k*N*m+i*m:k*N*m+(i+1)*m] = self.df_du(x[k-1,i],u[k,i])
            index += n*T + np.sum(h_plus_mask[:,i]) # skip  f(x,u)-x+,  h(x,x)

        if (DEBUG):
            drdu_num = jacobianNumerical(lambda uu:self.r(x,uu.reshape(u.shape),lamda,mu,h_plus_mask), u.flatten(),dim=dim_r)
            assert(np.linalg.norm(drdu-drdu_num)<1e-4)
            '''
            if (np.linalg.norm(drdu-drdu_num)>1e-4):
                for i in range(N):
                    for k in range(0,T):
                        val = drdu[:,k*N*m + i*m:k*N*m+i*m+m]
                        val_num = drdu_num[:,k*N*m + i*m:k*N*m+i*m+m]
                        if (np.linalg.norm(val - val_num)>1e-4):
                            print(f'k={k}, i={i},{np.nonzero(val-val_num)}')
                breakpoint()
            '''
        return drdu

    def dr_dlamda(self, x, u, lamda, mu, h_plus_mask):
        ''' return: dim(r)*dim(lamda) '''
        T = self.T; N = self.N; n = self.n; m = self.m
        dim_x = T*N*n; dim_u = T*N*m ; dim_lamda = T*N*n
        dim_r = N*(dim_x+dim_u+T*n)+np.sum(h_plus_mask)
        dr_dlamda = np.zeros((dim_r,dim_lamda))
        index = 0
        for i in range(N):
            for k in range(1,T):
                # dLLi_dxki_dlamda_ki
                dr_dlamda[index+(k-1)*N*n+i*n:index+(k-1)*N*n+(i+1)*n,k*N*n+i*n:k*N*n+(i+1)*n] = self.df_dx(x[k-1,i],u[k,i]).T
                # dLLi_dxki_dlamda_k-1,i
                dr_dlamda[index+(k-1)*N*n+i*n:index+(k-1)*N*n+(i+1)*n,(k-1)*N*n+i*n:(k-1)*N*n+(i+1)*n] = -np.eye(n)
            k = T
            dr_dlamda[index+(k-1)*N*n+i*n:index+(k-1)*N*n+(i+1)*n,(k-1)*N*n+i*n:(k-1)*N*n+(i+1)*n] = -np.eye(n)

            index += dim_x # skip dLL_dx, index now points at dLLi_du
            for k in range(0,T):
                dr_dlamda[index+k*N*m+i*m:index+k*N*m+(i+1)*m,k*N*n+i*n:k*N*n+(i+1)*n] = self.df_du(x[k-1,i],u[k,i]).T

            index += dim_u + n*T + np.sum(h_plus_mask[:,i]) # skip  dLL_du, f(x,u)-x+,  h(x,x)

        if (DEBUG):
            dr_dlamda_num = jacobianNumerical(lambda ll:self.r(x,u,ll.reshape(lamda.shape),mu,h_plus_mask), lamda.flatten(),dim=dim_r)
            print(f'dr_dlamda err {np.linalg.norm(dr_dlamda-dr_dlamda_num)}')
            diff = dr_dlamda_num - dr_dlamda
            assert(np.linalg.norm(dr_dlamda-dr_dlamda_num)<1e-4)
        return dr_dlamda

    def dLLi_dx_dmu(self,x,u,h_plus_mask,lamda,mu,i):
        ''' return: dim: dim_x*dim_mu '''
        T = self.T; N = self.N; n = self.n; m = self.m
        dim_x = T*N*n; dim_u = T*N*m
        dim_mu = T*N*N
        dLL_dx_dmu = np.zeros((dim_x,dim_mu))
        for k in range(1,T+1):
            for j in np.nonzero(h_plus_mask[k-1,i])[0]:
                dLLi_dxki_dmuijk = self.dh_dxi(x[k-1,i],x[k-1,j])
                dLLi_dxkj_dmuijk = self.dh_dxj(x[k-1,i],x[k-1,j])
                dLL_dx_dmu[(k-1)*N*n+i*n:(k-1)*N*n+(i+1)*n,(k-1)*N*N+i*N+j] = dLLi_dxki_dmuijk
                dLL_dx_dmu[(k-1)*N*n+j*n:(k-1)*N*n+(j+1)*n,(k-1)*N*N+i*N+j] = dLLi_dxkj_dmuijk
        return dLL_dx_dmu

    # TODO this is untested
    def dr_dmu(self, x, u, lamda, mu, h_plus_mask):
        ''' return: dim(r)*dim(mu) '''
        T = self.T; N = self.N; n = self.n; m = self.m
        dim_x = T*N*n; dim_u = T*N*m
        dim_r = N*(dim_x+dim_u+T*n)+np.sum(h_plus_mask)
        dim_mu = T*N*N

        dr_dmu = np.zeros((dim_r,dim_mu))
        index = 0
        for i in range(self.N):
            # dmu i,j,k
            dLL_dx_dmu = self.dLLi_dx_dmu(x,u,h_plus_mask,lamda,mu,i)
            dr_dmu[index:index+dim_x,:] = dLL_dx_dmu
            if (DEBUG):
                dLL_dx_dmu_num = jacobianNumerical(lambda mm:self.dLLi_dx(x,u,h_plus_mask,lamda,mm.reshape(mu.shape),i), mu.flatten(),dim=dim_x)
                assert(np.linalg.norm(dLL_dx_dmu-dLL_dx_dmu_num)<1e-4)

            index += dim_x + dim_u + n*T + np.sum(h_plus_mask[:,i])

        if (DEBUG):
            dr_dmu_num = jacobianNumerical(lambda mm:self.r(x,u,lamda,mm.reshape(mu.shape),h_plus_mask), mu.flatten(),dim=dim_r)
            print(f'drdx err {np.linalg.norm(dr_dmu-dr_dmu_num)}')
            assert(np.linalg.norm(dr_dmu-dr_dmu_num)<1e-4)
        return dr_dmu

    def dr_dy(self, x, u, lamda, mu, h_plus_mask):
        t.s('drdx')
        drdx = self.dr_dx(x, u, lamda, mu, h_plus_mask)
        t.e('drdx')
        t.s('drdu')
        drdu = self.dr_du(x, u, lamda, mu, h_plus_mask)
        t.e('drdu')
        t.s('drdlamda')
        drdlamda = self.dr_dlamda(x, u, lamda, mu, h_plus_mask)
        t.e('drdlamda')
        t.s('drdmu')
        drdmu = self.dr_dmu(x, u, lamda, mu, h_plus_mask)
        t.e('drdmu')
        Dr = np.hstack([drdx,drdu,drdlamda,drdmu])
        return Dr

    def getHplusMask(self,x):
        # h(i,i) should not be considered in either h_plus or h_minus
        # we check it in h_minux
        # for x 1-T, NOTE index start from 1
        h_plus_mask = np.zeros((self.T,self.N,self.N),dtype=bool)
        for k in range(1,self.T+1):
            for i in range(self.N):
                for j in range(i+1,self.N):
                    h_plus_mask[k-1,i,j] = h_plus_mask[k-1,j,i] = self.h(x[k-1,i],x[k-1,j]) >= 0
        return h_plus_mask


if __name__=="__main__":
    main = UnstructuredLaneChange()
    main.solve(save_gif=True,visualize=True)
    main.final()
    t.summary()
