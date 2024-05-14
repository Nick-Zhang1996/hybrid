import matplotlib.pyplot as plt
import numpy as np
from util import *
from time import time
from PIL import Image
import os
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
        x_i_k: 1..T, T*N*n
        u_i_k: 0..T-1, T*N*m
        lamda_i_k: 0..T-1 T*N*n
        mu_k_i_j: 1..T T*N*N
        '''

        # Problem formulation
        # decision variables:
        self.N = car_count
        self.T = 4
        self.track_width = 5
        self.track_length = 20
        self.dt = 0.5

        self.rho = 10.0
        self.rho_b = 1.5

        # dimension of u and x for single agent
        self.m = 2
        self.n = 4

        # backtracking line search param
        self.bc_a = 0.1 #alpha
        self.bc_b = 0.5 #beta

        # initial state, stated in unit of car size
        self.x0 = x0 = np.array([[0,-0.9,2.0,0.5],[0,0.4,2.0,0.2],[0,2.0,2.0,-0.3]])
        self.target_y = [-1.0,1.0,1.0]
        self.frame_vec = []


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

    def visualize(self,U,visualize=True,save_gif=False):
        if (not visualize and not save_gif):
            return
        else:
            fig = self._visualize(U)
            if (save_gif):
                fig.canvas.draw()
                frame = Image.frombytes('RGB',
                fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
                self.frame_vec.append(frame)
            if (visualize):
                plt.show()

        return

    def final(self):
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

    def resolveLogname(self,):
        # setup log file
        # log file will record state of the vehicle for later analysis
        logFolder = "./gifs/"
        logPrefix = "lane"
        logSuffix = ".gif"
        no = 1
        while os.path.isfile(logFolder+logPrefix+str(no)+logSuffix):
            no += 1

        log_no = no
        logFilename = logFolder+logPrefix+str(no)+logSuffix
        return logFilename

    def J(self,x,u,i):
        ''' 
        step cost for an agent, given x,u 
        x.shape (n) x = [x,y,vx,vy]
        u.shape (m) u = [ax, ay]
        i: agent id
        '''
        return (x[2] - 2.0)**2 + (x[1] - self.target_y[i])**2 + 1e-2*x[3]**2 + 1e-2*u.T @ np.eye(self.m) @ u

    def f(self,x,u):
        return np.array([x[0] + x[2]*self.dt + 0.5*self.dt*self.dt*u[0],
        x[1] + x[3]*self.dt + 0.5*self.dt*self.dt*u[1],
        x[2] + self.dt*u[0]/10,
        x[3] + self.dt*u[1]/10])

    def solve(self,save_gif=False,visualize=False):
        T = self.T; N = self.N; n = self.n
        u_ref = np.zeros((T,N,self.m))
        # x_1 .. x_T, NOTE the array index is offset from the math notation
        x_ref = self.rollout(self.x0,u_ref)
        lambda_ref = np.zeros((T,N,self.n))
        # defined for all h_k_i_j, but all values may not be used
        mu_ref = np.zeros((T,N,N))
        self.visualize(u_ref)
        for i in range(10):
            x_ref, u_ref, lambda_ref, mu_ref = self.step(x_ref,u_ref,lambda_ref,mu_ref)
            print(f'after iter {i}')
        print(u_ref)
        print(x_ref)
        self.visualize(u_ref,visualize,save_gif)

    def h(self, x_i, x_j):
        ''' car distance larger than 1.0 '''
        return -(x_i[0]-x_j[0])**2 - (x_i[1]-x_j[1])**2 + 1.0**2

    def L(self,x_k, u_k_i, x_k1_i, h_k_plus_mask,lamda_k, mu_k,i):
        # feasibility for h>0
        h_plus = np.sum( [ mu_k[i,j.item()] * ( self.h(x_k[i], x_k[j.item()]) ) for j in np.nonzero(h_k_plus_mask[i])[0] ])
        # barrier for h < 0
        h_minus = -1/self.rho*np.sum([np.log(-min(self.h(x_k[i], x_k[j.item()]),-1e-100)) if j != i else 0 for j in np.nonzero(~h_k_plus_mask[i])[0] ])
        dynamics = lamda_k[i].T @ ( self.f(x_k[i],u_k_i) - x_k1_i)
        return self.J(x_k[i], u_k_i, i) + h_plus + h_minus + dynamics

    def LLi(self,x,u,h_plus_mask,lamda,mu,i):
        T = self.T
        LLi_val = np.sum( [self.L(x[k-1],u[k,i], x[k,i],h_plus_mask[k-1], lamda[k], mu[k-1],i) for k in range(1,T)] )
        # x0 related terms
        LLi_val += self.J(self.x0[i],u[0,i],i) + lamda[0,i].T @ ( self.f(self.x0[i],u[0,i]) - x[0,i])
        # x_T related terms
        LLi_val += self.J(x[T-1,i],np.zeros_like(u[0,i]),i)
        h_plus = np.sum( [ mu[T-1,i,j.item()] * ( self.h(x[T-1,i], x[T-1,j.item()]) ) for j in np.nonzero(h_plus_mask[T-1,i])[0] ])
        h_minus = -1/self.rho*np.sum([np.log(-min(self.h(x[T-1,i], x[T-1,j.item()]),-1e-100)) if j != i else 0 for j in np.nonzero(~h_plus_mask[T-1,i])[0] ])
        LLi_val += h_plus + h_minus
        return LLi_val

    def r(self, x, u, lamda, mu, h_plus_mask):
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
        except ValueError:
            breakpoint()
        return r

    def debug_r_by_category(self,x,u,lamda,mu):
        T = self.T
        try:
            # h(i,i) should not be considered in either h_plus or h_minus
            # we check it in h_minux
            # for x 1-T
            h_plus_mask = np.zeros((self.T,self.N,self.N),dtype=bool)
            for k in range(1,self.T+1):
                for i in range(self.N):
                    for j in range(i+1,self.N):
                        h_plus_mask[k-1,i,j] = h_plus_mask[k-1,j,i] = self.h(x[k-1,i],x[k-1,j]) >= 0
            r_primal = 0; r_dynamics = 0; r_h=0
            for i in range(self.N):
                #print(f'agent {i}')
                dLL_dx = jacobianNumerical(lambda xx:self.LLi(xx.reshape(x.shape),u,h_plus_mask,lamda,mu,i), x.flatten())
                dLL_du = jacobianNumerical(lambda uu:self.LLi(x,uu.reshape(u.shape),h_plus_mask,lamda,mu,i), u.flatten())
                #print(f'DLL_dx {dLL_dx}')
                #print(f'DLL_du {dLL_du}')
                val = self.LLi(x,u,h_plus_mask,lamda,mu,i)
                #print(f'LLi {val}')

                r_primal += dLL_dx @ dLL_dx.T + dLL_du @ dLL_du.T
                # dynamics for f(x0,u0) = x1
                r_dynamics += np.sum((self.f(self.x0[i], u[0,i]) - x[0,i])**2)
                for k in range(1,self.T):
                    dual_fx = np.sum((self.f(x[k-1,i], u[k,i]) - x[k,i])**2)
                    r_dynamics += dual_fx
        except ValueError:
            breakpoint()
        return r_primal, r_dynamics, r_h

    def getHplusMask(self,x):
        # h(i,i) should not be considered in either h_plus or h_minus
        # we check it in h_minux
        # for x 1-T
        h_plus_mask = np.zeros((self.T,self.N,self.N),dtype=bool)
        for k in range(1,self.T+1):
            for i in range(self.N):
                for j in range(i+1,self.N):
                    h_plus_mask[k-1,i,j] = h_plus_mask[k-1,j,i] = self.h(x[k-1,i],x[k-1,j]) >= 0
        return h_plus_mask

    def step(self,x_ref,u_ref,lambda_ref,mu_ref):
        N = self.N; T = self.T; n = self.n; m = self.m
        # r0 + Dr*dr = 0
        h_plus_mask = self.getHplusMask(x_ref)
        r0 = self.r(x_ref,u_ref,lambda_ref,mu_ref,h_plus_mask)
        # y: x(T*N*n) ,u(T*N*m), lambda(T,N,n),mu(T,N,N)
        print(f'dim y: {(T*N*n) +(T*N*m)+ (N*T*n)+(T*N*N)}')
        y0 = np.hstack([x_ref.flatten(), u_ref.flatten(), lambda_ref.flatten(), mu_ref.flatten()])
        # x,u,lamda,mu = split_y(y)
        split_y = lambda y: (y[:T*N*n].reshape(T,N,n), y[T*N*n:T*N*n + T*N*m].reshape(T,N,m), y[T*N*n + T*N*m:T*N*n + T*N*m + N*T*n].reshape(T,N,n), y[T*N*n + T*N*m + N*T*n:].reshape(T,N,N))
        r_y_fun = lambda y: self.r(*split_y(y),h_plus_mask)
        Dr = jacobianNumerical(r_y_fun,y0,dim=r0.shape[0])
        dy, residuals, rank, s = np.linalg.lstsq(Dr,-r0)
        # Newton direction
        # line search

        # backtracking line search
        t = 1.0 # step size
        dy = dy.flatten()
        r0_norm = np.linalg.norm(r0)
        for i in range(5):
            r_t = r_y_fun(y0+t*dy)
            r_t_norm = np.linalg.norm(r_t)
            if (r_t_norm > (1-self.bc_a*t)*r0_norm):
                t *= self.bc_b
            else:
                break

        print(f't={t}')
        print(f'r0_norm {r0_norm} rt_norm {r_t_norm}')
        r_primal, r_dynamics, r_h = self.debug_r_by_category(*split_y(y0))
        print(f'r_0 r_primal={r_primal}, r_dynamics={r_dynamics}, r_h={r_h} ')
        r_primal, r_dynamics, r_h = self.debug_r_by_category(*split_y(y0+t*dy))
        print(f'r_t r_primal={r_primal}, r_dynamics={r_dynamics}, r_h={r_h} ')

        print(f't={t}')

        dx,du,dlamda,dmu = split_y(t*dy)
        print(f'dx norm {np.linalg.norm(dx):.4f}')
        print(f'du norm {np.linalg.norm(du):.4f}')
        print(f'dlamda norm {np.linalg.norm(dlamda):.4f}')
        print(f'dmu norm {np.linalg.norm(dmu):.4f}')

        # dynamics residual
        x, u, lamda, mu = split_y(y0)
        dyn_res = self.getDynamicsResiduals(x,u)
        print(f'old dyn residual {dyn_res:.4f}')
        x, u, lamda, mu = split_y(y0+t*dy)
        dyn_res = self.getDynamicsResiduals(x,u)
        print(f'new dyn residual {dyn_res:.4f}')
        #self.visualize(u_ref+du)
        print(f'x_ref {x_ref}')
        print(f'u_ref {u_ref}')
        return split_y(y0+t*dy)

    def getDynamicsResiduals(self,x, u):
        r = 0
        for i in range(self.N):
            dual_fx = self.f(self.x0[i], u[0,i]) - x[0,i]
            r += np.sum(dual_fx**2)
            for k in range(1,self.T):
                dual_fx = self.f(x[k-1,i], u[k,i]) - x[k,i]
                r += np.sum(dual_fx**2)
        return r

if __name__=="__main__":
    main = UnstructuredLaneChange()
    main.solve(save_gif=True,visualize=True)
    main.final()
    #U = np.zeros((main.T,main.N,main.m))
    #U[:,0,0] = 1.0
    #U[:,1,1] = 1.0
    #main.visualize(U)
