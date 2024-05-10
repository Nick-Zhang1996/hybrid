import matplotlib.pyplot as plt
import numpy as np
from util import *
from time import time
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

        # Problem formulation
        # decision variables:
        self.N = car_count
        self.T = 5
        self.track_width = 5
        self.track_length = 20
        self.dt = 0.5

        self.rho = 10.0
        self.rho_b = 1.5

        # dimension of u and x for single agent
        self.m = 2
        self.n = 4

        # backtracking line search param
        self.bc_a = 0.3 #alpha
        self.bc_b = 0.5 #beta

        # initial state, stated in unit of car size
        self.x0 = x0 = np.array([[0,0,1,0],[2,1,1.4,0],[1,-2,1,0.1]])
        self.target_y = [-1,1,2]
        # initial control guess
        U = np.zeros((self.T,self.N,self.m))
        X = self.rollout(x0,U)

    def rollout(self,x0,U):
        U = U.reshape(self.T,self.N,self.m)
        X = np.zeros((self.T+1,self.N,self.n))
        X[0,:,:] = x0.reshape(self.N,self.n)
        # x+ = x + vx*dt + 0.5*ax*dt*dt
        # vx+ = vx + ax*dt
        for i in range(self.N):
            for k in range(1,self.T+1):
                X[k,i] = self.f(X[k-1,i], U[k-1,i])
        return X[1:,:,:]

    def visualize(self,U):
        X = np.vstack([self.x0[np.newaxis,:,:],self.rollout(self.x0,U)])
        plt.vlines(x=-self.track_width/2,ymin=-1,ymax=self.track_length)
        plt.vlines(x=self.track_width/2,ymin=-1,ymax=self.track_length)
        for i in range(self.N):
            xx = X[:,i,0]
            yy = X[:,i,1]
            plt.plot(yy,xx,'*-')
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.show()
        return

    def J(self,x,u,i):
        ''' 
        step cost for an agent, given x,u 
        x.shape (n) x = [x,y,vx,vy]
        u.shape (m) u = [ax, ay]
        i: agent id
        '''
        return (x[2] - 2.0)**2 + (x[1] - self.target_y[i])**2 + x[3]**2 + u.T @ np.eye(self.m) @ u

    def f(self,x,u):
        return np.array([x[0] + x[2]*self.dt + 0.5*self.dt*self.dt*u[0],
        x[1] + x[3]*self.dt + 0.5*self.dt*self.dt*u[1],
        x[2] + self.dt*u[0]/10,
        x[3] + self.dt*u[1]/10])

    def solve(self):
        T = self.T; N = self.N; n = self.n
        u_ref = np.zeros((T,N,self.m))
        x_ref = np.vstack([self.x0[np.newaxis,:,:],self.rollout(self.x0,u_ref)])
        lambda_ref = np.zeros((T,N,self.n))
        # defined for all l_k_i_j, but all values may not be used
        mu_ref = np.zeros((T+1,N,N))
        self.visualize(u_ref)
        for i in range(10):
            x_ref, u_ref, lambda_ref, mu_ref = self.step(x_ref,u_ref,lambda_ref,mu_ref)
            print(f'after iter {i}')
            self.visualize(u_ref)
            breakpoint()

    def h(self, x_i, x_j):
        ''' car distance larger than 1.0 '''
        return -(x_i[0]-x_j[0])**2 + (x_i[1]-x_j[1])**2 + 1.0**2

    def L(self,x_k, u_k_i, x_k1_i, h_k_plus_mask,lamda, mu,k,i):
        # feasibility for h>0
        h_plus = np.sum( [ mu[k,i,j.item()] * ( self.h(x_k[i], x_k[j.item()]) ) for j in np.nonzero(h_k_plus_mask[i])[0] ])
        dynamics = lamda[k,i].T @ ( self.f(x_k[i],u_k_i) - x_k1_i)
        # barrier for h < 0
        h_minus = -1/self.rho*np.sum([np.log(-min(self.h(x_k[i], x_k[j.item()]),-1e-100)) if j != i else 0 for j in np.nonzero(~h_k_plus_mask[i])[0] ])
        return self.J(x_k[i], u_k_i, i) + h_plus + h_minus + dynamics

    def LLi(self,x,u,h_plus_mask,lamda,mu,i):
        return np.sum( [self.L(x[k],u[k,i], x[k+1,i],h_plus_mask[k], lamda, mu,k,i) for k in range(self.T)] )

    def r(self, x, u, lamda, mu):
        try:
            # h(i,i) should not be considered in either h_plus or h_minus
            # we check it in h_minux
            h_plus_mask = np.zeros((self.T,self.N,self.N),dtype=bool)
            for k in range(self.T):
                for i in range(self.N):
                    for j in range(i+1,self.N):
                        h_plus_mask[k,i,j] = h_plus_mask[k,j,i] = self.h(x[k,i],x[k,j]) >= 0

            r = 0
            for i in range(self.N):
                dLL_dx = jacobianNumerical(lambda xx:self.LLi(xx.reshape(x.shape),u,h_plus_mask,lamda,mu,i), x.flatten())
                dLL_du = jacobianNumerical(lambda uu:self.LLi(x,uu.reshape(u.shape),h_plus_mask,lamda,mu,i), u.flatten())
                r += dLL_dx @ dLL_dx.T + dLL_du @ dLL_du.T
                for k in range(self.T):
                    dual_fx = self.f(x[k,i], u[k,i]) - x[k+1,i]
                    dual_h_plus = np.sum( [ mu[k,i,j.item()] * ( self.h(x[k,i], x[k,j.item()]) ) for j in np.nonzero(h_plus_mask[k,i])[0] ])
                    r += np.sum(dual_fx**2) + dual_h_plus * dual_h_plus
        except ValueError:
            breakpoint()
        return r


    def step(self,x_ref,u_ref,lambda_ref,mu_ref):
        t0_step = time()
        N = self.N; T = self.T; n = self.n; m = self.m
        # r0 + Dr*dr = 0
        r0 = self.r(x_ref,u_ref,lambda_ref,mu_ref)
        # y: x(T+1*N*n) ,u(T*N*m), lambda(T,N,n),mu(T+1,N,N)
        print(f'dim y: {((T+1)*N*n) +(T*N*m)+ (N*T*n)+((T+1)*N*N)}')
        y0 = np.hstack([x_ref.flatten(), u_ref.flatten(), lambda_ref.flatten(), mu_ref.flatten()])
        # x,u,lamda,mu = split_y(y)
        split_y = lambda y: (y[:(T+1)*N*n].reshape(T+1,N,n), y[(T+1)*N*n:(T+1)*N*n + T*N*m].reshape(T,N,m), y[(T+1)*N*n + T*N*m:(T+1)*N*n + T*N*m + N*T*n].reshape(T,N,n), y[(T+1)*N*n + T*N*m + N*T*n:].reshape(T+1,N,N))
        #r_y_fun = lambda y: self.r(y[:(T+1)*N*n].reshape(T+1,N,n), y[(T+1)*N*n:(T+1)*N*n + T*N*m].reshape(T,N,m), y[(T+1)*N*n + T*N*m:(T+1)*N*n + T*N*m + N*T*n].reshape(T,N,n), y[(T+1)*N*n + T*N*m + N*T*n:].reshape(T+1,N,N))
        r_y_fun = lambda y: self.r(*split_y(y))
        t0_Dr = time()
        Dr = jacobianNumerical(r_y_fun,y0)
        dt_Dr = time()-t0_Dr
        print(f'Dr numerical jac: {dt_Dr}')
        dy, residuals, rank, s = np.linalg.lstsq(Dr,-r0)
        # Newton direction
        # line search
        dt_step = time()-t0_step
        print(f'step: {dt_step}')

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

        return split_y(y+t*dy)

if __name__=="__main__":
    main = UnstructuredLaneChange(3)
    main.solve()
    #U = np.zeros((main.T,main.N,main.m))
    #U[:,0,0] = 1.0
    #U[:,1,1] = 1.0
    #main.visualize(U)
