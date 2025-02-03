import numpy as np
from util import *
from TimeUtil import TimeUtil

from ResidualGame import ResidualGame
from SteinGame import SteinGame

# example for paper Stein Variational Game, Low Dimension Examples 1)
class BimodalConvergence(SteinGame):
    USE_CPP = False
    def __init__(self):
        super().__init__()

        '''
        x_i_k: 1..T, T*N*n  NOTE starts from 1
        u_i_k: 0..T-1, T*N*m
        lamda_i_k: 0..T-1 T*N*n
        mu_k_i_j: 1..T T*N*N NOTE starts from 1
        '''
        self.print_debug_enable()
        self.N = 2
        self.T = 10
        self.dt = dt = 0.1
        self.dynamics_residual_weight = 1.0

        self.tolerance = 5e-4
        self.iterations = 50 # 30

        # x^i: [position, velocity (1D)]
        # u^i: [acceleration]
        self.n = 2
        self.m = 1

        # bounds for visualization
        self.visual_x_lim = [-2.5,2.5]
        self.visual_y_lim = [-2.5,2.5]

        # step cost parameters
        # NOTE this lambda fun needs to be implemented in c++
        self.J_Q = np.diag([1,0])
        self.J_R = np.eye(self.m)*0.1
        self.A = np.array([[1,dt],[0,1]])
        self.B = np.array([[0.5*dt*dt],[dt]])

        # initial guess for u
        self.guess = np.zeros((self.T,self.N,self.m))
    def setup(self):
        # subclass responsible for loading cpp/eigen module
        if (self.USE_CPP or self.CPP_DEBUG):
            self.print_error('cpp implementation not available')

    def _visualize(self,U,X=None):
        if (X is None):
            X = np.vstack([self.x0[np.newaxis,:,:],self.rollout(self.x0,U)])
        fig, ax = plt.subplots()
        # target points, p1, p2
        ax.plot([-1],[1], 'o')
        ax.plot([1],[-1], 'o')

        # plot agent 0,1's position as x,y coordinate
        xx = X[:,0,0]
        yy = X[:,1,0]
        plt.plot(xx,yy,'*-')
        ax.set_aspect('equal', adjustable='box')
        return fig

    def _animation(self,U,X=None,gif_prefix=''):
        return None
    ''' --------  math functions and their derivatives ------ '''

    # step cost
    def J(self,x_k,u_k_i,i):
        '''
        step cost for an agent, given x,u
        x_k.shape (N*n) x_k_i = [r,v] : pos, vel
        u_k_i.shape (m) u_k_i = [a] : acc
        i: agent id
        '''
        val = u_k_i.T @ self.J_R @ u_k_i
        return val

    def dJi_dxi(self,x_k,u_k_i,i):
        return np.zeros((1,self.n))
    def dJi_dxj(self,x_k,u_k_i,i,j):
        return np.zeros((1,self.n))
    def dJi_dxi_dxi(self,x_k,u_k_i,i):
        return np.zeros((self.n,self.n))
    def dJi_dxi_dxj(self,x_k,u_k_i,i,j):
        return np.zeros((self.n,self.n))
    def dJi_dxj_dxj(self,x_k,u_k_i,i,j):
        return np.zeros((self.n,self.n))
    def dJi_du(self,x_k,u_k_i,i):
        val = 2* u_k_i.T @ self.J_R
        return val
    def dJi_dudu(self, x_k, u_k_i, i):
        return 2*self.J_R

    # terminal cost
    # TODO verify these derivatives
    def Jfi(self, x_T, i):
        return    self.sigmoid( (x_T[0] - self.p1).T @ self.J_Q @ (x_T[0] - self.p1)  \
                +  (x_T[1] - self.p2).T @ self.J_Q @ (x_T[1] - self.p2) ) \
                + self.sigmoid( (x_T[0] - self.p2).T @ self.J_Q @ (x_T[0] - self.p2)  \
                +  (x_T[1] - self.p1).T @ self.J_Q @ (x_T[1] - self.p1) )
    def dJfi_dxi(self, x_T, i):
        sigval1 = self.sigmoid( (x_T[0] - self.p1).T @ self.J_Q @ (x_T[0] - self.p1)  \
                +  (x_T[1] - self.p2).T @ self.J_Q @ (x_T[1] - self.p2) )
        sigval2 = self.sigmoid( (x_T[0] - self.p2).T @ self.J_Q @ (x_T[0] - self.p2)  \
                +  (x_T[1] - self.p1).T @ self.J_Q @ (x_T[1] - self.p1) )
        if (i == 0):
            return sigval1 * (1 - sigval1) * 2 * (x_T[i] - self.p1).T @ self.J_Q \
                 + sigval2 * (1 - sigval2) * 2 * (x_T[i] - self.p2).T @ self.J_Q
        else:
            return sigval2 * (1 - sigval2) * 2 * (x_T[i] - self.p1).T @ self.J_Q \
                 + sigval1 * (1 - sigval1) * 2 * (x_T[i] - self.p2).T @ self.J_Q

        return self.dJi_dxi(x_T,np.zeros(self.m),i)
    def dJfi_dxj(self, x_T, i,j):
        return self.dJfi_dxi(x_T,j)

    def dJfi_dxi_dxi(self, x_T, i):
        Q = self.J_Q
        sig1 = self.sigmoid( (x_T[0] - self.p1).T @ self.J_Q @ (x_T[0] - self.p1)  \
                +  (x_T[1] - self.p2).T @ self.J_Q @ (x_T[1] - self.p2) )
        sig2 = self.sigmoid( (x_T[0] - self.p2).T @ self.J_Q @ (x_T[0] - self.p2)  \
                +  (x_T[1] - self.p1).T @ self.J_Q @ (x_T[1] - self.p1) )
        if (i==0):
            # dsig1 dx0
            dsig1 = sig1 * (1 - sig1) * 2 * Q @ (x_T[0] - self.p1)
            # dsig2 dx0
            dsig2 = sig2 * (1 - sig2) * 2 * Q @ (x_T[0] - self.p2)
            retval = dsig1 * (1-sig1) * 2 * Q @ (x_T[0] - self.p1) \
                    + sig1 * (-dsig1) * 2 * Q @ (x_T[0] - self.p1) \
                    + sig1 * (1-sig1) * 2 * Q
            retval += dsig2 * (1-sig2) * 2 * Q @ (x_T[0] - self.p2) \
                    + sig2 * (-dsig2) * 2 * Q @ (x_T[0] - self.p2) \
                    + sig2 * (1-sig2) * 2 * Q
        else:
            # dsig1 dx1
            dsig1 = sig1 * (1 - sig1) * 2 * Q @ (x_T[0] - self.p2)
            # dsig2 dx1
            dsig2 = sig2 * (1 - sig2) * 2 * Q @ (x_T[0] - self.p1)

            retval = dsig1 * (1-sig1) * 2 * Q @ (x_T[0] - self.p2) \
                    + sig1 * (-dsig1) * 2 * Q @ (x_T[0] - self.p2) \
                    + sig1 * (1-sig1) * 2 * Q
            retval += dsig2 * (1-sig2) * 2 * Q @ (x_T[0] - self.p1) \
                    + sig2 * (-dsig2) * 2 * Q @ (x_T[0] - self.p1) \
                    + sig2 * (1-sig2) * 2 * Q

    def dJfi_dxi_dxj(self, x_T, i,j):
        Q = self.J_Q
        sig1 = self.sigmoid( (x_T[0] - self.p1).T @ self.J_Q @ (x_T[0] - self.p1)  \
                +  (x_T[1] - self.p2).T @ self.J_Q @ (x_T[1] - self.p2) )
        sig2 = self.sigmoid( (x_T[0] - self.p2).T @ self.J_Q @ (x_T[0] - self.p2)  \
                +  (x_T[1] - self.p1).T @ self.J_Q @ (x_T[1] - self.p1) )
        if (i==0):
            # dsig1 dx1
            dsig1 = sig1 * (1 - sig1) * 2 * Q @ (x_T[0] - self.p2)
            # dsig2 dx1
            dsig2 = sig2 * (1 - sig2) * 2 * Q @ (x_T[0] - self.p1)
            retval = dsig1 * (1-sig1) * 2 * Q @ (x_T[0] - self.p1) \
                    + sig1 * (-dsig1) * 2 * Q @ (x_T[0] - self.p1) \
                    + dsig2 * (1-sig2) * 2 * Q @ (x_T[0] - self.p2) \
                    + sig2 * (-dsig2) * 2 * Q @ (x_T[0] - self.p2)
        else:
            # dsig1 dx0
            dsig1 = sig1 * (1 - sig1) * 2 * Q @ (x_T[0] - self.p1)
            # dsig2 dx0
            dsig2 = sig2 * (1 - sig2) * 2 * Q @ (x_T[0] - self.p2)

            retval = dsig1 * (1-sig1) * 2 * Q @ (x_T[0] - self.p2) \
                    + sig1 * (-dsig1) * 2 * Q @ (x_T[0] - self.p2) \
                    + dsig2 * (1-sig2) * 2 * Q @ (x_T[0] - self.p1) \
                    + sig2 * (-dsig2) * 2 * Q @ (x_T[0] - self.p1)

    def dJfi_dxj_dxj(self, x_T, i,j):
        return self.dJfi_dxi_dxi(x_T, j)

    def sigmoid(self, val):
        return 1/(1+np.exp(-val))

    # --- dynamics and related derivatives ---
    # this problem has homogeneous agents, so [i] is irrelevant
    def f(self,x,u,i):
        return self.A @ x + self.B @ u
    def df_dx(self,x,u,i):
        return self.A
    def df_du(self,x,u,i):
        return self.B

    def h(self, x_i, x_j):
        return -1
    def dh_dxi(self,x_i,x_j):
        return 0
    def dh_dxj(self,x_i,x_j):
        return 0
    def dh_dxi_dxi(self,x_i,x_j):
        return 0
    def dh_dxj_dxi(self,x_i,x_j):
        return 0
    def dh_dxi_dxj(self,x_i,x_j):
        return 0
    def dh_dxj_dxj(self,x_i,x_j):
        return 0

    def testAnimation(self):
        u_ref = np.zeros((self.T,self.N,self.m))
        u_ref[:,0,:] = 1
        x_ref = self.rollout(self.x0,u_ref)
        full_x_ref = np.vstack([self.x0[np.newaxis,:,:],x_ref])
        self._visualize(u_ref,full_x_ref)
        plt.show()



if __name__=="__main__":
    #np.random.seed(0)
    main = BimodalConvergence()
    main.setup()
    #u_ref, full_x_ref, has_converged = main.solve(save_gif=False,visualize=True,animate=False)
    #main.final()
    #print(f'u_ref mean {np.mean(u_ref.flatten())} std {np.std(u_ref.flatten())}')
    main.testAnimation()

