import numpy as np
import scipy.sparse # sparse matrix operations
from abc import ABC,abstractmethod

from util import *
from TimeUtil import TimeUtil
from ResidualGame import ResidualGame

class SteinGame(ResidualGame):
    @abstractmethod
    def __init__(self):
        super().__init__()
        self.particles = 100
        self.epsilon = 1.0 # step size
        self.alpha = 1.0
        # theta = u in this version

    def solve(self,save_gif=False,visualize=False,animate=False):
        # Stage 1: Stein variational inference
        # sample initial particles
        self.dim_theta = self.T*self.N*self.m

        theta = self.initialSample()

        # one iteration
        for iter in range(self.iterations):
            self.theta_norm_median = np.median([ np.linalg.norm(particle) for particle in theta ])**2/ np.log(self.particles)
            # evaluate derivative on
            # calculate kernel table
            kernel_val_map = np.zeros((self.particles,self.particles))
            for i in range(self.particles):
                for j in range(i+1):
                    kernel_val_map[i,j] = kernel_val_map[j,i] = self.kernel(theta[i], theta[j])
            # find \nabla_theta log p_posterior(theta|O) for each particle
            posterior_log_grad = []
            for i in range(self.particles):
                # TODO after we figure out how to do this...
                print(f'particle {i}')
                val = - self.alpha * self.d_cost_d_theta(theta[i])[0] #+ 1.0/self.proposal(theta[i]) * self.d_proposal_d_theta(theta[i])
                posterior_log_grad.append(val.flatten())

            # find descent direction (for each particle)
            des_dir = []
            for i in range(self.particles):
                val = np.zeros(self.dim_theta)
                for j in range(self.particles):
                    val += self.kernel(theta[i], theta[j]) * posterior_log_grad[j] + self.d_kernel_d_theta_i(theta[j], theta[i])
                phi = 1/self.particles * val
                des_dir.append(phi)

            # update particle
            new_theta = []
            for i in range(self.particles):
                new_theta.append(theta[i] + self.epsilon * des_dir[i])

            # update prior distribution (empirical)
            # validate: check cost of new particles
            old_cost = np.sum([self.cost(val) for val in theta])
            new_cost = np.sum([self.cost(val) for val in new_theta])
            print(old_cost, new_cost)
            theta = new_theta

        # Stage 2: Residual Game
        # for now just find the best particle

    def cost(self, theta):
        return self.d_cost_d_theta(theta)[1]


    # TODO check
    def d_cost_d_theta(self, theta):
        T = self.T; N = self.N; m = self.m; n = self.n
        dim_u = (self.T, self.N, self.m)
        u = theta.reshape(dim_u)
        x = self.rollout(self.x0, u)
        h_plus_mask = self.getHplusMask(x)
        # lamda u
        # r = r0 + dr_dlamda @ dlamda + dr_dmu @ dmu
        lambda_ref = np.zeros((T,N,n))
        # defined for all h_k_i_j, but all values may not be used
        lamda_0 = np.zeros((T,N,n))
        mu_0 = np.zeros((T,N,N))
        r0 = self.r(x,u,lamda_0, mu_0, h_plus_mask)
        drdlamda = self.dr_dlamda(x, u, lamda_0, mu_0, h_plus_mask)
        drdmu = self.dr_dmu(x, u, lamda_0, mu_0, h_plus_mask)
        # r = r0 + [dr_dlamda, dr_dmu] @ [dlamda; dmu]
        Dr = np.hstack([drdlamda, drdmu])
        nonzero_rows = np.nonzero(np.sum(np.abs(Dr),axis=1))[0]
        nonzero_cols = np.nonzero(np.sum(np.abs(Dr),axis=0))[0]
        reduced_Dr = Dr[nonzero_rows,:][:,nonzero_cols]

        sparse_Dr = scipy.sparse.csc_matrix(reduced_Dr, dtype=float)
        reduced_dy, istop, itn, normr = scipy.sparse.linalg.lsqr(sparse_Dr,-r0[nonzero_rows])[:4]
        dy = np.zeros(T*N*n+T*N*N)
        dy[nonzero_cols] = reduced_dy
        split_y = lambda y: (y[:N*T*n].reshape(T,N,n), y[T*N*n:].reshape(T,N,N))
        lamda_f, mu_f = split_y(dy)
        r_min = self.r(x,u,lamda_f, mu_f, h_plus_mask)
        assert (np.linalg.norm(r0) > np.linalg.norm(r_min))

        drdu = self.dr_du(x, u, lamda_f, mu_f, h_plus_mask)
        grad = 2* r_min.T @ drdu
        return grad, np.linalg.norm(r_min)

    # TODO check
    def d_kernel_d_theta_i(self, theta_i, theta_j, kernel_val=None):
        ''' derivative of kernel for RKHS, mapping to positive scalar '''
        return self.kernel(theta_i,theta_j) * (-1/self.theta_norm_median)*2*(theta_i-theta_j).T

    def kernel(self, theta_i, theta_j):
        ''' kernel for RKHS, mapping to positive scalar '''
        return np.exp(- np.linalg.norm(theta_i-theta_j)**2/self.theta_norm_median)

    def initialSample(self):
        ''' make [self.particles] samples of size theta from an initial belief'''
        # TODO need to tune scale
        return np.random.multivariate_normal(np.zeros(self.dim_theta),np.diag([3.0]*self.dim_theta), self.particles)

    def proposal(self,theta):
        ''' evaluate marginal of proposal distribution, output: scalar '''
        return

    def d_proposal_d_theta(self,theta):
        return
