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
        self.particles = 10
        self.epsilon = 1.0 # step size
        self.alpha = 1.0
        self.stein_iterations = 5
        # theta = u in this version
        self.stein_profiler = TimeUtil(True)

    def solve(self,save_gif=False,visualize=False,animate=False):
        # Stage 1: Stein variational inference
        # sample initial particles
        self.dim_theta = self.T*self.N*self.m

        theta = self.initialSample()
        self.stein_profiler.s()

        # one iteration
        for iter in range(self.stein_iterations):
            self.theta_norm_median = np.median([ np.linalg.norm(particle) for particle in theta ])**2/ np.log(self.particles)
            # evaluate derivative on
            # calculate kernel table
            kernel_val_map = np.zeros((self.particles,self.particles))
            self.stein_profiler.s('kernel map')
            for i in range(self.particles):
                for j in range(i+1):
                    kernel_val_map[i,j] = kernel_val_map[j,i] = self.kernel(theta[i], theta[j])
            self.stein_profiler.e('kernel map')
            # find \nabla_theta log p_posterior(theta|O) for each particle
            posterior_log_grad = []
            if (iter > 0):
                old_cost_vec = cost_vec
            cost_vec = []
            for i in range(self.particles):
                # TODO after we figure out how to do this...
                retval = self.d_cost_d_theta(theta[i])
                cost_vec.append(retval[1])
                val = - self.alpha * retval[0] #+ 1.0/self.proposal(theta[i]) * self.d_proposal_d_theta(theta[i])
                posterior_log_grad.append(val.flatten())

            # find descent direction (for each particle)
            self.stein_profiler.s('dec dir')
            des_dir = []
            for i in range(self.particles):
                val = np.zeros(self.dim_theta)
                for j in range(self.particles):
                    val += self.kernel(theta[i], theta[j]) * posterior_log_grad[j] + self.d_kernel_d_theta_i(theta[j], theta[i])

                phi = 1/self.particles * val
                des_dir.append(phi)
            self.stein_profiler.e('dec dir')

            # update particle
            new_theta = []
            for i in range(self.particles):
                new_theta.append(theta[i] + self.epsilon * des_dir[i])

            # update prior distribution (empirical)
            # validate: check cost of new particles
            #old_cost = np.sum([self.cost(val) for val in theta])
            if (iter > 0):
                count = np.sum( (np.array(cost_vec) - np.array(old_cost_vec)) < 0)
                self.print_info(f' cost decrease particle ratio : {count/self.particles}')
            new_cost = np.sum(cost_vec)
            self.print_info(f'overall cost: ', new_cost)
            #old_min_cost = np.min([self.cost(val) for val in theta])
            new_min_cost = np.min(cost_vec)
            self.print_info(f'min cost: ', new_min_cost)
            theta = new_theta

        # Stage 2: Residual Game
        min_idx = np.argsort(cost_vec)
        dim_u = (self.T, self.N, self.m)
        for i in range(10):
            u_ref, full_x_ref, has_converged = ResidualGame.solve(self,u_ref = theta[min_idx[i]].reshape(dim_u))
            if (has_converged):
                self.print_info('converged')
                break

        self.stein_profiler.e()
        return u_ref, full_x_ref, has_converged

    def cost(self, theta):
        return self.d_cost_d_theta(theta)[1]

    # TODO check
    def d_cost_d_theta(self, theta):
        self.stein_profiler.s('init')
        T = self.T; N = self.N; m = self.m; n = self.n
        dim_u = (self.T, self.N, self.m)
        u = theta.reshape(dim_u)
        x = self.rollout(self.x0, u)
        self.stein_profiler.e('init')

        self.stein_profiler.s('h_plus')
        h_plus_mask = self.getHplusMask(x)
        self.stein_profiler.e('h_plus')

        self.stein_profiler.s('residual')
        # lamda u
        # r = r0 + dr_dlamda @ dlamda + dr_dmu @ dmu
        lambda_ref = np.zeros((T,N,n))
        # defined for all h_k_i_j, but all values may not be used
        lamda_0 = np.zeros((T,N,n))
        mu_0 = np.zeros((T,N,N))
        r0 = self.r(x,u,lamda_0, mu_0, h_plus_mask)
        drdlamda = self.dr_dlamda(x, u, lamda_0, mu_0, h_plus_mask)
        drdmu = self.dr_dmu(x, u, lamda_0, mu_0, h_plus_mask)
        self.stein_profiler.e('residual')

        # r = r0 + [dr_dlamda, dr_dmu] @ [dlamda; dmu]
        self.stein_profiler.s('reduce')
        Dr = np.hstack([drdlamda, drdmu])
        nonzero_rows = np.nonzero(np.sum(np.abs(Dr),axis=1))[0]
        nonzero_cols = np.nonzero(np.sum(np.abs(Dr),axis=0))[0]
        reduced_Dr = Dr[nonzero_rows,:][:,nonzero_cols]
        self.stein_profiler.e('reduce')

        self.stein_profiler.s('lsqr')
        sparse_Dr = scipy.sparse.csc_matrix(reduced_Dr, dtype=float)
        reduced_dy, istop, itn, normr = scipy.sparse.linalg.lsqr(sparse_Dr,-r0[nonzero_rows])[:4]
        self.stein_profiler.e('lsqr')
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
        return np.random.multivariate_normal(np.zeros(self.dim_theta),np.diag([0.5]*self.dim_theta), self.particles)

    def proposal(self,theta):
        ''' evaluate marginal of proposal distribution, output: scalar '''
        return

    def d_proposal_d_theta(self,theta):
        return

    def final(self):
        super().final()
        self.stein_profiler.summary()
