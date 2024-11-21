import numpy as np
import scipy.sparse # sparse matrix operations
from abc import ABC,abstractmethod
import scipy.sparse.linalg # sparse matrix operations
import matplotlib.pyplot as plt

from util import *
from TimeUtil import TimeUtil
from ResidualGame import ResidualGame
from Cluster import Cluster

class SteinGame(ResidualGame):
    @abstractmethod
    def __init__(self):
        super().__init__()
        self.particles = 100
        self.epsilon = 1.0 # step size
        self.alpha = 1.0
        self.stein_iterations = 30
        # theta = u in this version
        self.stein_profiler = TimeUtil(True)

    def solve(self,save_gif=False,visualize=False,animate=False):
        self.dim_theta = self.T*self.N*self.m
        T = self.T; N = self.N; n = self.n

        # Stage 1: Stein variational inference
        # sample initial particles
        self.stein_profiler.s()
        theta = self.initialSample()
        # dual variable for theta
        theta_dual = [np.zeros(T*N*n + T*N*N ) for i in range(self.particles)]
        # one iteration
        for iter in range(self.stein_iterations):
            self.theta_norm_median = np.median([ np.linalg.norm(particle) for particle in theta ])**2/ np.log(self.particles)
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
            self.stein_profiler.s('particle grad')
            for i in range(self.particles):
                #retval = self.d_cost_d_theta(theta[i])
                #current_grad = retval[0]
                #alt_grad = jacobianNumerical(self.cost, theta[i])
                grad, posterior_residual, new_dual = self.d_theta(theta[i], theta_dual[i])
                # NOTE not keeping dual
                #theta_dual[i] = new_dual
                cost_vec.append(posterior_residual)
                val = - self.alpha * grad #+ 1.0/self.proposal(theta[i]) * self.d_proposal_d_theta(theta[i])
                posterior_log_grad.append(val.flatten())
            self.stein_profiler.e('particle grad')

            # find descent direction (for each particle)
            self.stein_profiler.s('dec dir')
            des_dir = []
            for i in range(self.particles):
                val = np.zeros(self.dim_theta)
                #for j in range(self.particles):
                #    val += self.kernel(theta[i], theta[j]) * posterior_log_grad[j] #+ 0.01*self.d_kernel_d_theta_i(theta[j], theta[i])
                #phi = 1/self.particles * val
                val += posterior_log_grad[i]
                phi = val
                des_dir.append(phi)
            self.stein_profiler.e('dec dir')

            # update particle
            new_theta = []
            for i in range(self.particles):
                new_theta.append(theta[i] + self.epsilon * des_dir[i])
            new_theta = np.array(new_theta)
            # resample really bad samples
            sort_idx = np.argsort(cost_vec)
            bad_samples_idx = sort_idx[-int(0.1*self.particles):]
            new_theta[bad_samples_idx] = self.initialSample(count=len(bad_samples_idx))
            for i in bad_samples_idx:
                theta_dual[i] = np.zeros(T*N*n + T*N*N)


            # DEBUG: check cost of new particles
            #old_cost = np.sum([self.cost(val) for val in theta])
            if (iter > 0):
                count = np.sum( (np.array(cost_vec) - np.array(old_cost_vec)) < 0)
                self.print_info(f' cost decrease particle ratio : {count/self.particles}')
            cost_vec = np.array(cost_vec)
            mean_cost = np.mean(cost_vec[sort_idx[:-int(0.1*self.particles)]])
            self.print_info(f'overall mean cost: ', mean_cost)
            min_cost = np.min(cost_vec)
            self.print_info(f'min cost: ', min_cost)
            theta = new_theta
            # DEBUG plot cost
            #plt.hist(cost_vec[sort_idx[:-int(0.1*self.particles)]])
            #plt.show()

        # Stage 2: Residual Game for particle refinement
        self.stein_profiler.s('particle refine')

        # DEBUG check results after stein
        '''
        min_idx = np.argsort(cost_vec)[:10]
        for u_ref in np.array(theta)[min_idx]:
            self.visualize(u_ref,visualize=visualize, animate=animate,gif_prefix='before')
        '''

        min_idx = np.argsort(cost_vec)[:int(0.9*len(cost_vec))]
        good_u_ref = np.array(theta)[min_idx]

        # refinement
        '''
        min_idx = np.argsort(cost_vec)
        dim_u = (self.T, self.N, self.m)
        good_u_ref = []
        good_x_ref = []
        for i in range(len(min_idx)):
            u_ref, full_x_ref, has_converged = ResidualGame.solve(self,u_ref = theta[min_idx[i]].reshape(dim_u))
            if (has_converged):
                # check distance from current solution to existing solutions
                for cand_x_ref in good_x_ref:
                    distance = 0
                    for i in range(self.N):
                        for k in range(self.T):
                            distance += np.linalg.norm(full_x_ref[k,i].flatten() - cand_x_ref[k,i].flatten())

                good_u_ref.append(u_ref)
                good_x_ref.append(full_x_ref)
                self.print_info('converged')
        self.stein_profiler.e('particle refine')
        '''
        good_x_ref = []
        for i in range(len(min_idx)):
            x_ref = self.rollout(self.x0,good_u_ref[i])
            full_x_ref = np.vstack([self.x0[np.newaxis,:,:],x_ref])
            good_x_ref.append(full_x_ref)

        # DEBUG check: do particles converge to the same equilibrium? how many equilibriums?
        # check distance between equilibriums:
        '''
        # debug
        # does close particles yield close solutions? - Yes
        solution_distance = np.diag([np.inf]*elite_sample_count)
        for i in range(elite_sample_count):
            for j in range(i):
                solution_distance[i,j] = solution_distance[j,i] = np.linalg.norm(good_u_ref[i] - good_u_ref[j])

        particle_distance = np.diag([np.inf]*elite_sample_count)
        for i in range(elite_sample_count):
            for j in range(i):
                particle_distance[i,j] = particle_distance[j,i] = np.linalg.norm(theta[i]- theta[j])
        vetted_solution_distance = solution_distance.flatten()[np.logical_not(np.isinf(solution_distance.flatten()))]
        vetted_particle_distance = particle_distance.flatten()[np.logical_not(np.isinf(solution_distance.flatten()))]
        cov = np.cov( np.vstack([vetted_solution_distance, vetted_particle_distance]).T )
        print(cov)
        '''
        self.print_info(f'good u_ref {len(good_u_ref)}')
        if (len(good_u_ref) == 0):
            has_converged = False
        else:
            has_converged = True

        # form belief
        self.belief_support = np.array(good_u_ref)
        self.belief_weight = np.array([1/len(good_u_ref)]*len(good_u_ref))

        '''
        for u_ref in good_u_ref:
            self.visualize(u_ref,visualize=visualize, animate=animate,gif_prefix='before')
        '''

        self.stein_profiler.e()
        return good_u_ref[0], full_x_ref[0], has_converged

    # given observed state, update belief, assume ego agent is agent 0
    # u: current control
    # k: time step
    # TODO maybe we should add a prior based on residual
    def update(self, u, k):
        ego_agent_index = 0 # if changed, need to update following code
        u_others = u.reshape(( self.N, self.m))[1:,:]

        prob = np.zeros( len(self.belief_support))
        for i in range(len(prob)):
            reference = self.belief_support[i].reshape((self.T,self.N,self.m))[k,0:,:]
            prob[i] = np.exp(self.kernel(u, reference)) * self.belief_weight[i]
        self.belief_weight = prob / np.sum(prob)
        return


    # find gradient direction for minimizing residual
    def d_theta(self, theta, dual):
        N = self.N; T = self.T; n = self.n; m = self.m
        dim_u = (self.T, self.N, self.m)
        u_ref = theta.reshape(dim_u)
        x_ref = self.rollout(self.x0,u_ref)

        #lambda_ref = np.zeros((T,N,self.n))
        #mu_ref = np.zeros((T,N,N))
        lambda_ref = dual[:T*N*n].reshape((T,N,n))
        mu_ref = dual[T*N*n:].reshape((T,N,N))

        '''
        _x_ref = x_ref; _u_ref = u_ref
        try:
            for i in range(30):
                retval = self.cpp.step(_x_ref, _u_ref, lambda_ref, mu_ref)
                new_x_ref, new_u_ref, lambda_ref, mu_ref = [np.array(val) for val in retval]
                _x_ref = new_x_ref; _u_ref = new_u_ref
        except StopIteration as e:
            #self.print_info(e)
            new_x_ref = _x_ref; new_u_ref = _u_ref
        '''
        try:
            retval = self.cpp.step(x_ref, u_ref, lambda_ref, mu_ref)
            new_x_ref, new_u_ref, lambda_ref, mu_ref = [np.array(val) for val in retval]
        except StopIteration as e:
            #self.print_info(e)
            new_x_ref = x_ref; new_u_ref = u_ref

        new_dual = np.hstack([lambda_ref.flatten(), mu_ref.flatten()])

        h_plus_mask = self.getHplusMask(new_x_ref)
        r0 = self.r(new_x_ref,new_u_ref,lambda_ref,mu_ref,h_plus_mask)
        grad = -(new_u_ref - u_ref).reshape(1,-1)
        return grad, np.linalg.norm(r0), new_dual


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

    def initialSample(self,count=None):
        ''' make [self.particles] samples of size theta from an initial belief'''
        # TODO need to tune scale
        if (count is None):
            return np.random.multivariate_normal(np.zeros(self.dim_theta),np.diag([2*0.5]*self.dim_theta), self.particles)
        else:
            return np.random.multivariate_normal(np.zeros(self.dim_theta),np.diag([2*0.5]*self.dim_theta), count)

    def proposal(self,theta):
        ''' evaluate marginal of proposal distribution, output: scalar '''
        return

    def d_proposal_d_theta(self,theta):
        return

    def final(self):
        super().final()
        self.stein_profiler.summary()
