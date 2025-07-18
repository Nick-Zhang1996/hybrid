import numpy as np
from Problem import *
from utilities.util import *
import scipy.stats as stats
from scipy.optimize import minimize
from math import exp, log
from itertools import compress


# solves for simple problems
class Solver:

    def __init__(self, problem):
        self.problem = problem
        self.iterations = 30
        self.samples = 20
        return

    def solve(self, visualize=False, save_gif=False):
        for i in range(self.iterations):
            retval = self.step(i, visualize, save_gif)
            if (retval is None):
                break
            else:
                x, fun_x = retval

        self.problem.final()
        return x, fun_x

    def step(self, i, visualize=False, save_gif=False):
        x = 0
        fun_x = 0
        return x, fun_x

    def getResiduals(self, x):
        hx = np.array([hh(x) for hh in self.hx])
        lx = np.array([ll(x) for ll in self.lx])
        residual_h = np.sum(hx[hx > 0]**2)
        residual_l = np.sum(lx**2)
        #print(f' residual h: {residual_h},residual l: {residual_l}')
        return residual_l + residual_h


class Scipy(Solver):

    def __init__(self, problem):
        super().__init__(problem)
        self.guess = np.random.random(self.problem.n)

        # penalty coefficient for constraints
        self.rho = 10
        self.rho_b = 1.5  # exponential growth rate for rho
        # list of inequality constraint functions
        self.hx = []
        # list of equality constraint functions
        self.lx = []
        return

    def addHx(self, fun):
        '''  h(x) <= 0 '''
        self.hx.append(fun)
        return

    def addLx(self, fun):
        '''  l(x) = 0 '''
        self.lx.append(fun)
        return

    def evaluate(self, x):
        return self.problem.evaluate(x) + 0.5 * self.rho * sum([
            lx(x)**2 for lx in self.lx
        ]) + 0.5 * self.rho * sum([max(0, hx(x))**2 for hx in self.lx])

    def solve(self, visualize=False, save_gif=False):
        res = minimize(lambda x: self.evaluate(x), x0=self.guess)
        x = res['x']
        fun_x = res['fun']
        return x, fun_x


class CEM(Solver):

    def __init__(self, problem):
        super().__init__(problem)
        self.mean = np.zeros(problem.n)
        self.cov = np.diag([1.0] * problem.n)
        self.elite_ratio = 0.3
        # penalty coefficient for constraints
        self.rho = 10
        self.rho_b = 1.5  # exponential growth rate for rho

        # list of inequality constraint functions
        self.hx = []
        # list of equality constraint functions
        self.lx = []
        return

    def addHx(self, fun):
        '''  h(x) <= 0 '''
        self.hx.append(fun)
        return

    def addLx(self, fun):
        '''  l(x) = 0 '''
        self.lx.append(fun)
        return

    def evaluate(self, x):
        return self.problem.evaluate(x) + 0.5 * self.rho * sum([
            lx(x)**2 for lx in self.lx
        ]) + 0.5 * self.rho * sum([max(0, hx(x))**2 for hx in self.lx])

    def step(self, i, visualize=False, save_gif=False):
        # sample in param space
        particles = np.random.multivariate_normal(self.mean,
                                                  self.cov,
                                                  size=self.samples)
        #particles = particles.clip(-1,1)
        # resample points out of bounds
        mask = np.any(np.logical_or(particles < -1, particles > 1), axis=1)
        while (np.any(mask)):
            size = np.sum(mask)
            particles[mask, :] = np.random.multivariate_normal(self.mean,
                                                               self.cov,
                                                               size=size)
            mask = np.any(np.logical_or(particles < -1, particles > 1), axis=1)
        # rollout
        rollout_cost = [
            self.evaluate(particle.flatten()) for particle in particles
        ]
        # select elite samples
        elite_idx = np.argsort(rollout_cost)[:int(self.samples *
                                                  self.elite_ratio)]
        self.mean = np.mean(particles[elite_idx], axis=0)
        self.cov = np.cov(particles[elite_idx].T).reshape(
            (self.problem.n, self.problem.n)) * 1.5

        self.problem.visualize(val_vec=particles,
                               fun_val_vec=rollout_cost,
                               visualize=visualize,
                               save_gif=save_gif)
        self.rho *= self.rho_b
        return particles[elite_idx[0]], rollout_cost[elite_idx[0]]


# TODO kinda useless now, revisit after Newton method
# TODO add line search, stopping criterion
class GradientDescent(Solver):

    def __init__(self, problem):
        super().__init__(problem)
        self.step_size = 0.1
        self.decay_factor = 0.1
        self.guess = np.random.random(self.problem.n)
        return

    def step(self, i, visualize=False, save_gif=False):
        J = self.problem.jacobian(self.guess)
        D = dirDer(lambda x: self.problem.evaluate(x), self.guess,
                   -J.flatten())
        step = J / np.abs(D)
        norm = np.linalg.norm(step)

        step = step / norm * self.max_step_size * np.exp(
            -i * self.decay_factor)

        self.problem.visualize(self.guess.reshape(1, -1),
                               dir_vec=-step.reshape(1, -1),
                               visualize=visualize,
                               save_gif=save_gif)
        self.guess -= step.flatten()

        return self.guess, self.problem.evaluateNoCount(self.guess)


class Newton(Solver):

    def __init__(self, problem, x0=None):
        super().__init__(problem)
        self.max_step_size = 0.1
        self.decay_factor = 0.1
        if (x0 is None):
            self.guess = (np.random.random(problem.n) - 0.5) * 2
        else:
            self.guess = x0
        # list of inequality constraint functions
        self.hx = []
        # lagrange multiplier
        self.v = []
        # list of equality constraint functions
        self.lx = []
        self.u = []
        # parameter in barrier function, larger means more 'strict'
        self.rho = 2
        self.rho_b = 1.5  # exponential growth rate for rho

        # backtracking line search param
        self.bc_a = 0.3  #alpha
        self.bc_b = 0.5  #beta

        return

    def addHx(self, fun):
        '''  h(x) <= 0 '''
        self.hx.append(fun)
        self.v.append(0.0)
        return

    def addLx(self, fun):
        '''  l(x) = 0 '''
        self.lx.append(fun)
        self.u.append(0.0)
        return

    def evaluate(self, x, h_neg=[]):
        return self.problem.evaluate(x) + -1 / self.rho * sum(
            [log(-min(h(x), -1e-100)) for h in h_neg])

    def evaluateNoCount(self, x, h_neg=[]):
        return self.problem.evaluateNoCount(x) + -1 / self.rho * sum(
            [log(-min(h(x), -1e-100)) for h in h_neg])

    def step(self, i, visualize=False, save_gif=False):
        # TODO move to post initialization
        self.u = np.array(self.u)
        self.v = np.array(self.v)

        # f^: evaluate()
        # \hat{x}: x0
        n = self.problem.n
        x0 = self.guess
        h_x0 = np.array([h(x0) for h in self.hx])
        # identify h- and h+
        h_pos = list(compress(self.hx, h_x0 > 0))
        h_neg = list(compress(self.hx, h_x0 <= 0))
        # Hessian, Jacobian for f^
        H = hessianNumerical(lambda x: self.evaluateNoCount(x, h_neg), x0)
        J = jacobianNumerical(lambda x: self.evaluateNoCount(x, h_neg), x0)

        # If negative hessian, do gradient descent
        # TODO is this reasonable?
        try:
            np.linalg.cholesky(H)
        except np.linalg.LinAlgError:
            H = np.eye(n)
            #print('negative hessian')

        # assemble l^(x) = [l(x), h+(x)]
        l_hat = self.lx + h_pos
        m_l_hat = len(l_hat)
        # dual variable associated with l_hat, i.e. \lambda
        u_hat = np.hstack([self.u, self.v[h_x0 > 0]]).reshape(m_l_hat, 1)

        if (m_l_hat > 0):
            l_hat_x0 = np.vstack([ll(x0) for ll in l_hat])
            J_l_hat = jacobianNumerical(
                lambda x: np.array([l(x) for l in l_hat]), x0, dim=len(l_hat))
            # linear system for primal-dual problem: A @ [dx,lambda]^T = B
            A = np.block([[H, J_l_hat.T],
                          [J_l_hat, np.zeros((m_l_hat, m_l_hat))]])
            B = np.vstack([-J.T - J_l_hat.T @ u_hat, -l_hat_x0])
        else:
            # TODO maybe we don't need this
            l_hat_x0 = np.zeros((0, 1))
            J_l_hat = np.zeros((0, n))
            A = H
            B = -J.T
        assert (A.shape == (n + m_l_hat, n + m_l_hat))
        assert (B.shape == (n + m_l_hat, 1))
        # y: concatenated [dx, lambda]
        y, residuals, rank, s = np.linalg.lstsq(A, B)
        dx = y[:n, :].flatten()
        du = y[n:, :].flatten()

        f_x0 = self.evaluate(x0)
        J_f = J
        J_l = J_l_hat
        l_x = np.array([l(x0) for l in l_hat]).reshape(m_l_hat, 1)
        r0 = np.vstack([J_f.T + J_l.T @ u_hat, l_x])
        r0_norm = np.linalg.norm(r0)
        # stopping criteria
        if (r0_norm < 1e-5):
            return x0, f_x0

        # backtracking line search
        t = 1.0  # step size
        for i in range(5):
            J_f = jacobianNumerical(lambda x: self.evaluateNoCount(x, h_neg),
                                    x0 + t * dx)
            J_l = jacobianNumerical(lambda x: np.array([l(x) for l in l_hat]),
                                    x0 + t * dx,
                                    dim=len(l_hat))
            l_x = np.array([l(x0 + t * dx) for l in l_hat]).reshape(m_l_hat, 1)
            r_t = np.vstack(
                [J_f.T + J_l.T @ (u_hat + t * du.reshape(m_l_hat, 1)), l_x])
            r_t_norm = np.linalg.norm(r_t)
            if (r_t_norm > (1 - self.bc_a * t) * r0_norm):
                t *= self.bc_b
            else:
                break
        #print(f't={t}')

        line_search_dx = t * dx
        line_search_du = t * du
        primal_res = np.linalg.norm(r_t[:n])
        dual_res = np.linalg.norm(r_t[n:])
        if (primal_res < 1e-10 and dual_res < 1e-10):
            return None
        #print(f'dx = {dx}')
        #print(f't = {t}')
        #print(f'primal res:{primal_res}, dual res:{dual_res}')
        '''

        # DEBUG plot primal and dual residual as a function of t
        t_vec = []
        r_primal_vec = []
        r_dual_vec = []
        r_norm_vec = []
        #t = min(1,0.1/r0_norm) # step size
        t = 1.0 # step size
        while True:
            J_f = jacobianNumerical(lambda x:self.evaluate(x,h_neg), x0 + t*dx)
            J_l = jacobianNumerical(lambda x:np.array([l(x) for l in l_hat]), x0 + t*dx,dim=len(l_hat))
            l_x = np.array([l(x0+t*dx) for l in l_hat]).reshape(m_l_hat,1)
            r_t = np.vstack([J_f.T + J_l.T @ (u_hat+t*du),l_x])
            r_t_norm = np.linalg.norm(r_t)
            t_vec.append(t)
            r_primal_vec.append(np.linalg.norm(r_t[0,:n]))
            r_dual_vec.append(  np.linalg.norm(l_x))
            r_norm_vec.append(  r_t_norm )
            t *= self.bc_b
            if (t < np.linalg.norm(line_search_dx)/np.linalg.norm(dx)):
                break

        '''
        self.problem.visualize(self.guess.reshape(1, -1),
                               dir_vec=line_search_dx.reshape(1, -1),
                               visualize=visualize,
                               save_gif=save_gif)
        '''

        # DEBUG plot
        t = 1.0
        tt = np.linspace(0,t,2)
        plt.plot(t_vec, r_primal_vec,'*', label='primal')
        plt.plot(t_vec, r_dual_vec,'*', label='dual')
        plt.plot(t_vec, r_norm_vec,'*', label='total')
        plt.plot(tt,(1-self.bc_a*tt)*r0_norm)
        plt.legend()
        plt.show()
        breakpoint()
        '''

        self.u += du[:len(self.lx)]
        self.v[h_x0 > 0] += line_search_du[len(self.lx):]
        self.guess += line_search_dx
        self.rho *= self.rho_b
        return self.guess, self.problem.evaluateNoCount(self.guess)


class DualAscent(Solver):

    def __init__(self, problem, x0=None):
        super().__init__(problem)
        # primal descent
        self.primal_max_step_size = 0.1
        _, self.primal_decay_factor = self.findExpCoeff(
            1.0, 0.5, self.iterations)
        # dual ascent
        self.dual_step_size, self.dual_decay_factor = self.findExpCoeff(
            2.0, 1.0, self.iterations)

        self.x = np.random.random(problem.n) if x0 is None else x0
        # list of inequality constraint functions
        self.hx = []
        # lagrange multiplier for h(x)
        self.v = []
        # list of equality constraint functions
        self.lx = []
        # lagrange multiplier for l(x)
        self.u = []
        return

    def findExpCoeff(self, s_0, s_f, iterations):
        ''' 
        Let learning step size be s(i) = A*exp^(i*B)
        find A,B such that  s(0) = s_0, s(iterations-1) = s_f
        '''
        A = s_0
        B = log(s_f / A) / (iterations - 1)
        return A, B

    def primalLr(self, i):
        return exp(i * self.primal_decay_factor)

    def dualLr(self, i):
        return self.dual_step_size * exp(i * self.primal_decay_factor)

    def addHx(self, fun):
        '''  h(x) <= 0 '''
        self.hx.append(fun)
        self.v.append(0.0)
        return

    def addLx(self, fun):
        '''  l(x) = 0 '''
        self.lx.append(fun)
        self.u.append(0.0)
        return

    def step(self, i, visualize=False, save_gif=False):
        self.u = np.array(self.u)
        self.v = np.array(self.v)
        # quadratic penalty for constraint violation
        p = 100.0 * exp(i)
        # construct augmented lagrangian
        Lx = lambda x, u, v: self.problem.evaluate(x) + sum(
            [vv * hh(x) for (vv, hh) in zip(v, self.hx)]) + 0.5 * p * sum([
                hh(x)**2 if hh(x) > 0 else 0 for hh in self.hx
            ]) + sum([uu * ll(x)
                      for (uu, ll) in zip(u, self.lx)]) + 0.5 * p * sum(
                          [ll(x)**2 if ll(x) > 0 else 0 for ll in self.lx])
        # Primal descent
        J = jacobianNumerical(lambda x: Lx(x, self.u, self.v), self.x)  # 1*n
        norm = np.linalg.norm(J)
        if (norm > self.primal_max_step_size):
            step = -J / norm * self.primal_max_step_size * self.primalLr(i)
        else:
            step = -J * self.primalLr(i)
        self.x += step.flatten()
        print(f'iter={i} lr={self.primalLr(i):.2f}')

        # Dual ascent
        if (len(self.v) > 0):
            self.v += self.dualLr(i) * np.array([hh(self.x) for hh in self.hx])
            self.v[self.v < 0] = 0.0
            #print('h',[hh(self.x) for hh in self.hx],'v',self.v)

        if (len(self.u) > 0):
            self.u += self.dualLr(i) * np.array([ll(self.x) for ll in self.lx])
            #print('l',[ll(self.x) for ll in self.lx],'u',self.u)

        self.problem.visualize(self.x.reshape(1, -1),
                               dir_vec=step.reshape(1, -1),
                               visualize=visualize,
                               save_gif=save_gif)
        return self.x, self.problem.evaluateNoCount(self.x)


class Hybrid(Solver):
    '''Hybrid solver, sample'''

    def __init__(self, problem):
        super().__init__(problem)
        self.mean = np.zeros(problem.n)
        self.cov = np.diag([1.0] * problem.n)
        self.elite_ratio = 0.3
        self.max_step_size = 0.1
        self.decay_factor = 0.1

        # list of inequality constraint functions
        self.hx = []
        # lagrange multiplier for h(x)
        self.u = []
        # list of equality constraint functions
        self.lx = []
        # lagrange multiplier for h(x)
        self.v = []
        return

    def addHx(self, fun):
        '''  h(x) <= 0 '''
        self.hx.append(fun)
        self.v.append(0.0)
        return

    def addLx(self, fun):
        '''  l(x) = 0 '''
        self.lx.append(fun)
        self.u.append(0.0)
        return

    def step(self, i, visualize=False, save_gif=False):
        # sample in param space
        # Gaussian
        #particles = np.random.multivariate_normal(self.mean,self.cov,size=self.samples)
        #particles = particles.clip(-1,1)
        # uniform
        #particles = np.random.uniform(low=-1, high=1,size=self.samples)[:,np.newaxis]
        # truncated Gaussian
        if (np.linalg.norm(self.cov) < 1e-10):
            return None
        particles = np.random.multivariate_normal(self.mean,
                                                  self.cov,
                                                  size=self.samples)
        mask = np.any(np.logical_or(particles < -1, particles > 1), axis=1)
        while (np.any(mask)):
            size = np.sum(mask)
            particles[mask, :] = np.random.multivariate_normal(self.mean,
                                                               self.cov,
                                                               size=size)
            mask = np.any(np.logical_or(particles < -1, particles > 1), axis=1)

        # rollout
        rollout_cost = [
            self.problem.evaluate(particle.flatten()) for particle in particles
        ]
        old_particles = particles.copy()
        dir_vec = []
        for guess in particles:
            # Hybrid step: Newton
            '''
            J = self.problem.jacobian(guess)
            if (np.linalg.norm(J) < 1e-2):
                break
            step = -J
            '''
            # Hybrid step: random
            '''
            step = np.random.uniform(size=self.problem.n)
            D = dirDer(lambda x:self.problem.evaluate(x),guess, step.flatten())
            step = step/np.abs(D)
            '''
            '''
            # common to Newton/random
            norm = np.linalg.norm(step)
            if (D<0 or norm>self.max_step_size):
                step = step/norm*self.max_step_size*np.exp(-i*self.decay_factor)
            guess += step.flatten()
            dir_vec.append(step.flatten())
            '''

            # Hybrid step: Dual Ascent/Newton
            old_guess = guess.copy()
            # DA
            #kernel = DualAscent(self.problem,x0=guess)
            # Newton step
            kernel = Newton(self.problem, x0=guess)

            kernel.hx = self.hx
            kernel.lx = self.lx
            kernel.v = [0.0] * len(self.hx)
            kernel.u = [0.0] * len(self.lx)
            for i in range(10):
                guess, _ = kernel.step(i)
            step = guess - old_guess
            dir_vec.append(step.flatten())

        self.problem.visualize(old_particles,
                               rollout_cost,
                               np.array(dir_vec),
                               visualize=visualize,
                               save_gif=save_gif)

        # select elite samples
        elite_idx = np.argsort(rollout_cost)[:int(self.samples *
                                                  self.elite_ratio)]
        self.mean = np.mean(particles[elite_idx], axis=0)
        self.cov = np.cov(particles[elite_idx].T).reshape(
            (self.problem.n, self.problem.n)) * 1.5
        return particles[elite_idx[0]], rollout_cost[elite_idx[0]]


if __name__ == '__main__':
    #solver = CEM()
    #solver = GradientDescent()
    #problem = PerlinNoise()
    #problem = ParabolaWithSineNoise()
    problem = ParabolaWithSineNoise2D()
    breakpoint()

    solver = Newton(problem)
    #solver = Hybrid(problem)
    solver.solve(True)
