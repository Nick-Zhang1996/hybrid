import numpy as np
from Problem import *
from util import *

# solves for simple problems
class Solver:
    def __init__(self,problem):
        self.problem = problem
        self.iterations = 5
        self.samples = 50
        return
    def solve(self):
        for i in range(self.iterations):
            self.step(i)
    def step(self,i,visualize=False):
        x = 0
        fun_x = 0
        return x,fun_x

class CEM(Solver):
    def __init__(self,problem):
        super().__init__(problem)
        self.mean = np.zeros(problem.n)
        self.cov = np.diag([1.0]*problem.n)
        self.elite_ratio = 0.3
        return

    def step(self,i,visualize=False):
        # sample in param space
        particles = np.random.multivariate_normal(self.mean,self.cov,size=self.samples)
        particles = particles.clip(-1,1)
        # rollout
        rollout_cost = [self.problem.evaluate(particle.flatten()) for particle in particles]
        # select elite samples
        elite_idx = np.argsort(rollout_cost)[:int(self.samples*self.elite_ratio)]
        self.mean = np.mean(particles[elite_idx],axis=0) 
        self.cov = np.cov(particles[elite_idx].T).reshape((self.problem.n,self.problem.n))*1.5

        if (visualize):
            self.problem.visualize(particles, rollout_cost)
        return particles[elite_idx[0]], rollout_cost[elite_idx[0]]

class GradientDescent(Solver):
    def __init__(self,problem):
        super().__init__(problem)
        self.step_size = 0.1
        self.decay_factor = 0.1
        self.guess = np.random.random(self.problem.n)
        return

    def step(self,i,visualize=False):
        J = self.problem.jacobian(self.guess)
        '''
        if (np.linalg.norm(J) < 1e-2):
            break
        '''
        step = self.step_size*np.exp(-i*self.decay_factor)*J.flatten()
        if (visualize):
            self.problem.visualize(self.guess.reshape(1,-1),dir_vec=-self.step.reshape(1,-1))
        #print(f'guess val = {problem.evaluate(guess)}')
        #print(f'estimated guess val = {problem.evaluate(guess)-J@step}')

        self.guess -= step
        if (visualize):
            self.problem.visualize(self.guess.reshape(1,-1))
        return self.guess, self.problem.evaluate(self.guess)



class Newton(Solver):
    def __init__(self,problem):
        super().__init__(problem)
        self.max_step_size = 0.1
        self.decay_factor = 0.1
        self.guess = np.random.random(problem.n)
        return

    def step(self,i,visualize=False):
        # 1*n
        J = self.problem.jacobian(self.guess)
        '''
        if (np.linalg.norm(J) < 1e-2):
            break
        '''
        D = dirDer(lambda x:self.problem.evaluate(x),self.guess, -J.flatten())
        step = J/np.abs(D)
        norm = np.linalg.norm(step)
        if (D<0 or norm>self.max_step_size):
            step = step/norm*self.max_step_size*np.exp(-i*self.decay_factor)

        if (visualize):
            self.problem.visualize(self.guess.reshape(1,-1),dir_vec=-step.reshape(1,-1))
        self.guess -= step.flatten()
        return self.guess, self.problem.evaluate(self.guess)

class Hybrid(Solver):
    '''Hybrid solver, sample'''
    def __init__(self,problem):
        super().__init__(problem)
        self.mean = np.zeros(problem.n)
        self.cov = np.diag([1.0]*problem.n)
        self.elite_ratio = 0.3
        self.max_step_size = 0.1
        self.decay_factor = 0.1
        return

    def step(self,i,visualize=False):
        # sample in param space
        particles = np.random.multivariate_normal(self.mean,self.cov,size=self.samples)
        particles = particles.clip(-1,1)
        # rollout
        rollout_cost = [self.problem.evaluate(particle.flatten()) for particle in particles]
        # NOTE Hybrid step: Newton
        old_particles = particles.copy()
        dir_vec = []
        for guess in particles:
            J = self.problem.jacobian(guess)
            if (np.linalg.norm(J) < 1e-2):
                break
            D = dirDer(lambda x:self.problem.evaluate(x),guess, -J.flatten())
            step = J/np.abs(D)
            norm = np.linalg.norm(step)
            if (D<0 or norm>self.max_step_size):
                step = step/norm*self.max_step_size*np.exp(-i*self.decay_factor)
            guess -= step.flatten()
            dir_vec.append(step.flatten())
        if (visualize):
            self.problem.visualize(old_particles, rollout_cost,np.array(dir_vec))

        # select elite samples
        elite_idx = np.argsort(rollout_cost)[:int(self.samples*self.elite_ratio)]
        self.mean = np.mean(particles[elite_idx],axis=0)
        self.cov = np.cov(particles[elite_idx].T).reshape((self.problem.n,self.problem.n))*1.5
        return particles[elite_idx[0]], rollout_cost[elite_idx[0]]

if __name__=='__main__':
    #solver = CEM()
    #solver = GradientDescent()
    problem = PerlinNoise()
    #problem = ParabolaWithSineNoise()
    #problem = ParabolaWithSineNoise2D()

    solver = Newton()
    #solver = Hybrid()
    solver.solve(problem)

