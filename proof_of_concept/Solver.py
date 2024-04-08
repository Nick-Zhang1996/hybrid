import numpy as np
from Problem import *
from util import *

# solves for simple problems
class Solver:
    def __init__(self):
        return

class CEM(Solver):
    def __init__(self):
        return

    def solve(self,problem):
        iterations = 10
        mean = np.zeros(problem.n)
        cov = np.diag([1.0]*problem.n)
        samples = 100
        elite_ratio = 0.3
        for i in range(iterations):
            # sample in param space
            particles = np.random.multivariate_normal(mean,cov,size=samples)
            particles = particles.clip(-1,1)
            # rollout
            rollout_cost = [problem.evaluate(particle.flatten()) for particle in particles]
            problem.visualize(particles, rollout_cost)
            # select elite samples
            elite_idx = np.argsort(rollout_cost)[:int(samples*elite_ratio)]
            mean = np.mean(particles[elite_idx],axis=0) 
            cov = np.cov(particles[elite_idx].T).reshape((problem.n,problem.n))*1.5
            print(particles[elite_idx[0]])
            print(rollout_cost[elite_idx[0]])

class GradientDescent(Solver):
    def __init__(self):
        return

    def solve(self,problem):
        iterations = 10
        guess = np.random.random(problem.n)
        step_size = 0.1
        decay_factor = 0.1

        for i in range(iterations):
            J = problem.jacobian(guess)
            if (np.linalg.norm(J) < 1e-2):
                break
            step = step_size*np.exp(-i*decay_factor)*J.flatten()
            problem.visualize(guess.reshape(1,-1),dir_vec=-step.reshape(1,-1))
            print(f'guess val = {problem.evaluate(guess)}')
            print(f'estimated guess val = {problem.evaluate(guess)-J@step}')

            guess -= step
        problem.visualize(guess.reshape(1,-1))



class Newton(Solver):
    def __init__(self):
        return

    def solve(self,problem):
        iterations = 10
        max_step_size = 0.1
        decay_factor = 0.1
        guess = np.random.random(problem.n)
        # DEBUG
        guess = np.array([0.6,0.6])

        for i in range(iterations):
            # 1*n
            J = problem.jacobian(guess)
            if (np.linalg.norm(J) < 1e-2):
                break
            D = dirDer(lambda x:problem.evaluate(x),guess, -J.flatten())
            step = J/np.abs(D)
            norm = np.linalg.norm(step)
            if (D<0 or norm>max_step_size):
                step = step/norm*max_step_size*np.exp(-i*decay_factor)

            problem.visualize(guess.reshape(1,-1),dir_vec=-step.reshape(1,-1))
            print(f'guess val = {problem.evaluate(guess)}')
            print(f'estimated guess val = {problem.evaluate(guess)-J@step.T}')

            guess -= step.flatten()
        problem.visualize(guess.reshape(1,-1))


if __name__=='__main__':
    #solver = CEM()
    #solver = GradientDescent()
    problem = PerlinNoise()
    #problem = ParabolaWithSineNoise()
    #problem = QuadraticParabolaWithSineNoise()

    solver = Newton()
    solver.solve(problem)

