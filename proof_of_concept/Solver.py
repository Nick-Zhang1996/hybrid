import numpy as np
from Problem import *

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
            #breakpoint()
            cov = np.cov(particles[elite_idx].T).reshape((problem.n,problem.n))*1.5
            print(particles[elite_idx[0]])
            print(rollout_cost[elite_idx[0]])

if __name__=='__main__':
    solver = CEM()
    problem = ParabolaWithSineNoise()
    solver.solve(problem)

