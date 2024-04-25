# solve example problems
from Problem import *
from Solver import *


if __name__=='__main__':
    problem = ParabolaWithSineNoise2D()
    solver = Newton(problem)
    # TODO put constraints in getHx
    problem.setConstraints(solver)
    x,fun_x = solver.solve(visualize=True,save_gif=False)
    hx = np.array([hh(x) for hh in solver.hx])
    lx = np.array([ll(x) for ll in solver.lx])
    residual_h = np.sum(hx[hx>0]**2)
    residual_l = np.sum(lx**2)
    print(f' residual h: {residual_h},residual l: {residual_l}')

