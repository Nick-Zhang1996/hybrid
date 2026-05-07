# solve example problems
from Problem import *
from Solver import *


if __name__=='__main__':
    problem = ParabolaWithSineNoise3D()
    solver = DualAscent(problem)
    problem.setConstraints(solver)
    #x,fun_x = solver.solve(visualize=True,save_gif=False)
    for i in range(solver.iterations):
        retval = solver.step(i,visualize=True,save_gif=False)
        if (retval is None):
            break
        x,fun_x = retval
        hx = np.array([hh(x) for hh in solver.hx])
        lx = np.array([ll(x) for ll in solver.lx])
        residual_h = np.sum(hx[hx>0]**2)
        residual_l = np.sum(lx**2)
        print(f'fun_x {fun_x} residual h: {residual_h},residual l: {residual_l}')
    breakpoint()

