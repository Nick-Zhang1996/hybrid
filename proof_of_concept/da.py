# dual ascent test code
from Problem import *
from Solver import *

problem = ParabolaWithSineNoise2D()
solver = Scipy(problem)
print(solver.solve())


if __name__=='__main__':
    #solver = CEM()
    #solver = GradientDescent()
    #problem = PerlinNoise()
    #problem = ParabolaWithSineNoise()
    problem = ParabolaWithSineNoise2D()

    #solver = Newton(problem)
    #solver = Hybrid(problem)

    x_vec = []
    fun_x_vec = []
    residual_vec = []
    for i in range(10):
        #solver = DualAscent(problem)
        solver = Hybrid(problem)
        solver.addHx(lambda x:(x[0]-0.15)**2+(x[1]-0.05)**2-0.3**2)
        x,fun_x = solver.solve(False)
        residual = np.array([hh(x) for hh in solver.hx])
        residual = np.sum(residual[residual>0]**2)
        x_vec.append(x)
        fun_x_vec.append(fun_x)
        residual_vec.append(residual)

    print(f'x (mean/std) {np.mean(x_vec):.2f} / {np.std(x_vec):.2f}')
    print(f'fun_x (mean/std/min) {np.mean(fun_x_vec):.2f} / {np.std(fun_x_vec):.2f}/{np.min(fun_x_vec):.2f}')
    print(f'residual (mean/std/max) {np.mean(residual_vec):.2f} / {np.std(residual_vec):.2f} / {np.max(residual_vec):.2f}')
    idx = np.argmin(fun_x_vec)
    print(f'x: {x_vec[idx]}')
    #breakpoint()


