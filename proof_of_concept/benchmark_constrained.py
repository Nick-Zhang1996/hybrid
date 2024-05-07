# test all solvers on all problems, plot cost vs evaluations
import matplotlib.pyplot as plt
from Problem import *
from Solver import *

problem_vec = [ParabolaWithSineNoise2D,ParabolaWithSineNoise3D]
#solver_vec = [CEM,Newton,Hybrid,DualAscent]
#solver_rerun_vec = [3,100,1,30]
solver_vec = [Newton,Hybrid,DualAscent]
solver_rerun_vec = [50,1,40]
# minimum function value
value_lut = dict()
# number of function evaluations (excluding jacobian/hessian)
evaluation_lut = dict()
# Constraint residuals
residuals_lut = dict()

for problem_class in problem_vec:
    for solver_class,reruns in zip(solver_vec,solver_rerun_vec):
        print(f'{str(problem_class)}, {str(solver_class)}')
        value_lut[(problem_class,solver_class)] = []
        evaluation_lut[(problem_class,solver_class)] = []
        residuals_lut[(problem_class,solver_class)] = []

        for experiment_idx in range(10):
            print(f'exp {experiment_idx}')
            values = []
            evaluations = []
            residuals = []
            fun_x_min = 1e1
            problem = problem_class()
            for rerun_idx in range(reruns):
                print(f'rerun {rerun_idx}')
                solver = solver_class(problem)
                problem.setConstraints(solver)
                for i in range(solver.iterations):
                    print(f'i{i} ',end='')
                    retval = solver.step(i)
                    if retval is not None:
                        x,fun_x = retval
                    evaluations.append(problem.evaluations)
                    residual = solver.getResiduals(x)
                    residuals.append(residual)
                    if (residual < 1e-3):
                        fun_x_min = min(fun_x,fun_x_min)
                    values.append(fun_x_min)

            value_lut[(problem_class,solver_class)].append(values)
            evaluation_lut[(problem_class,solver_class)].append(evaluations)
            residuals_lut[(problem_class,solver_class)].append(residuals)

# plot results
fig,axes = plt.subplots(len(problem_vec),2)
for problem_class,ax in zip(problem_vec,axes):
    xlim = 1e99
    for solver_class in solver_vec:
        #breakpoint()
        value_mean = np.mean(value_lut[(problem_class,solver_class)], axis=0)
        evaluation_mean = np.mean(evaluation_lut[(problem_class,solver_class)], axis=0)
        value_std = np.std(value_lut[(problem_class,solver_class)], axis=0)
        evaluation_std = np.std(evaluation_lut[(problem_class,solver_class)], axis=0)
        ax[0].plot(evaluation_mean, value_mean,label=str(solver_class))
        #ax[0].fill_between(evaluation_mean, value_mean-value_std, value_mean+value_std,alpha=0.4)
        xlim = min(evaluation_mean[-1],xlim)
        print(str(problem_class), str(solver_class),evaluation_mean[-1])

        residuals_mean = np.mean(residuals_lut[(problem_class,solver_class)], axis=0)
        residuals_std = np.std(residuals_lut[(problem_class,solver_class)], axis=0)
        ax[1].plot(evaluation_mean, residuals_mean,label=str(solver_class))
        #ax[1].fill_between(evaluation_mean, residuals_mean-residuals_std, residuals_mean+residuals_std,alpha=0.4)

    ax[0].set_title(str(problem_class))
    ax[1].set_title('constraint residuals')
    ax[1].set_yscale('log')
    #ax.set_xlim([0,xlim])
    #ax.set_xlim([0,1000])
plt.legend()
plt.show()
breakpoint()


