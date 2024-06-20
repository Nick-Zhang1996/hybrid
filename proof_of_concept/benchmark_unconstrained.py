# test all solvers on all problems, plot cost vs evaluations
import matplotlib.pyplot as plt
from Problem import *
from Solver import *

problem_vec = [PerlinNoise,ParabolaWithSineNoise2D]
solver_vec = [CEM,Newton,Hybrid,Scipy]
solver_rerun_vec = [1,200,1,40]
value_lut = dict()
evaluation_lut = dict()

for problem_class in problem_vec:
    for solver_class,reruns in zip(solver_vec,solver_rerun_vec):
        value_lut[(problem_class,solver_class)] = []
        evaluation_lut[(problem_class,solver_class)] = []
        for experiment_idx in range(10):
            values = []
            evaluations = []
            fun_x_min = 1e99
            problem = problem_class()
            for rerun_idx in range(reruns):
                solver = solver_class(problem)
                if (solver_class == Scipy):
                    x,fun_x = solver.solve()
                    fun_x_min = min(fun_x,fun_x_min)
                    values.append(fun_x_min)
                    evaluations.append(problem.evaluations)
                else:
                    for i in range(solver.iterations):
                        retval = solver.step(i)
                        if retval is not None:
                            x,fun_x = retval
                        fun_x_min = min(fun_x,fun_x_min)
                        values.append(fun_x_min)
                        evaluations.append(problem.evaluations)
            value_lut[(problem_class,solver_class)].append(values)
            evaluation_lut[(problem_class,solver_class)].append(evaluations)

# plot results
fig,axes = plt.subplots(len(problem_vec))
for problem_class,ax in zip(problem_vec,axes):
    xlim = 1e99
    for solver_class in solver_vec:
        #breakpoint()
        value_mean = np.mean(value_lut[(problem_class,solver_class)], axis=0)
        evaluation_mean = np.mean(evaluation_lut[(problem_class,solver_class)], axis=0)
        value_std = np.std(value_lut[(problem_class,solver_class)], axis=0)
        evaluation_std = np.std(evaluation_lut[(problem_class,solver_class)], axis=0)
        ax.plot(evaluation_mean, value_mean,label=str(solver_class))
        ax.fill_between(evaluation_mean, value_mean-value_std, value_mean+value_std,alpha=0.4)
        xlim = min(evaluation_mean[-1],xlim)
        print(str(problem_class), str(solver_class),evaluation_mean[-1])
    ax.set_title(str(problem_class))
    #ax.set_xlim([0,xlim])
    #ax.set_xlim([0,1000])
plt.legend()
plt.show()
breakpoint()


