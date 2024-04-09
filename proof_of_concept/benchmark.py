import matplotlib.pyplot as plt
from Problem import *
from Solver import *

problem_vec = [ParabolaWithSineNoise,PerlinNoise,ParabolaWithSineNoise2D]
solver_vec = [CEM, GradientDescent,Newton,Hybrid]
solver_rerun_vec = [5,200,40,1]
value_lut = dict()
evaluation_lut = dict()

for problem_class in problem_vec:
    for solver_class,reruns in zip(solver_vec,solver_rerun_vec):
        value_lut[(problem_class,solver_class)] = []
        evaluation_lut[(problem_class,solver_class)] = []
        for experiment_idx in range(100):
            values = []
            evaluations = []
            fun_x_min = 1e99
            problem = problem_class()
            for rerun_idx in range(reruns):
                solver = solver_class(problem)
                for i in range(solver.iterations):
                    x,fun_x = solver.step(i)
                    values.append(min(fun_x,fun_x_min))
                    fun_x_min = min(fun_x)
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
    #ax.set_xlim([0,xlim])
    ax.set_xlim([0,2000])
plt.legend()
plt.show()


