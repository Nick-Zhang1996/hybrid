"""Benchmark the interior-point game solver on the intersection game."""
import numpy as np  # Only needed if you want to calculate percentiles
import matplotlib.pyplot as plt
import logging
import numpy as np

from rd3g.games.intersection_casadi import create_random_game
from rd3g.solvers.interior_point_game import InteriorPointGame, InteriorPointGameConfig
from rd3g.solvers.rd3g_casadi import RD3GCasadi, RD3GCasadiConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('rd3g.solvers.interior_point_game')
logger.setLevel(logging.WARNING)

logger = logging.getLogger('main')
logger.setLevel(logging.INFO)

SOLVER = 'ipm'
good_u_vec = []
conv_mean_vec = []
conv_var_vec = []
optimal_mean_vec = []
optimal_var_vec = []
time_mean_vec = []
time_var_vec = []
converged_time_mean_vec = []
converged_time_var_vec = []
sol_vec = []


def create_solver(game, cpp):
    if SOLVER == 'ipm':
        solver_config = InteriorPointGameConfig(inertia_correction=False)
        return InteriorPointGame(solver_config, game, cpp_only=cpp)
    if SOLVER == 'rd3g_casadi':
        solver_config = RD3GCasadiConfig(inertia_correction=False)
        return RD3GCasadi(solver_config, game, cpp_only=cpp)
    raise ValueError(f'Unknown solver {SOLVER!r}')


for car_count in range(2, 9):
    converge_vec = []
    optimal_vec = []
    dt_vec = []
    for i in range(100):
        np.random.seed(i)
        game = create_random_game(car_count=car_count, horizon=20)
        solver = create_solver(game, cpp)
        if cpp:
            solver.init_cpp_backend()
            sol = solver.solve_cpp_backend()
        else:
            sol = solver.solve()
        sol_vec.append(sol)
        # sol = solver.solve_cpp_backend_rand_restart(restarts=10)

        # solver.final()
        converge_vec.append(sol.has_converged)
        optimal_vec.append(sol.is_optimal and sol.has_converged)
        dt_vec.append(sol.elapsed_time)
        logger.debug(
            f'run {i}, {sol.iterations=}, {sol.elapsed_time=:.6f}, {sol.residual=:.6f} {sol.has_converged=}, {sol.is_optimal=}')
        if sol.is_optimal and sol.has_converged:
            good_u_vec.append(sol.u)

        if i % 10 == 9:
            convergence_rate = np.mean(converge_vec)
            optimal_rate = np.mean(optimal_vec)
            median_dt = np.median(dt_vec)
            mean_dt = np.mean(dt_vec)
            logger.debug(
                f'{convergence_rate=}, {optimal_rate=}, {mean_dt*1000=:.1f}ms, {median_dt*1000=:.1f}ms')
    converge_arr = np.asarray(converge_vec, dtype=bool)
    dt_arr = np.asarray(dt_vec)
    converged_dt_arr = dt_arr[converge_arr]

    converge = np.mean(converge_arr).item()
    optimal = np.mean(optimal_vec).item()
    median_dt_ms = np.median(dt_arr).item() * 1000
    mean_dt_ms = np.mean(dt_arr).item() * 1000
    var_dt_ms = np.var(dt_arr).item() * 1000
    converged_mean_dt_ms = np.mean(converged_dt_arr).item() * \
        1000 if converged_dt_arr.size else float('nan')
    converged_var_dt_ms = np.var(converged_dt_arr).item() * \
        1000 if converged_dt_arr.size else float('nan')
    time_mean_vec.append(mean_dt_ms)
    time_var_vec.append(var_dt_ms)
    converged_time_mean_vec.append(converged_mean_dt_ms)
    converged_time_var_vec.append(converged_var_dt_ms)
    conv_mean_vec.append(converge)
    optimal_mean_vec.append(optimal)
    conv_var_vec.append(np.var(converge_vec).item())
    optimal_var_vec.append(np.var(optimal_vec).item())
    logger.info(
        f'{car_count} cars {converge=}, {optimal=}, {mean_dt_ms=:.1f}ms, {var_dt_ms=:.1f}ms, '
        f'{converged_mean_dt_ms=:.1f}ms, {converged_var_dt_ms=:.1f}ms')

# Report mean and covariance of optimal results, used as param for initial guess
# stacked_u = np.hstack([val.reshape(game.m, game.N*game.T, order='F') for val in good_u_vec])
# mean = np.mean(stacked_u, axis=1)
# cov = np.cov(stacked_u)
# logger.info(f'{mean=}, {cov=}')
print(f'{time_mean_vec=}')
print(f'{time_var_vec=}')
print(f'{converged_time_mean_vec=}')
print(f'{converged_time_var_vec=}')
print(f'{conv_mean_vec=}')
print(f'{conv_var_vec=}')
print(f'{optimal_mean_vec=}')
print(f'{optimal_var_vec=}')

# check solution iteration count

# 1. Filter the vector for converged solutions and extract the iteration counts
converged_iterations = [sol.iterations for sol in sol_vec]

# 2. Plot the histogram
plt.figure(figsize=(8, 5))
# Adjust 'bins' based on the spread of your iterations (e.g., 'auto' or a specific number like 20)
plt.hist(converged_iterations, bins='auto', color='royalblue', edgecolor='black', alpha=0.8)

# 3. Add formatting for readability
plt.title('Distribution of Iterations ', fontsize=14)
plt.xlabel('Number of Iterations', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 4. Optional: Draw a vertical line for the 95th percentile to help pick a default
percentile_95 = np.percentile(converged_iterations, 95)
plt.axvline(percentile_95, color='red', linestyle='dashed', linewidth=2,
            label=f'95th Percentile ({percentile_95:.0f})')
plt.legend()

# Display the plot
plt.show()
