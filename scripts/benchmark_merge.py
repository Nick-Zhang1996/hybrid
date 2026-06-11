"""Benchmark the interior-point game solver on the merge game."""
import logging
import numpy as np

from rd3g.games.car_merge_kinematic_bicycle_casadi import create_random_game
from rd3g.solvers.interior_point_game import InteriorPointGame, InteriorPointGameConfig
from rd3g.solvers.ilqgame import ILQGame, ILQGameConfig
from rd3g.solvers.rd3g_casadi import RD3GCasadi, RD3GCasadiConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('rd3g.solvers.interior_point_game')
logger.setLevel(logging.WARNING)
logger = logging.getLogger('rd3g.solvers.ilqgame')
logger.setLevel(logging.ERROR)

logger = logging.getLogger('main')
logger.setLevel(logging.INFO)


def create_solver(solver_name, game, cpp, vne, precond):
    if solver_name == 'ipm':
        solver_config = InteriorPointGameConfig(
            inertia_correction=False,
            variational_gne=vne,
            precondition_with_potential=precond,
            abs_split=True)
        return InteriorPointGame(solver_config, game, cpp_only=cpp)
    if solver_name == 'rd3g':
        if vne:
            raise ValueError('rd3g_casadi does not support variational_gne=True')
        solver_config = RD3GCasadiConfig(inertia_correction=False)
        return RD3GCasadi(solver_config, game, cpp_only=cpp)
    if solver_name == 'ilqgame':
        if cpp:
            raise ValueError('ilqgame does not support cpp=True')
        solver_config = ILQGameConfig(
            variational_gne=vne,
            iterations=50,
            step_size=0.5,
            barrier_weight=1e-2,
        )
        return ILQGame(solver_config, game, cpp_only=cpp)
    raise ValueError(f'Unknown solver {solver_name!r}')


def benchmark(solver_name, cpp, vne, precond):
    good_u_vec = []
    conv_mean_vec = []
    conv_var_vec = []
    optimal_mean_vec = []
    optimal_var_vec = []
    time_mean_vec = []
    time_var_vec = []
    for car_count in range(2, 9):
        converge_vec = []
        optimal_vec = []
        dt_vec = []
        for i in range(100):
            np.random.seed(i)
            game = create_random_game(car_count=car_count, horizon=20, variational_gne=vne)
            solver = create_solver(solver_name, game, cpp, vne, precond)
            if cpp:
                solver.init_cpp_backend()
                sol = solver.solve_cpp_backend()
            else:
                sol = solver.solve()
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
        conv_mean_vec.append(converge)
        optimal_mean_vec.append(optimal)
        logger.info(
            f'{car_count} cars {converge=}, {optimal=}, {mean_dt_ms=:.1f}ms, {var_dt_ms=:.1f}ms, '
            f'{converged_mean_dt_ms=:.1f}ms, {converged_var_dt_ms=:.1f}ms')

    # Report mean and covariance of optimal results, used as param for initial guess
    # stacked_u = np.hstack([val.reshape(game.m, game.N*game.T, order='F') for val in good_u_vec])
    # mean = np.mean(stacked_u, axis=1)
    # cov = np.cov(stacked_u)
    # logger.info(f'{mean=}, {cov=}')
    print(f'{solver_name=}, {cpp=}, {vne=}, {precond=}')
    print(f'{time_mean_vec=}')
    print(f'{time_var_vec=}')
    print(f'{conv_mean_vec=}')
    print(f'{optimal_mean_vec=}')


if __name__ == '__main__':
    # benchmark(solver_name='ipm', cpp=True, vne=True, precond=True)
    benchmark(solver_name='ilqgame', cpp=False, vne=True, precond=True)
