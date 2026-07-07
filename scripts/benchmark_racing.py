"""Benchmark game solvers on the racing game."""
import logging

import numpy as np

from rd3g.games.car_racing_casadi import create_random_game
from rd3g.solvers.interior_point_game import InteriorPointGame, InteriorPointGameConfig
from rd3g.solvers.ilqgame import ILQGame, ILQGameConfig
from rd3g.solvers.rd3g_casadi import RD3GCasadi, RD3GCasadiConfig
from rd3g.solvers.algames_julia import AlgamesJulia, AlgamesJuliaConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('rd3g.solvers.interior_point_game')
logger.setLevel(logging.INFO)
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
            abs_split=False)
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
            iterations=20,
            step_size=0.5,
            barrier_weight=1e-2,
        )
        return ILQGame(solver_config, game, cpp_only=cpp)
    if solver_name == 'algames':
        if cpp:
            raise ValueError('algames does not support cpp=True')
        solver_config = AlgamesJuliaConfig(
            iterations=20,
            warmup_solve=True,
            persistent_worker=True,
            reuse_problem=False,
        )
        return AlgamesJulia(solver_config, game)
    raise ValueError(f'Unknown solver {solver_name!r}')


def reset_solver_for_game(solver, game):
    """Attach a new game sample and reset mutable per-solve state."""
    solver.game = game
    if hasattr(solver, 'x0'):
        solver.x0 = game.config.x0
    if hasattr(solver, 'reg'):
        solver.reg = solver.config.reg
    if hasattr(solver, 'rho'):
        solver.rho = solver.config.rho_0
    if hasattr(solver, 'line_search_fail_count'):
        solver.line_search_fail_count = 0
    if hasattr(solver, 'residual_vec'):
        solver.residual_vec = []
    if hasattr(solver, 'violation_vec'):
        solver.violation_vec = []
    if hasattr(solver, 'barrier_weight'):
        solver.barrier_weight = float(solver.config.barrier_weight)
    if hasattr(solver, 'last_policy'):
        solver.last_policy = None


def solve_with_initial_guess(solver, cpp, u_ref):
    if cpp:
        return solver.solve_cpp_backend(u_ref)
    if isinstance(solver, AlgamesJulia):
        return solver.solve()
    return solver.solve(u_ref)


def benchmark(solver_name, cpp, vne, precond, repeat_count=100, car_counts=range(2, 9), horizon=20):
    logger.info(f'Benchmarking {solver_name}')
    good_u_vec = []
    conv_mean_vec = []
    conv_var_vec = []
    optimal_mean_vec = []
    optimal_var_vec = []
    time_mean_vec = []
    time_var_vec = []
    converged_time_mean_vec = []
    converged_time_var_vec = []

    for car_count in car_counts:
        converge_vec = []
        optimal_vec = []
        dt_vec = []
        np.random.seed(0)
        game = create_random_game(car_count=car_count, horizon=horizon,
                                  variational_gne=vne)
        solver = create_solver(solver_name, game, cpp, vne, precond)
        if cpp:
            solver.init_cpp_backend()

        for i in range(repeat_count):
            np.random.seed(i)
            game = create_random_game(car_count=car_count, horizon=horizon,
                                      variational_gne=vne)
            reset_solver_for_game(solver, game)
            u_ref = game.initial_control_guess()
            sol = solve_with_initial_guess(solver, cpp, u_ref)
            # sol = solver.solve_cpp_backend_rand_restart(restarts=10)

            # solver.final()
            converge_vec.append(sol.has_converged)
            optimal_vec.append(sol.is_optimal and sol.has_converged)
            dt_vec.append(sol.elapsed_time)
            logger.debug(
                f'run {i}, {sol.iterations=}, {sol.elapsed_time=:.6f}, '
                f'{sol.residual=:.6f} {sol.has_converged=}, {sol.is_optimal=}')
            if sol.is_optimal and sol.has_converged:
                good_u_vec.append(sol.u)
            print(f'repeat {i + 1}/{repeat_count}', end='\r', flush=True)

            if i % 10 == 9:
                convergence_rate = np.mean(converge_vec)
                optimal_rate = np.mean(optimal_vec)
                median_dt = np.median(dt_vec)
                mean_dt = np.mean(dt_vec)
                logger.debug(
                    f'{convergence_rate=}, {optimal_rate=}, '
                    f'{mean_dt*1000=:.1f}ms, {median_dt*1000=:.1f}ms')
        if solver_name == 'algames':
            solver.final()
        print()
        converge = np.mean(converge_vec).item()
        optimal = np.mean(optimal_vec).item()
        converge_arr = np.asarray(converge_vec, dtype=bool)
        dt_arr = np.asarray(dt_vec)
        converged_dt_arr = dt_arr[converge_arr]
        # DEBUG: show histogram of runtime
        if False:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.hist(dt_arr * 1000, bins='auto', color='royalblue', edgecolor='black', alpha=0.8)
            ax.set_title(f'{solver_name} {car_count} cars solve times')
            ax.set_xlabel('dt (ms)')
            ax.set_ylabel('count')
            fig.tight_layout()
            # fig.savefig(f'/tmp/racing_dt_hist_{solver_name}_{car_count}_cars.png')
            plt.show()
            plt.close(fig)

        median_dt_ms = np.median(dt_arr).item() * 1000
        mean_dt_ms = np.mean(dt_arr).item() * 1000
        var_dt_ms = np.var(dt_arr*1000).item()
        converged_mean_dt_ms = np.mean(converged_dt_arr).item() * \
            1000 if converged_dt_arr.size else float('nan')
        converged_var_dt_ms = np.var(
            converged_dt_arr*1000).item() if converged_dt_arr.size else float('nan')
        time_mean_vec.append(mean_dt_ms)
        time_var_vec.append(var_dt_ms)
        converged_time_mean_vec.append(converged_mean_dt_ms)
        converged_time_var_vec.append(converged_var_dt_ms)
        conv_mean_vec.append(converge)
        optimal_mean_vec.append(optimal)
        conv_var_vec.append(np.var(converge_vec).item())
        optimal_var_vec.append(np.var(optimal_vec).item())
        logger.info(
            f'{car_count} cars {converge=}, {optimal=}, {mean_dt_ms=:.1f}ms, '
            f'{var_dt_ms=:.3f}ms, {converged_mean_dt_ms=:.1f}ms, '
            f'{converged_var_dt_ms=:.3f}ms')

    # Report mean and covariance of optimal results, used as param for initial guess
    # stacked_u = np.hstack([val.reshape(game.m, game.N*game.T, order='F') for val in good_u_vec])
    # mean = np.mean(stacked_u, axis=1)
    # cov = np.cov(stacked_u)
    # logger.info(f'{mean=}, {cov=}')
    print(f'{solver_name=}, {cpp=}, {vne=}, {precond=}')
    print(f'{time_mean_vec=}')
    print(f'{time_var_vec=}')
    print(f'{converged_time_mean_vec=}')
    print(f'{converged_time_var_vec=}')
    print(f'{conv_mean_vec=}')
    print(f'{conv_var_vec=}')
    print(f'{optimal_mean_vec=}')
    print(f'{optimal_var_vec=}')


def notify_done():
    """Say something to grep attention when running the script interactively."""
    try:
        import pyttsx3
    except ImportError:
        return
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.say('Solution Ready')
    engine.runAndWait()


if __name__ == '__main__':
    benchmark(solver_name='ipm', cpp=True, vne=True, precond=False)
    # benchmark(solver_name='ipm', cpp=True, vne=True, precond=False)
    # benchmark(solver_name='ipm', cpp=True, vne=True, precond=True)
    # benchmark(solver_name='algames', cpp=False, vne=True, precond=False)
    # benchmark(solver_name='ilqgame', cpp=False, vne=True, precond=False)
    notify_done()
