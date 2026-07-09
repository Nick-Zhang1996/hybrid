"""Solve the air traffic control game from every runway-assignment guess."""
import itertools
import logging

import numpy as np

from rd3g.games.air_traffic_control_casadi import create_random_game
from rd3g.solvers.interior_point_game import InteriorPointGame, InteriorPointGameConfig

logging.basicConfig(level=logging.INFO)
solver_logger = logging.getLogger('rd3g.solvers.interior_point_game')
solver_logger.setLevel(logging.DEBUG)

logger = logging.getLogger('main')
logger.setLevel(logging.INFO)


def _wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def create_runway_assignment_guess(game, runway_assignment):
    """Crudely guide each aircraft to its assigned runway, independently."""
    config = game.config
    if len(runway_assignment) != config.N:
        raise ValueError('runway_assignment must contain one runway per aircraft')

    state = np.array(config.x0, dtype=float, copy=True)
    u_ref = np.zeros((config.m, config.N, config.T), dtype=float, order='F')

    for k in range(config.T):
        for aircraft_idx, runway_idx in enumerate(runway_assignment):
            if not 0 <= runway_idx < config.runway_count:
                raise ValueError(f'invalid runway index {runway_idx}')

            position = state[:2, aircraft_idx]
            runway_position = config.runway_positions[:, runway_idx]
            delta = runway_position - position
            runway_heading = config.runway_headings[runway_idx, 0]
            runway_axis = np.array([
                np.cos(runway_heading),
                np.sin(runway_heading),
            ])
            past_threshold = np.dot(position - runway_position, runway_axis) >= 0.0
            desired_heading = (
                runway_heading
                if past_threshold
                else np.arctan2(delta[1], delta[0])
            )

            speed = state[2, aircraft_idx]
            heading = state[3, aircraft_idx]
            acceleration = np.clip(
                (config.V_min - speed) / config.dt,
                -config.a_max,
                config.a_max,
            )
            turn_rate = np.clip(
                _wrap_angle(desired_heading - heading) / config.dt,
                -config.omega_max,
                config.omega_max,
            )
            u_ref[:, aircraft_idx, k] = [acceleration, turn_rate]

            # Roll out this aircraft alone; these guesses intentionally ignore traffic.
            state[0, aircraft_idx] += speed * np.cos(heading) * config.dt
            state[1, aircraft_idx] += speed * np.sin(heading) * config.dt
            state[2, aircraft_idx] += acceleration * config.dt
            state[3, aircraft_idx] += turn_rate * config.dt

    return np.asfortranarray(u_ref.reshape(config.m * config.N, config.T, order='F'))


def solve_all_runway_assignments(solver, cpp=False):
    """Solve all A**N runway assignments and return the least-residual solution."""
    config = solver.game.config
    assignments = itertools.product(range(config.runway_count), repeat=config.N)
    scenario_count = config.runway_count**config.N
    best_solution = None
    best_assignment = None
    best_residual = np.inf
    scenario_results = []

    for scenario_idx, assignment in enumerate(assignments, start=1):
        logger.info(
            'Solving runway assignment %d/%d: %s',
            scenario_idx,
            scenario_count,
            assignment,
        )
        u_ref = create_runway_assignment_guess(solver.game, assignment)
        solution = (
            solver.solve_cpp_backend(u_ref)
            if cpp
            else solver.solve(u_ref)
        )
        residual = float(solution.residual)
        scenario_results.append((assignment, solution))
        if best_solution is None or (
                np.isfinite(residual) and residual < best_residual):
            best_solution = solution
            best_assignment = assignment
            best_residual = residual

    logger.info('Runway assignment results:')
    for assignment, solution in scenario_results:
        logger.info(
            '  assignment=%s residual=%.6f optimal=%s converged=%s',
            assignment,
            float(solution.residual),
            solution.is_optimal,
            solution.has_converged,
        )
    logger.info(
        'Best runway assignment: %s, residual=%.6f',
        best_assignment,
        best_solution.residual,
    )
    return best_solution


def main():
    np.random.seed(3)
    cpp = False
    variational_gne = True

    game = create_random_game(
        aircraft_count=3,
        horizon=20,
        variational_gne=variational_gne,
    )
    solver_config = InteriorPointGameConfig(
        inertia_correction=False,
        variational_gne=variational_gne,
        precondition_with_potential=True,
        abs_split=False,
        iterations=30,
    )
    solver = InteriorPointGame(solver_config, game, cpp_only=cpp)

    if cpp:
        solver.init_cpp_backend()
    sol = solve_all_runway_assignments(solver, cpp=cpp)

    logger.info(
        '%s, %s, %s, %s, %s',
        f'sol.iterations={sol.iterations}',
        f'sol.elapsed_time={sol.elapsed_time:.6f}',
        f'sol.residual={sol.residual:.6f}',
        f'sol.is_optimal={sol.is_optimal}',
        f'sol.has_converged={sol.has_converged}',
    )
    solver.visualize(sol.u)
    solver.final()


if __name__ == '__main__':
    main()
