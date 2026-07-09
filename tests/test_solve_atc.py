from types import SimpleNamespace

import numpy as np

from rd3g.core.base_solver import Solution
from scripts.solve_atc import (
    create_runway_assignment_guess,
    solve_all_runway_assignments,
)


def _game(aircraft_count=2, runway_count=2):
    config = SimpleNamespace(
        N=aircraft_count,
        m=2,
        T=4,
        dt=1.0,
        V_min=70.0,
        a_max=0.3,
        omega_max=0.1,
        runway_count=runway_count,
        runway_positions=np.array([[0.0, 0.0], [0.0, 300.0]]),
        runway_headings=np.array([[0.0], [-0.35]]),
        x0=np.array([
            [-500.0, -500.0],
            [0.0, 300.0],
            [80.0, 80.0],
            [0.3, -0.3],
        ]),
    )
    return SimpleNamespace(config=config)


def test_runway_assignment_guess_has_solver_shape_and_order():
    game = _game()

    guess = create_runway_assignment_guess(game, (0, 1))

    assert guess.shape == (game.config.m * game.config.N, game.config.T)
    assert np.isfortran(guess)
    assert np.all(np.abs(guess[1::2]) <= game.config.omega_max)


def test_solve_all_runway_assignments_returns_least_residual(caplog):
    class FakeSolver:
        def __init__(self):
            self.game = _game()
            self.calls = []

        def solve(self, u_ref):
            self.calls.append(u_ref)
            return Solution(residual=float(5 - len(self.calls)))

    solver = FakeSolver()

    solution = solve_all_runway_assignments(solver)

    assert len(solver.calls) == solver.game.config.runway_count**solver.game.config.N
    assert solution.residual == 1.0
    summary_rows = [
        record.message for record in caplog.records
        if record.message.startswith('  assignment=')
    ]
    assert len(summary_rows) == 4
    assert all('optimal=False converged=False' in row for row in summary_rows)
