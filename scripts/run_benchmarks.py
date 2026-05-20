"""Run merge benchmarks for a few solver configurations and save the aggregates."""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rd3g.games.car_merge_kinematic_bicycle_casadi import create_random_game
from rd3g.solvers.interior_point_game import InteriorPointGame, InteriorPointGameConfig
from rd3g.solvers.rd3g_casadi import RD3GCasadi, RD3GCasadiConfig

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


DEFAULT_OUTPUT_PATH = ROOT_DIR / "outputs" / "merge_ipm_benchmark.pkl"


logging.basicConfig(level=logging.INFO)
logging.getLogger("rd3g.solvers.interior_point_game").setLevel(logging.WARNING)
logging.getLogger("rd3g.solvers.rd3g_casadi").setLevel(logging.WARNING)
LOGGER = logging.getLogger("merge_benchmark")


@dataclass(frozen=True)
class BenchmarkSpec:
    """Solver settings for one benchmark line in the plot."""

    label: str
    solver_name: str
    cpp_backend: bool = True
    variational_gne: bool = False
    precondition_with_potential: bool = False
    abs_split: bool = False


BENCHMARK_SPECS = [
    BenchmarkSpec(
        label="RD3G CasADi",
        solver_name="rd3g_casadi",
        variational_gne=False,
    ),
    BenchmarkSpec(
        label="IPM (no abs split)",
        solver_name="ipm",
        variational_gne=True,
        precondition_with_potential=True,
        abs_split=False,
    ),
    BenchmarkSpec(
        label="IPM (abs split)",
        solver_name="ipm",
        variational_gne=True,
        precondition_with_potential=True,
        abs_split=True,
    ),
]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of random seeds to evaluate for each car count.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=20,
        help="Planning horizon passed to create_random_game().",
    )
    parser.add_argument(
        "--min-cars",
        type=int,
        default=2,
        help="Smallest car count to benchmark.",
    )
    parser.add_argument(
        "--max-cars",
        type=int,
        default=8,
        help="Largest car count to benchmark.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Where to save the pickle file.",
    )
    return parser.parse_args()


def create_solver(spec: BenchmarkSpec, game):
    """Create the requested solver instance for one random game."""
    if spec.solver_name == "ipm":
        solver_config = InteriorPointGameConfig(
            inertia_correction=False,
            variational_gne=spec.variational_gne,
            precondition_with_potential=spec.precondition_with_potential,
            abs_split=spec.abs_split,
        )
        return InteriorPointGame(solver_config, game, cpp_only=spec.cpp_backend)

    if spec.solver_name == "rd3g_casadi":
        if spec.variational_gne:
            raise ValueError("RD3G CasADi does not support variational_gne=True.")
        solver_config = RD3GCasadiConfig(inertia_correction=False)
        return RD3GCasadi(solver_config, game, cpp_only=spec.cpp_backend)

    raise ValueError(f"Unknown solver {spec.solver_name!r}")


def solve_once(
    spec: BenchmarkSpec,
    *,
    car_count: int,
    horizon: int,
    seed: int,
):
    """Solve one random merge game and return the solution object."""
    np.random.seed(seed)
    game = create_random_game(
        car_count=car_count,
        horizon=horizon,
        variational_gne=spec.variational_gne,
    )
    solver = create_solver(spec, game)

    if spec.cpp_backend:
        try:
            solver.init_cpp_backend()
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "Failed to load the compiled C++ backend from build/lib. "
                f"Set cpp_backend=False for {spec.label!r}, or rebuild the "
                "extensions for this Python version."
            ) from exc
        return solver.solve_cpp_backend()

    return solver.solve()


def benchmark_configuration(
    spec: BenchmarkSpec,
    *,
    car_counts: range,
    runs: int,
    horizon: int,
) -> dict:
    """Benchmark one solver configuration across all requested car counts."""
    runtime_mean = []
    runtime_var = []
    conv_mean = []
    optimal_mean = []

    for car_count in car_counts:
        converged = []
        optimal = []
        elapsed_times = []

        for seed in range(runs):
            solution = solve_once(
                spec,
                car_count=car_count,
                horizon=horizon,
                seed=seed,
            )
            converged.append(solution.has_converged)
            optimal.append(solution.has_converged and solution.is_optimal)
            elapsed_times.append(solution.elapsed_time)

        elapsed_times_array = np.asarray(elapsed_times, dtype=float)
        converged_array = np.asarray(converged, dtype=bool)
        optimal_array = np.asarray(optimal, dtype=bool)

        mean_runtime_ms = float(np.mean(elapsed_times_array) * 1000.0)
        # Mirror the scaling used by the existing benchmark scripts and plot.
        var_runtime_ms = float(np.var(elapsed_times_array) * 1000.0)
        convergence_rate = float(np.mean(converged_array))
        optimal_rate = float(np.mean(optimal_array))

        runtime_mean.append(mean_runtime_ms)
        runtime_var.append(var_runtime_ms)
        conv_mean.append(convergence_rate)
        optimal_mean.append(optimal_rate)

        LOGGER.info(
            "%s | %d cars | conv=%.2f | optimal=%.2f | mean=%.1f ms | var=%.4f",
            spec.label,
            car_count,
            convergence_rate,
            optimal_rate,
            mean_runtime_ms,
            var_runtime_ms,
        )

    return {
        "label": spec.label,
        "solver_name": spec.solver_name,
        "cpp_backend": spec.cpp_backend,
        "variational_gne": spec.variational_gne,
        "precondition_with_potential": spec.precondition_with_potential,
        "abs_split": spec.abs_split,
        "runtime_mean": np.asarray(runtime_mean, dtype=float),
        "runtime_var": np.asarray(runtime_var, dtype=float),
        "conv_mean": np.asarray(conv_mean, dtype=float),
        "optimal_mean": np.asarray(optimal_mean, dtype=float),
    }


def benchmark_all_configurations(
    args: argparse.Namespace,
    *,
    car_counts: range,
    horizon: int,
) -> list[dict]:
    """Benchmark all configured solvers sequentially."""
    return [
        benchmark_configuration(
            spec,
            car_counts=car_counts,
            runs=args.runs,
            horizon=horizon,
        )
        for spec in BENCHMARK_SPECS
    ]


def save_results(
    output_path: Path,
    *,
    car_counts: range,
    configs: list[dict],
    runs: int,
    horizon: int,
) -> None:
    """Save all benchmark results to a pickle file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cars": np.asarray(list(car_counts), dtype=float),
        "configs": configs,
        "metadata": {
            "runs": runs,
            "horizon": horizon,
        },
    }
    with output_path.open("wb") as file:
        pickle.dump(payload, file)


def main() -> None:
    """Run the requested benchmark sweep."""
    args = parse_args()
    if args.min_cars > args.max_cars:
        raise ValueError("--min-cars must be less than or equal to --max-cars.")
    if args.runs <= 0:
        raise ValueError("--runs must be positive.")

    car_counts = range(args.min_cars, args.max_cars + 1)
    configs = benchmark_all_configurations(
        args,
        car_counts=car_counts,
        horizon=args.horizon,
    )

    save_results(
        args.output,
        car_counts=car_counts,
        configs=configs,
        runs=args.runs,
        horizon=args.horizon,
    )
    print(f"Saved benchmark results to {args.output}")


if __name__ == "__main__":
    main()
