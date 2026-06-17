"""Python adapter for the Algames Julia benchmark backend."""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rd3g.core.base_solver import BaseSolver, BaseSolverConfig, Solution
from rd3g.utilities.util import BASEDIR


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AlgamesJuliaConfig(BaseSolverConfig):
    """Configuration for the Julia Algames benchmark adapter."""

    tolerance: float = 5e-4
    iterations: int = 20
    line_search_max_iter: int = 10
    outer_iterations: int = 1
    rho_0: float = 1.0
    rho_trial: float = 1.0
    rho_increase: float = 10.0
    rho_max: float = 1e7
    seed: int = 100
    amplitude_init: float = 1e-8
    julia_cmd: str | None = None
    compiled_modules: bool = True
    warmup_solve: bool = True
    persistent_worker: bool = True
    reuse_problem: bool = False
    depot_path: str | None = None
    extra_julia_args: tuple[str, ...] = field(default_factory=tuple)
    keep_files: bool = False


class AlgamesJulia(BaseSolver):
    """Call the benchmark implementations in ``extern/AlgamesDriving.jl``."""

    def __init__(self, config: AlgamesJuliaConfig, game):
        super().__init__(config, game)
        self.N = self.game.config.N
        self.T = self.game.config.T
        self.n = self.game.config.n
        self.m = self.game.config.m
        self.last_solution: Solution | None = None
        self.backend_script = (
            Path(BASEDIR)
            / "extern"
            / "AlgamesDriving.jl"
            / "bin"
            / "rd3g_benchmark_cli.jl"
        )
        self._worker: subprocess.Popen[str] | None = None
        self._worker_dir: tempfile.TemporaryDirectory[str] | None = None
        self._worker_counter = 0
        self._stderr_thread: threading.Thread | None = None

    def solve(self) -> Solution:
        """Solve the currently attached game with the Julia backend."""
        payload = self._build_payload()
        result = self._call_julia(payload)
        residual = result.get("residual")
        residual = float("inf") if residual is None else float(residual)
        u = result.get("u")
        x = result.get("x")
        if u:
            u = np.asarray(u, dtype=float)
        else:
            u = None
        if x:
            x = np.asarray(x, dtype=float)
        else:
            x = None

        if not result.get("ok", False):
            logger.warning("Algames Julia backend failed: %s", result.get("error", "unknown error"))

        solution = Solution(
            elapsed_time=float(result.get("elapsed_time", 0.0) or 0.0),
            iterations=int(result.get("iterations", 0) or 0),
            u=u,
            x=x,
            residual=residual,
            has_converged=bool(result.get("has_converged", False)),
            is_optimal=bool(result.get("is_optimal", False)),
        )
        self.last_solution = solution
        return solution

    def _rollout_full_x(self, u_ref, x_ref=None):
        """Return a full state trajectory for visualization."""
        u_ref = np.asarray(u_ref, dtype=float).reshape((self.m, self.N, self.T), order="F")
        if x_ref is None:
            if self.last_solution is None or self.last_solution.x is None:
                raise ValueError("x_ref is required before AlgamesJulia.solve() has been called.")
            x_ref = self.last_solution.x
        x_ref = np.asarray(x_ref, dtype=float)
        if x_ref.shape == (self.n, self.N, self.T + 1):
            return x_ref
        x_ref = x_ref.reshape((self.n, self.N, self.T), order="F")
        return np.dstack([self.game.config.x0[:, :, np.newaxis], x_ref])

    def visualize(self, u_ref, x_ref=None, save=False, show=True):
        """Visualize an Algames solution through the underlying game renderer."""
        u_ref = np.asarray(u_ref, dtype=float).reshape((self.m, self.N, self.T), order="F")
        full_x = self._rollout_full_x(u_ref, x_ref)
        return self.game.visualize(u_ref, full_x, show=show, save=save)

    def animate(self, u_ref, x_ref=None, save_gif=False, save_snapshots=False):
        """Animate an Algames solution through the underlying game renderer."""
        u_ref = np.asarray(u_ref, dtype=float).reshape((self.m, self.N, self.T), order="F")
        full_x = self._rollout_full_x(u_ref, x_ref)
        return self.game.animate(
            u_ref, full_x, show=True, save_gif=save_gif, save_snapshots=save_snapshots
        )

    def final(self):
        """Compatibility hook for scripts that call solver.final()."""
        self.close()
        return None

    def close(self):
        """Stop the persistent Julia worker, if one is running."""
        worker = self._worker
        self._worker = None
        if worker is not None and worker.poll() is None:
            try:
                assert worker.stdin is not None
                worker.stdin.write("EXIT\n")
                worker.stdin.flush()
                worker.wait(timeout=5)
            except Exception:
                worker.kill()
                worker.wait(timeout=5)
        if self._worker_dir is not None:
            self._worker_dir.cleanup()
            self._worker_dir = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _build_payload(self) -> dict[str, Any]:
        cfg = self.game.config
        payload: dict[str, Any] = {
            "problem": self._problem_name(),
            "T": int(cfg.T),
            "dt": float(cfg.dt),
            "N": int(cfg.N),
            "n": int(cfg.n),
            "m": int(cfg.m),
            "variational_gne": bool(getattr(cfg, "variational_gne", False)),
            "x0": self._array(cfg.x0),
            "target_x_ref": self._array(cfg.target_x_ref),
            "J_R": self._array(cfg.J_R),
            "collision_radius": float(getattr(cfg, "collision_radius", 0.0)),
            "tolerance": float(self.config.tolerance),
            "iterations": int(self.config.iterations),
            "line_search_max_iter": int(self.config.line_search_max_iter),
            "outer_iterations": int(self.config.outer_iterations),
            "rho_0": float(self.config.rho_0),
            "rho_trial": float(self.config.rho_trial),
            "rho_increase": float(self.config.rho_increase),
            "rho_max": float(self.config.rho_max),
            "seed": int(self.config.seed),
            "amplitude_init": float(self.config.amplitude_init),
            "warmup_solve": bool(self.config.warmup_solve),
            "reuse_problem": bool(self.config.reuse_problem),
        }

        if payload["problem"] in {"merge", "racing"}:
            payload["J_Qr"] = self._array(cfg.J_Qr)
        elif payload["problem"] == "intersection":
            payload["J_Qr_diag_vec"] = self._array(cfg.J_Qr_diag_vec)

        if payload["problem"] == "racing":
            payload.update(self._racing_track_payload())

        return payload

    def _problem_name(self) -> str:
        cls_name = self.game.__class__.__name__
        if cls_name == "CarMergeKinematicBicycleCasadi":
            return "merge"
        if cls_name == "IntersectionCasadi":
            return "intersection"
        if cls_name == "CarRacingCasadi":
            return "racing"
        raise ValueError(f"Unsupported Algames benchmark game class {cls_name!r}")

    def _racing_track_payload(self) -> dict[str, Any]:
        cfg = self.game.config
        track_data = self.game.track.data
        s_vec = np.asarray(track_data.s_vec, dtype=float)
        curvature_vec = np.asarray(track_data.curvature_vec, dtype=float)
        left_vec = np.asarray(track_data.left_width_vec, dtype=float) - float(cfg.bdry_margin)
        right_vec = np.asarray(track_data.right_width_vec, dtype=float) - float(cfg.bdry_margin)

        max_s = s_vec[-1]
        track_s = np.hstack([s_vec[:-1] - max_s, s_vec, s_vec[1:] + max_s])
        track_curvature = np.hstack([curvature_vec[:-1], curvature_vec, curvature_vec[1:]])
        track_left = np.hstack([left_vec[:-1], left_vec, left_vec[1:]])
        track_right = np.hstack([right_vec[:-1], right_vec, right_vec[1:]])

        car_param = self.game.car_param
        return {
            "track_s": self._array(track_s),
            "track_curvature": self._array(track_curvature),
            "track_left_width": self._array(track_left),
            "track_right_width": self._array(track_right),
            "car_lf": float(car_param.lf),
            "car_lr": float(car_param.lr),
        }

    def _call_julia(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.config.persistent_worker:
            return self._call_julia_worker(payload)
        return self._call_julia_once(payload)

    def _call_julia_once(self, payload: dict[str, Any]) -> dict[str, Any]:
        julia_cmd = self._resolve_julia_cmd()
        if not self.backend_script.exists():
            raise FileNotFoundError(f"Missing Algames backend script: {self.backend_script}")

        if self.config.keep_files:
            workdir = Path(tempfile.mkdtemp(prefix="rd3g_algames_"))
            cleanup = None
        else:
            cleanup = tempfile.TemporaryDirectory(prefix="rd3g_algames_")
            workdir = Path(cleanup.name)

        try:
            input_path = workdir / "input.jl"
            output_path = workdir / "output.json"
            input_path.write_text(
                "RD3G_ALGAMES_INPUT = " + self._julia_literal(payload) + "\n",
                encoding="utf-8",
            )

            cmd = [julia_cmd, *self.config.extra_julia_args]
            if not self.config.compiled_modules:
                cmd.append("--compiled-modules=no")
            cmd.extend([str(self.backend_script), str(input_path), str(output_path)])
            env = os.environ.copy()
            depot_path = self.config.depot_path or self._default_depot_path()
            if depot_path:
                env["JULIA_DEPOT_PATH"] = depot_path

            completed = subprocess.run(
                cmd,
                cwd=BASEDIR,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "Algames Julia backend failed with exit code "
                    f"{completed.returncode}.\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
                )
            return json.loads(output_path.read_text(encoding="utf-8"))
        finally:
            if cleanup is not None:
                cleanup.cleanup()
            elif self.config.keep_files:
                logger.info("Kept Algames Julia files in %s", workdir)

    def _call_julia_worker(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._ensure_worker()
        assert self._worker is not None
        assert self._worker.stdin is not None
        assert self._worker.stdout is not None
        assert self._worker_dir is not None

        self._worker_counter += 1
        workdir = Path(self._worker_dir.name)
        input_path = workdir / f"input_{self._worker_counter}.jl"
        output_path = workdir / f"output_{self._worker_counter}.json"
        input_path.write_text(
            "RD3G_ALGAMES_INPUT = " + self._julia_literal(payload) + "\n",
            encoding="utf-8",
        )

        self._worker.stdin.write(f"SOLVE\t{input_path}\t{output_path}\n")
        self._worker.stdin.flush()
        response = self._worker.stdout.readline().strip()
        if response != "OK":
            raise RuntimeError(f"Algames Julia worker returned {response!r}")
        return json.loads(output_path.read_text(encoding="utf-8"))

    def _ensure_worker(self):
        if self._worker is not None and self._worker.poll() is None:
            return
        if self._worker is not None:
            self.close()

        julia_cmd = self._resolve_julia_cmd()
        if not self.backend_script.exists():
            raise FileNotFoundError(f"Missing Algames backend script: {self.backend_script}")

        self._worker_dir = tempfile.TemporaryDirectory(prefix="rd3g_algames_worker_")
        cmd = [julia_cmd, *self.config.extra_julia_args]
        if not self.config.compiled_modules:
            cmd.append("--compiled-modules=no")
        cmd.extend([str(self.backend_script), "--server"])
        env = os.environ.copy()
        depot_path = self.config.depot_path or self._default_depot_path()
        if depot_path:
            env["JULIA_DEPOT_PATH"] = depot_path

        self._worker = subprocess.Popen(
            cmd,
            cwd=BASEDIR,
            env=env,
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
        )
        self._stderr_thread = threading.Thread(
            target=self._drain_worker_stderr,
            args=(self._worker,),
            daemon=True,
        )
        self._stderr_thread.start()

        assert self._worker.stdout is not None
        ready = self._worker.stdout.readline().strip()
        if ready != "READY":
            self.close()
            raise RuntimeError(f"Algames Julia worker failed to start; got {ready!r}")

    @staticmethod
    def _drain_worker_stderr(worker: subprocess.Popen[str]):
        if worker.stderr is None:
            return
        for line in worker.stderr:
            logger.debug("Algames Julia worker: %s", line.rstrip())

    @staticmethod
    def _array(value: Any) -> Any:
        return np.asarray(value, dtype=float).tolist()

    @staticmethod
    def _default_depot_path() -> str | None:
        home_depot = Path.home() / ".julia"
        if home_depot.exists():
            return f"/tmp/rd3g_julia_depot:{home_depot}"
        return None

    @staticmethod
    def _find_julia_cmd() -> str:
        juliaup_dir = Path.home() / ".julia" / "juliaup"
        candidates = [
            path for path in sorted(juliaup_dir.glob("julia-*/bin/julia"))
            if path.is_file() and os.access(path, os.X_OK)
        ]
        if candidates:
            return str(candidates[-1])
        candidates = [
            path for path in sorted(juliaup_dir.glob("julia-*/*/julia"))
            if path.is_file() and os.access(path, os.X_OK)
        ]
        if candidates:
            return str(candidates[-1])
        julia = shutil.which("julia")
        if julia is None:
            raise FileNotFoundError(
                "Could not find Julia. Set AlgamesJuliaConfig(julia_cmd=...) "
                "to the Julia executable."
            )
        return julia

    def _resolve_julia_cmd(self) -> str:
        if self.config.julia_cmd is not None:
            return self.config.julia_cmd
        return self.__class__._find_julia_cmd()

    @classmethod
    def _julia_literal(cls, value: Any) -> str:
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, dict):
            items = [
                f"{cls._julia_literal(str(key))} => {cls._julia_literal(val)}"
                for key, val in value.items()
            ]
            return "Dict{String,Any}(" + ", ".join(items) + ")"
        if isinstance(value, (list, tuple)):
            return "Any[" + ", ".join(cls._julia_literal(item) for item in value) + "]"
        if isinstance(value, str):
            return json.dumps(value)
        if isinstance(value, (bool, np.bool_)):
            return "true" if bool(value) else "false"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            number = float(value)
            if math.isnan(number):
                return "NaN"
            if math.isinf(number):
                return "Inf" if number > 0 else "-Inf"
            return repr(number)
        if value is None:
            return "nothing"
        raise TypeError(f"Cannot serialize {type(value).__name__} to a Julia literal")
