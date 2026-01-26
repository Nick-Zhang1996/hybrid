""" Base class for Differential Dynamic Game Solver"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BaseSolverConfig():
    """Configs for Residual Game."""
    tolerance: float = 5e-4
    iterations: int = 30


@dataclass(frozen=True)
class Solution():
    """Solution to a Game, return type of BaseSolver"""
    elapsed_time: float = 0
    """ Total iterations run """
    iterations: int = 0
    u: Any = None
    """(m,N,T) Open-loop Nash policy"""
    x: Any = None
    """(n,N,T) State trajectory """
    residual: float = 0
    has_converged: bool = False
    is_optimal: bool = False


class BaseSolver(ABC):
    """Base class for DDG solver"""

    def __init__(self, config: BaseSolverConfig, game):
        self.config = config
        self.game = game

    def solve(self) -> Solution:
        """ Solve game. """
