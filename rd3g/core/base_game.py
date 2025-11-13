""" Base class for Differential Dynamic Game Problem"""
from abc import ABC, abstractmethod
from typing import Any
from functools import lru_cache
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class BaseGameConfig():
    """ Base Class for game configuration"""
    T: int = 0
    """ horizon """
    dt: float = 0
    N: int = 0
    n: int = 0
    m: int = 0
    x0: Any = None


class BaseGame(ABC):
    """Base class for a Differential Dynamic Game Problem"""

    def __init__(self, config: BaseGameConfig):
        self.config = config
        self.dt = config.dt
        self.T = config.T
        self.N = config.N
        self.n = config.n
        self.m = config.m
        self.x0 = config.x0

    @abstractmethod
    def visualize(self, x0, u, x=None, show=True, save=False):
        """ Visualize the game with an image 
        Args:
            x0: initial state (N, n)
            u: control, (T, N, m)
            x: state trajectory (T,N,n), if None, calculate from x0 and 8
            show: if True, visualize with matplotlib
            save: if True, save as png
        Return:
            img: image in form of a uint8 array (H, W, 3), color: (R,G,B)
            """
        # fig = self._visualize(u, x)
        # filename = f'logs/{fig_name}.png'
        # plt.savefig(filename)
        # logger.info(f'saved figure to {filename}')

    @abstractmethod
    def animate(self, x0, u, x=None, show=False, save=False):
        """ Animate the game
        Args:
            x0: initial state (N, n)
            u: control, (T, N, m)
            x: state trajectory (T,N,n), if None, calculate from x0 and 8
            show: if True, visualize with matplotlib
        Return:
            img: list of images in form of a list of uint8 array (H, W, 3), color: (R,G,B)
            """

    def rollout(self, x0: np.ndarray, u: np.ndarray) -> np.ndarray:
        """ Rollout control to get state trajectory (cached)
        Args:
            x0: (N,m)
            u: (T,N,m), u0..u_T-1, will be reshaped
        Return:
            X: (T,N,n) x1..xT
        """
        # use cached version
        # x0_tuple = tuple(x0.flatten())
        # u_tuple = tuple(u.flatten())
        # return self._rollout_cached(x0_tuple, u_tuple)
        return self._rollout(x0, u)

    @lru_cache(maxsize=128)
    def _rollout_cached(self, x0: tuple, u: tuple):
        x0_np = np.array(x0)
        u_np = np.array(u).reshape((self.T, self.N, self.m))
        return self._rollout(x0_np, u_np)

    def _rollout(self, x0: np.ndarray, u: np.ndarray) -> np.ndarray:
        """ Rollout control to get state trajectory
        Args:
            x0: (N,m)
            u: (T,N,m), u0..u_T-1, will be reshaped
        Return:
            X: (T,N,n) x1..xT
        """
        assert u.shape == (self.T, self.N, self.m)
        # u = u.reshape(self.T, self.N, self.m)
        X = np.zeros((self.T + 1, self.N, self.n))
        X[0, :, :] = x0.reshape(self.N, self.n)
        # x+ = x + vx*dt + 0.5*ax*dt*dt
        # vx+ = vx + ax*dt
        for i in range(self.N):
            for k in range(1, self.T + 1):
                X[k, i] = self.f(X[k - 1, i], u[k - 1, i], i).flatten()
        return X[1:, :, :]

    def setup_rd3g_cpp(self, solver_config):
        """Load RD3G cpp backend for this game, return the cpp module."""
        del solver_config
        # subclass responsible for loading specific cpp/eigen module
        # and setting x0
        # logger.info(' ------------------------------------------------------------------------- ')
        # logger.info('This game does not have a custom cpp backend')
        # logger.info(' ------------------------------------------------------------------------- ')
        # self.cpp = ParticleGame(...)
        # self.cpp.set_x0(self.x0)
        return None

    @abstractmethod
    def J(self, x_k, u_k_i, i):
        '''
        Stage cost for an agent
        Args:
            x_k: (N,n) state for all agents at stage k
            u_k_i: (m) control for agent i at stage k
            i: agent index
        Return:
            val: (float) stage cost for agent i at stage k
        '''

    @abstractmethod
    def f(self, x, u, i):
        ''' Dynamics function x_{t+1} = f(x_t,u,i)
        Args:
            x: (n) State for agent i
            u: (m) Control for agent i
        Return:
            (n) The next state, progressed by self.dt
        '''

    @abstractmethod
    def h(self, x_i, x_j):
        """ Collision constraint between agent i and j, h <= 0
        Args:
            x_i: (n,) State of i at time k
            x_j: (n,) State of j at time k
        Return:
            val: float val<=0 when constraint satisfied
        """
