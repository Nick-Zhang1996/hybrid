""" Base class for Differential Dynamic Game Problem"""
from abc import ABC, abstractmethod

from dataclasses import dataclass


@dataclass(frozen=True)
class BaseGameConfig():
    """ Base Class for game configuration"""
    horizon: int = 0
    dt: float = 0
    N: int = 0
    n: int = 0
    m: int = 0


class BaseGame(ABC):
    """Base class for a Differential Dynamic Game Problem"""

    def visualize(self, x0, u, x=None, show=False):
        """ Visualize the game with an image 
        Args:
            x0: initial state (N, n)
            u: control, (T, N, m)
            x: state trajectory (T,N,n), if None, calculate from x0 and 8
            show: if True, visualize with matplotlib
        Return:
            img: image in form of a uint8 array (H, W, 3), color: (R,G,B)
            """

    def animate(self, x0, u, x=None, show=False):
        """ Animate the game
        Args:
            x0: initial state (N, n)
            u: control, (T, N, m)
            x: state trajectory (T,N,n), if None, calculate from x0 and 8
            show: if True, visualize with matplotlib
        Return:
            img: list of images in form of a list of uint8 array (H, W, 3), color: (R,G,B)
            """

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

    def f(self, x, u, i):
        ''' Dynamics function x_{t+1} = f(x_t,u,i)
        Args:
            x: (n,) State for agent i
            u: (m,) Control for agent i
        Return:
            (n,) The next state, progressed by self.dt

        this problem has homogeneous agents, so [i] is irrelevant'''
