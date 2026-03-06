""" Car Racing game, CasADi version """
import os
import logging
from typing import Any
from math import degrees
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.animation import FuncAnimation
import casadi as cas

from buzzracer.tracks.curvilinear_track import CurvilinearTrack

from rd3g.utilities.util import BASEDIR, resolve_logname
from rd3g.core.base_casadi_game import CasadiGameConfig
from rd3g.core.base_jax_game import BaseGame

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass(frozen=True)
class CarRacingCasadiConfig(CasadiGameConfig):
    """ Base Class for game configuration. """
    T: int = 0
    dt: float = 0.1
    N: int = 3
    n: int = 4
    m: int = 2
    n_hi: int = 0  # Total number of constraints for EACH agent, e.g. pairwise collision only: N*T

    collision_radius: float = 2.0  # TODO use BuzzRacer size
    """ Minimum distance between two cars"""

    x0: Any = None
    """ Initial state for all agents, dim: (n,N)"""
    J_R: Any = None
    """ Cost matrix for control effort dim: (m,m)"""

    def __post_init__(self):
        assert self.x0.shape == (self.n, self.N), (
            'Incorrect self.x0 dimension, '
            f'should be {(self.n, self.N)}, but got {self.x0.shape}')
        assert self.J_R.shape == (self.m, self.m)
        return super().__post_init__()


class CarRacingCasadi(BaseGame):
    ''' Car Racing Game with Kinematic Bicycle Model, with CasADi
        u = [steering, throttle]
        x = [x,y,v,theta]: x: upwards, y:right, theta: ccw (right hand coord)
        agent count: N, time step: 1..T+1
        X (game state) = concatenated state, first by agent, then by time)
        state p of agent i at time k: X[k,i,p] or X.flatten()[k*N*m + i*m + p]
        U (control) = concatenated control  dim: T*N*m
        control p of agent i at time k: U[k,i,p] or U.flatten()[k*N*m + i*m + p]
        x_i_k: 1..T, T*N*n  NOTE starts from 1
        u_i_k: 0..T-1, T*N*m
        lamda_i_k: 0..T-1 T*N*n
        mu_k_i_j: 1..T T*N*N NOTE starts from 1
    '''

    def __init__(self, config: CarRacingCasadiConfig, track: CurvilinearTrack):
        super().__init__(config)

        # n_hi is a new concept
        self.n_hi = config.n_hi
        # bounds for visualization
        self.visual_x_lim = [-10, 10]
        self.visual_y_lim = [-10, 10]

        self.car_scale = 0.0045 / 2
        color_names = [
            'purple', 'yellow', 'red', 'green', 'orange', 'pink', 'cyan',
            'hot_pink'
        ]
        self.car_img_vec = [
            mpimg.imread(
                os.path.join(BASEDIR, 'rd3g', 'resources', f'porsche_{color}.png'))
            for color in color_names
        ]
