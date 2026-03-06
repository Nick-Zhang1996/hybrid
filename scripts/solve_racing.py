""" Prototype to solve car racing game with RD3G CasADi """
import logging
import numpy as np

from buzzracer.tracks.track import TrackConfig
from buzzracer.tracks.nascar_track import NascarTrack

from rd3g.games.car_racing_casadi import CarRacingCasadi, CarRacingCasadiConfig
from rd3g.solvers.rd3g_casadi import RD3GCasadi, RD3GCasadiConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('rd3g.solvers.rd3g_casadi')
logger.setLevel(logging.DEBUG)

logger = logging.getLogger('main')
logger.setLevel(logging.INFO)

np.random.seed(5)
# np.random.seed()
cpp = True
game_config = CarRacingCasadiConfig(x0=np.zeros((4,3)))
n = game_config.n
N = game_config.N
m = game_config.m
T = game_config.T
track_config = TrackConfig()
track = NascarTrack(track_config)
game = CarRacingCasadi(game_config, track)
u = np.zeros((m,N,T))
x = np.zeros((n,N,T+1))
game.visualize(u, x)
