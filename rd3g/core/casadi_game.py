""" Base class for CasADi Game """
# BaseGame is obselete and this will be the new BaseGame after refactor

import logging
from dataclasses import dataclass, field, fields
from abc import ABC, abstractmethod

import casadi as cas
import numpy as np

from rd3g.core.base_game import BaseGameConfig

logger = logging.getLogger('Casadi Game')
logger.setLevel(logging.INFO)


# Unused
@dataclass
class CasadiGameState:
    """ Game state with CasADi support. Similar to CasadiGameConfig, but mutable"""
    _param_dict: dict = field(init=False, repr=False)
    _double_param_np: dict = field(init=False, repr=False)
    _double_param_sx: dict = field(init=False, repr=False)
    _param_shape_dict: dict = field(init=False, repr=False)

    def __post_init__(self):
        double_param = self.get_double_param_np()
        double_param_sx = cas.SX.sym('double_param', len(double_param))
        self._double_param_sx = double_param_sx
        double_offset = 0
        param_dict = {}
        param_shape_dict = {}
        for _field in fields(self):
            if 'param' in _field.name:
                logger.debug(f'skipping {_field.name}')
                continue
            val = getattr(self, _field.name)
            if isinstance(val, np.ndarray):
                if val.dtype == float:
                    param_dict[_field.name] = double_param_sx[
                        double_offset:double_offset + val.size].reshape(val.shape)
                    param_shape_dict[_field.name] = val.shape
                    double_offset += val.size
                else:
                    logger.warning(f"{_field.name} has unsupported numpy type {val.dtype}")
            else:
                logger.warning(f"{_field.name} has unsupported type {type(val)}")

        self._param_dict = param_dict
        self._param_shape_dict = param_shape_dict

    def __setattr__(self, name, value):
        """ Set new numerical value to param. do necessary checks"""
        if name.startswith('_'):
            super().__setattr__(name, value)
            return
        assert name in self._param_dict.keys()
        assert value.shape == self._param_shape_dict[name]
        assert value.flags['C_CONTIGUOUS']
        super().__setattr__(name, value)

    def get_double_param_np(self):
        """ Create a flattened np array of all double params with current value"""
        double_param_list = []
        for _field in fields(self):
            if 'param' in _field.name:
                logger.debug(f'skipping {_field.name}')
                continue
            val = getattr(self, _field.name)
            if isinstance(val, np.ndarray):
                if val.dtype == float:
                    double_param_list.append(val.flatten(order='F'))
                else:
                    logger.warning(f"{_field.name} has unsupported numpy type {val.dtype}")
            else:
                logger.warning(f"{_field.name} has unsupported type {type(val)}")
        double_param = np.asarray(np.hstack(double_param_list), order='F')
        return double_param

    def get_double_param_sx(self):
        """ Get flattened CasADi SX parameter of all double params"""
        return self._double_param_sx

    def get_param(self, param_name):
        """ Get CasADi SX parameter by name """
        return self._param_dict[param_name]


@dataclass(frozen=True)
class CasadiGameConfig(BaseGameConfig):
    """ Game config with support for CasADi compatible flat param vector"""
    # _int_param: list = field(init=False, repr=False)
    # _double_param: list = field(init=False, repr=False)
    _param_dict: dict = field(init=False, repr=False)
    _int_param_np: dict = field(init=False, repr=False)
    _double_param_np: dict = field(init=False, repr=False)
    _int_param_sx: dict = field(init=False, repr=False)
    _double_param_sx: dict = field(init=False, repr=False)

    def __post_init__(self):
        """ Put int param and double param into SX variables.
        This solves the issue of passing parameters to casadi functions.
        casadi functions need access to constant problem-specific parameters like
        x0, x_ref, cost matrices, collision radius, etc.
        However we want to maintain identical signature across problems.
        So we need to pass a config struct to all casadi functions.
        Since casadi doesn't understand structs, we concatenate all params into a flat vector,
        and send this big flat param vector to all casadi functions.
        We provide member method .get_param(PARAM_NAME) to retrieve the specific param,
        sliced and reshaped from the big param. 
        This provides the SX object for constructing a symbolic function.
        We provide get_XX_param_sx() to retrieve the flat param symbotic vector in SX.
        We provide get_XX_param_np() to retrieve the flat param numerical vector in np array.
        There are two param vectors, one for double, one for int
        """

        int_param_list = []
        double_param_list = []
        for _field in fields(self):
            if 'param' in _field.name:
                logger.debug(f'skipping {_field.name}')
                continue
            val = getattr(self, _field.name)
            if isinstance(val, np.ndarray):
                if val.dtype == float:
                    double_param_list.append(val.flatten(order='F'))
                elif val.dtype == int:
                    int_param_list.append(val.flatten(order='F'))
                else:
                    logger.warning(f"{_field.name} has unsupported numpy type {val.dtype}")
            elif isinstance(val, float):
                double_param_list.append([val])
            elif isinstance(val, int):
                int_param_list.append([val])
            else:
                logger.warning(f"{_field.name} has unsupported type {type(val)}")
        int_param = np.asarray(np.hstack(int_param_list), order='F')
        double_param = np.asarray(np.hstack(double_param_list), order='F')
        object.__setattr__(self, '_int_param_np', int_param)
        object.__setattr__(self, '_double_param_np', double_param)
        int_param_sx = cas.SX.sym('int_param', len(int_param))
        double_param_sx = cas.SX.sym('double_param', len(double_param))
        object.__setattr__(self, '_int_param_sx', int_param_sx)
        object.__setattr__(self, '_double_param_sx', double_param_sx)
        int_offset = 0
        double_offset = 0
        param_dict = {}
        for _field in fields(self):
            if 'param' in _field.name:
                logger.debug(f'skipping {_field.name}')
                continue
            val = getattr(self, _field.name)
            if isinstance(val, np.ndarray):
                if val.dtype == float:
                    param_dict[_field.name] = double_param_sx[
                        double_offset:double_offset + val.size].reshape(val.shape)
                    double_offset += val.size

                elif val.dtype == int:
                    param_dict[_field.name] = int_param_sx[
                        int_offset:int_offset + val.size].reshape(val.shape)
                    int_offset += val.size
                else:
                    logger.warning(f"{_field.name} has unsupported numpy type {val.dtype}")
            elif isinstance(val, float):
                param_dict[_field.name] = double_param_sx[
                    double_offset:double_offset + 1]
                double_offset += 1
            elif isinstance(val, int):
                param_dict[_field.name] = int_param_sx[
                    int_offset:int_offset + 1]
                int_offset += 1
            else:
                logger.warning(f"{_field.name} has unsupported type {type(val)}")
        object.__setattr__(self, '_param_dict', param_dict)

    def get_param(self, param_name):
        """ Get sliced CasADi parameter from config"""
        return self._param_dict[param_name]

    def get_int_param_np(self):
        return self._int_param_np

    def get_double_param_np(self):
        return self._double_param_np

    def get_int_param_sx(self):
        return self._int_param_sx

    def get_double_param_sx(self):
        return self._double_param_sx


class CasadiGame(ABC):
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
    def visualize(self, u, x, show=True, save=False):
        """ Visualize the game with given initial state (x0) and control (u) in a single frame.
        Args:
            u: (m,N,T)
            x: (n, N, T+1)
            show: if True, visualize with matplotlib
            save: if True, save as png
        """

    @abstractmethod
    def animate(self, u, x, show=True, save_gif=False, save_snapshots=False):
        """ Animate the game with given initial state (x0) and control (u).
        Args:
            u: (m,N,T)
            x: (n, N, T+1)
            show: if True, visualize with matplotlib
            save_gif:
            save_snapshots:
        """

    def F(self, x_k, u_k):
        """ Dynamics for all agents
        Args:
            x_k: (n,N)
            u_k: (m,N)
        Return:
            x_k_next: (n,N)
        """
        x_k_next_vec = []
        for i in range(self.N):
            i_onehot = cas.SX.eye(self.N)[:, i]
            x_k_next_vec.append(self.f(x_k[:, i], u_k[:, i], i_onehot))
        retval = cas.horzcat(*x_k_next_vec)
        assert retval.shape == (self.n, self.N)
        return retval

    def rollout(self, x0, u):
        """ Rollout control to get state trajectory, casadi compatible
        Args:
            x0: (n,N)
            u: (m*N, T), u0..u_T-1
        Return:
            X: (n*N, T) x1..xT
        """
        assert u.shape == (self.m*self.N, self.T)
        assert x0.shape == (self.n, self.N)

        x_k = cas.SX.sym('x_k_', (self.n*self.N))
        u_k = cas.SX.sym('u_k_', (self.m*self.N))
        x_k_next = cas.vec(self.F(cas.reshape(x_k, self.n, self.N),
                                  cas.reshape(u_k, self.m, self.N)))
        config_params = [self.config.get_int_param_sx(), self.config.get_double_param_sx()]
        config_param_repmat = [cas.repmat(param, 1, self.T) for param in config_params]
        accum_fun = cas.Function('accum_fun', [x_k, u_k]+config_params, [x_k_next, 0])
        rollout_fun = accum_fun.mapaccum(self.T)

        X, _ = rollout_fun(cas.vec(x0), u, *config_param_repmat)
        assert X.shape == (self.n*self.N, self.T)
        return X

    @abstractmethod
    def J(self, x_k, u_k_i, i_onehot):
        """
        Stage cost for an agent
        Args:
            x_k: (n,N) state for all agents at stage k
            u_k_i: (m, 1) control for agent i at stage k
            i_onehot: (N,1) agent id in one-hot encoding, i.e. i=1,N=4 -> [0,1,0,0], column vector
        Return:
            val: stage cost for agent i at stage k
        """

    @abstractmethod
    # pylint: disable-next=arguments-renamed
    def Jfi(self, x_T, i_onehot):
        """ Final cost"""
        return self.J(x_T, cas.SX.zeros(self.m), i_onehot)

    @abstractmethod
    def f(self, x, u, i):
        """ Dynamics function x_{t+1} = f(x_t,u,i)
        Args:
            x_k_i: (n,1) State for agent i
            u_k_i: (m,1) Control for agent i
            i_onehot: agent id, in one-hot encoding (N), i.e. i=1,N=4 -> [0,1,0,0], column vector
        Return:
            (n,1) The next state, progressed by self.dt
        """

    @abstractmethod
    def h(self, x, u):
        """ Inequality constraint function.
        h() is a mapping from (x,u) to all constraints.
        Args:
            x: (n*N,T), states, casadi.SX symbolic variable
            u: (m*N,T), controls, casadi.SX symbolic variable
        Returns:
            h_vec: (n_hi, N), constraints vector, sadisfied when h_vec <= 0
        """
