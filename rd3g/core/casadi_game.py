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


@dataclass
class CasadiGameContext:
    """ Game context with CasADi SX object.
    context: (n_s*N, T) Game context, changes between iteration, but constant within iteration. 
        This contains variables too expensive to AD. 
        e.g. path curvature at each player position.
        Correspond to k=0..T-1
    """
    context: np.ndarray
    _context_sx: dict = field(init=False, repr=False)
    _shape: tuple = field(init=False, repr=False)

    def __post_init__(self):
        self._context_sx = cas.SX.sym('context', *self.context.shape)
        self._shape = self.context.shape

    def set_context(self, value):
        assert value.shape == self._shape
        assert value.flags['F_CONTIGUOUS']
        self.context = value

    def get_context_np(self):
        """ Create current context as a flattened np array"""
        return np.asarray(self.context.flatten(), order='F')

    def get_context_sx(self):
        """ Get CasADi SX parameter for the context"""
        return self._context_sx


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
                    logger.warning(
                        f"{_field.name} has unsupported numpy type {val.dtype}"
                    )
            elif isinstance(val, float):
                double_param_list.append([val])
            elif isinstance(val, int):
                int_param_list.append([val])
            else:
                logger.warning(
                    f"{_field.name} has unsupported type {type(val)}")
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
                        double_offset:double_offset + val.size].reshape(
                            val.shape)
                    double_offset += val.size

                elif val.dtype == int:
                    param_dict[
                        _field.name] = int_param_sx[int_offset:int_offset +
                                                    val.size].reshape(
                                                        val.shape)
                    int_offset += val.size
                else:
                    logger.warning(
                        f"{_field.name} has unsupported numpy type {val.dtype}"
                    )
            elif isinstance(val, float):
                param_dict[_field.name] = double_param_sx[
                    double_offset:double_offset + 1]
                double_offset += 1
            elif isinstance(val, int):
                param_dict[_field.name] = int_param_sx[int_offset:int_offset +
                                                       1]
                int_offset += 1
            else:
                logger.warning(
                    f"{_field.name} has unsupported type {type(val)}")
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
        self.n_hi = config.n_hi
        self.n_c = config.n_c
        self.m = config.m
        self.x0 = config.x0
        c = config
        empty = np.zeros((c.n_c*c.N, c.T), order='F')
        self.context = CasadiGameContext(context=empty)

        x_k_i = cas.SX.sym('x_k_i', self.n, 1)
        u_k_i = cas.SX.sym('u_k_i', self.m, 1)
        i_onehot = cas.SX.sym('i_onehot', self.N, 1)
        state_i_k = cas.SX.sym('state_i_k', self.n_c, 1)
        f_args = [x_k_i, u_k_i, i_onehot, state_i_k]
        gc = self.config
        config_params = [gc.get_int_param_sx(), gc.get_double_param_sx()]
        x_next_val = self.f(*f_args)
        self.f_casadi = cas.Function('f', f_args + config_params, [x_next_val])

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

    def rollout(self, x0, u):
        """ Rollout control to get state trajectory. Casadi compatible
        Args:
            x0: (n,N)
            u: (m*N, T), u0..u_T-1
            context: (n_c*N, T)
        Return:
            X: (n*N, T) x1..xT
        """
        m = self.m
        N = self.N
        T = self.T
        eye = cas.DM.eye(self.N)

        x = x0
        X = []
        for k in range(T):
            x_next = []
            for i in range(N):
                xki = x[:, i]
                uki = cas.reshape(u[:, k], m, N)[:, i]
                contextki = self.get_context(xki)
                i_onehot = eye[:, i]
                xi_next = self.f(xki, uki, i_onehot, contextki)
                x_next.append(xi_next)
            x = cas.horzcat(*x_next)
            # (nN, 1)
            X.append(cas.vertcat(*x_next))

        return cas.horzcat(*X)

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
    def f(self, x_k_i, u_k_i, i_onehot, context_i_k):
        """ Dynamics function x_{t+1} = f(x_t,u,i)
        Args:
            x_k_i: (n,1) State for agent i
            u_k_i: (m,1) Control for agent i
            i_onehot: (N, 1) agent id, in one-hot encoding (N), i.e. i=1,N=4 -> [0,1,0,0], column vector
            context_i_k: (n_c,) Game context
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

    def get_context(self, x_k_i):
        """ Calculate game context for one agent at one step
        Args:
            x_k_i: (n, 1), cas.DM
        Return:
            Context vector for this state, (n_s, 10
        """
        del x_k_i
        return cas.SX.zeros(0, 1)
