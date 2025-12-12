""" CasADi infrastructure """
import logging
from dataclasses import dataclass, field, fields

import casadi as cas
import numpy as np

from rd3g.core.base_game import BaseGameConfig

logger = logging.getLogger('Casadi Game')
logger.setLevel(logging.INFO)


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
                    double_param_list.append(val.T.flatten())  # .T to convert to column major
                elif val.dtype == int:
                    int_param_list.append(val.T.flatten())
                else:
                    logger.error(f"Unsupported numpy type {val.dtype}")
            elif isinstance(val, float):
                double_param_list.append([val])
            elif isinstance(val, int):
                int_param_list.append([val])
            else:
                logger.error(f"Unsupported type {type(val)}")
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
                    logger.error(f"Unsupported numpy type {val.dtype}")
            elif isinstance(val, float):
                param_dict[_field.name] = double_param_sx[
                    double_offset:double_offset + 1]
                double_offset += 1
            elif isinstance(val, int):
                param_dict[_field.name] = int_param_sx[
                    int_offset:int_offset + 1]
                int_offset += 1
            else:
                logger.error(f"Unsupported type {type(val)}")
        object.__setattr__(self, '_param_dict', param_dict)

    def get_param(self, param_name):
        return self._param_dict[param_name]

    def get_int_param_np(self):
        return self._int_param_np

    def get_double_param_np(self):
        return self._double_param_np

    def get_int_param_sx(self):
        return self._int_param_sx

    def get_double_param_sx(self):
        return self._double_param_sx
