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
        """ Put int param and double param into SX variables"""
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
