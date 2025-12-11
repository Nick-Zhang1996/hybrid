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

    def __post_init__(self):
        """ Put int param and double param into SX variables"""
        int_param_list = []
        double_param_list = []
        for _field in fields(self):
            if _field.contains('param'):
                logger.debug(f'skipping {_field}')
                continue
            val = getattr(self, _field)
            if isinstance(val, np.ndarray):
                if val.dtype is np.float:
                    double_param_list.append(val.flatten())
                elif val.dtype is np.int:
                    int_param_list.append(val.flatten())
                else:
                    logger.error(f"Unsupported numpy type {val.dtype}")
            elif isinstance(val, float):
                double_param_list.append([val])
            elif isinstance(val, int):
                int_param_list.append([val])
            else:
                logger.error(f"Unsupported type {type(val)}")
        int_param = np.flatten(int_param_list, order='F')
        double_param = np.flatten(double_param_list, order='F')
        # object.__setattr__(self, '_int_param', int_param)
        # object.__setattr__(self, '_double_param', double_param)
        int_param_sx = cas.SX(int_param)
        double_param_sx = cas.SX(double_param)
        int_offset = 0
        double_offset = 0
        param_dict = {}
        for _field in fields(self):
            if _field.contains('param'):
                logger.debug(f'skipping {_field}')
                continue
            val = getattr(self, _field)
            if isinstance(val, np.ndarray):
                if val.dtype is np.float:
                    param_dict[_field] = double_param_sx[
                        double_offset:double_offset + val.size].reshape(val.shape)
                    double_offset += val.size

                elif val.dtype is np.int:
                    param_dict[_field] = int_param_sx[
                        int_offset:int_offset + val.size].reshape(val.shape)
                    int_offset += val.size
                else:
                    logger.error(f"Unsupported numpy type {val.dtype}")
            elif isinstance(val, float):
                param_dict[_field] = double_param_sx[
                    double_offset:double_offset + 1]
                double_offset += 1
            elif isinstance(val, int):
                param_dict[_field] = int_param_sx[
                    int_offset:int_offset + 1]
                int_offset += 1
            else:
                logger.error(f"Unsupported type {type(val)}")
        object.__setattr__(self, '_param_dict', param_dict)
