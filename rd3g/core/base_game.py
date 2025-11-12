""" Base class for Differential Dynamic Game Problem"""
from abc import ABC, abstractmethod

from dataclasses import dataclass


@dataclass(frozen=True)
class BaseGameConfig():
    """ Base Class for game configuration"""


class BaseGame(ABC):
    """Base class for a Differential Dynamic Game Problem"""
