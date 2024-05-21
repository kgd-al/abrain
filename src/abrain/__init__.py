"""Docstring for abrain module/top-level package"""

import importlib.metadata

from .core.cppn import (
    Point2D, Point3D,
    CPPN, CPPN2D, CPPN3D)
from .core.ann import ANN2D, ANN3D

from .core.config import Config
from .core.genome import Genome

try:  # pragma: no cover
    # __package__ allows for the case where __name__ is "__main__"
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"


__all__ = ['Genome',
           'CPPN',
           'Point2D', 'CPPN2D', 'ANN2D',
           'Point3D', 'CPPN3D', 'ANN3D',
           'Config']
