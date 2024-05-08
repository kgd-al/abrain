"""Docstring for abrain module/top-level package"""

import importlib.metadata

from ._cpp.phenotype import (CPPN,
                             Point2D, CPPN2D, ANN2D,
                             Point3D, CPPN3D, ANN3D)

# from .core.ann import plotly_render
from .core.config import Config
from .core.genome import Genome, GIDManager

try:  # pragma: no cover
    # __package__ allows for the case where __name__ is "__main__"
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"


__all__ = ['Genome', 'GIDManager',
           'CPPN',
           'Point2D', 'CPPN2D', 'ANN2D',
           'Point3D', 'CPPN3D', 'ANN3D',
           'Config']#,
           # 'plotly_render']
