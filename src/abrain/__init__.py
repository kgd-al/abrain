"""Docstring for abrain module/top-level package"""

from .core.config import Config
from .core.genome import Genome, GIDManager

from .core.ann import ANN
from .core.ann import plotly_render
from ._cpp.phenotype import Point

from ._cpp.phenotype import CPPN

import importlib.metadata
try:  # pragma: no cover
    # __package__ allows for the case where __name__ is "__main__"
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"


__all__ = ['Genome', 'ANN', 'Config', 'Point', 'GIDManager', 'CPPN',
           'plotly_render']
