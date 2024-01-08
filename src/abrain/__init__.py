"""Docstring for abrain module/top-level package"""

import importlib.metadata

from ._cpp.phenotype import ANN as CppANN
from ._cpp.phenotype import CPPN as CppCPPN
from ._cpp.phenotype import Point as CppPoint
from ._python.ann import ANN as PyANN
from ._python.cppn import CPPN as PyCPPN
from ._python.point import Point as PyPoint
from .core.ann import plotly_render
from .core.config import Config
from .core.genome import Genome, GIDManager

try:  # pragma: no cover
    # __package__ allows for the case where __name__ is "__main__"
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

ANN = CppANN
CPPN = CppCPPN
Point = CppPoint


def use_pure_python(pure_python: bool = True):
    global ANN, CPPN, Point
    if pure_python:
        ANN = PyANN
        CPPN = PyCPPN
        Point = PyPoint
    else:
        ANN = CppANN
        CPPN = CppCPPN
        Point = CppPoint


__all__ = ['Genome', 'ANN', 'Config', 'Point', 'GIDManager', 'CPPN',
           'plotly_render']
