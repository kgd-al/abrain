import logging

# flake8: noqa
from .._cpp.phenotype import (
    Point2D, Point3D,
    CPPN, CPPN2D, CPPN3D
)
from .._cpp.config import Config as _Config

logger = logging.getLogger(__name__)

CPPN2D.Output = _Config.ESHNOutputs
CPPN3D.Output = _Config.ESHNOutputs
