from typing import Iterable

import numpy as np

from .config import Config


class Point:
    DIMENSIONS = Config.dimension

    def __init__(self, *coords):
        self._data = np.zeros(shape=(Point.DIMENSIONS,), dtype=float,)
        coords = coords[0] if len(coords) == 1 else coords
        assert len(coords) == Point.DIMENSIONS
        for i, c in enumerate(coords):
            self._set_i(i, c)

    @classmethod
    def null(cls): return Point([0 for _ in range(cls.DIMENSIONS)])

    def x(self): return self._data[0]
    def y(self): return self._data[1]
    def z(self): return self._data[2]
    def __iter__(self) -> Iterable[float]: return self._data.__iter__()

    def __eq__(self, other: 'Point') -> bool:
        return np.array_equal(self._data, other._data)

    def __hash__(self) -> int:
        return hash(tuple(self._data))

    def __repr__(self):
        return f"{__name__}.Point({type(self._data)}, {self._data})"

    def _set_i(self, i, v):
        self._data[i] = round(v, self.DIMENSIONS)

    def set(self, v):
        for i in range(self.DIMENSIONS):
            self._set_i(i, v)

    def __add__(self, other: 'Point') -> 'Point':
        return Point([v0 + v1 for v0, v1 in zip(self._data, other._data)])

    def __sub__(self, other: 'Point') -> 'Point':
        return Point([v0 - v1 for v0, v1 in zip(self._data, other._data)])

    def __mul__(self, value: float) -> 'Point':
        return Point([v * value for v in self._data])

    def length(self):
        return np.sqrt(np.sum(self._data**2))
