import dataclasses
import logging
import pprint
import random
import time
from collections import namedtuple
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import _evolvable_substrate
from ._functions import functions
from .cppn import CPPN
from .point import Point
from .config import Config


logger = logging.getLogger(__name__)


class ANN:
    @dataclass
    class Stats:
        axons: float = -1
        depth: np.uint = -1
        hidden: np.uint = -1
        edges: np.uint = -1
        density: float = -1
        iterations: np.uint = -1
        time: dict = dataclasses.field(
            default_factory=lambda: dict(build=-1, eval=-1))

        def dict(self): return dataclasses.asdict(self)

    def __init__(self):
        self._matrix: Optional[np.ndarray] = None
        self._inputs_idx, self._outputs_idx = {}, {}
        self._inputs, self._hidden, self._outputs = None, None, None
        self.NI, self.NH, self.NO = 0, 0, 0
        self._stats = ANN.Stats()
        self._act_function = functions[Config.activation_function]

    def reset(self):
        self._hidden[:] = 0

    def __repr__(self):
        return "Pure-Python ANN"

    def neuron_at(self, p: Point):
        raise NotImplementedError("Though I really should")

    def ibuffer(self): return self._inputs
    def obuffer(self): return self._outputs
    def buffers(self): return self.ibuffer(), self.obuffer()

    def __call__(self, ibuffer, obuffer, substeps: int = 1):
        start = _time()

        # print(ibuffer)
        assert ibuffer is self._inputs
        # print(self._inputs.shape, self._hidden.shape, self._matrix.shape)
        output = self._matrix.dot(np.concatenate(([1], self._inputs, self._hidden)))
        output = self._act_function(output)
        # print(output.shape)
        # print(output)
        self._hidden[:] = output[:self.NH]
        assert obuffer is self._outputs
        self._outputs[:] = output[self.NH:]

        self._stats.time['eval'] += _time_diff(start)

    def stats(self): return self._stats

    def empty(self, strict: bool = False):
        if strict:
            return self._stats.hidden == 0
        else:
            return self._stats.edges == 0

    def perceptron(self):
        return self.empty(strict=True) and not self.empty(strict=False)

    @classmethod
    def build(cls, inputs, outputs, genome):
        """
        Build an ANN

        :param inputs:
        :param outputs:
        :param genome:
        :return:
        """
        start = _time()

        cppn = CPPN(genome)
        ann = ANN()

        hidden, connections, iterations = (
            _evolvable_substrate.connect(cppn, inputs, outputs))

        ann.NI, ann.NH, ann.NO = len(inputs), len(hidden), len(outputs)

        ann._inputs = np.zeros(shape=(ann.NI,))
        ann._hidden = np.zeros(shape=(ann.NH,))
        ann._outputs = np.zeros(shape=(ann.NO,))

        ann._matrix = np.zeros(shape=(ann.NH+ann.NO, 1+ann.NI+ann.NH, ))

        for i, p in enumerate(inputs + hidden):
            ann._inputs_idx[p] = i+1

        for i, p in enumerate(hidden + outputs):
            ann._outputs_idx[p] = i
            ann._matrix[ann._outputs_idx[p], 0] = cppn(p, Point.null(),
                                                       CPPN.Output.BIAS)

        # print(f"[kgd-debug:{__name__}] hidden neurons: {pprint.pformat(hidden)}")

        for c in connections:
            ann._matrix[ann._outputs_idx[c.dst], ann._inputs_idx[c.src]] = c.weight

        ann._compute_stats(connections)
        ann._stats.iterations = iterations
        ann._stats.time['build'] = _time_diff(start)
        ann._stats.time['eval'] = 0
        return ann

    def _compute_stats(self, connections):
        self._stats.edges = len(connections)
        self._stats.axons = sum([(c.dst - c.src).length() for c in connections])

        self._stats.hidden = self.NH
        self._stats.density = self._stats.edges
        if self.NH == 0:
            self._stats.depth = 1
            self._stats.density /= self.NI * self.NO
        else:
            self._stats.depth = self._compute_depth(connections)
            self._stats.density /= (self.NI * self.NH + self.NH * self.NO)

    def _compute_depth(self, connections):
        return -1


def _time(): return time.perf_counter()
def _time_diff(start): return _time() - start
