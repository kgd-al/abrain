import itertools
import logging
import pprint
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Set, Optional, Callable, List

import numpy as np

from . import _functions
from .point import Point
from ..core.config import Config
from ..core.genome import Genome


class CPPN:
    class Output(Enum):
        WEIGHT = 0
        LEO = 1
        BIAS = 2

    @dataclass
    class _Link:
        weight: float
        node: 'CPPN._Node'

    @dataclass
    class _Node:
        name: str = field(default_factory=str)

        data: Optional[float] = None
        func: Optional[Callable[[float], float]] = None
        links: Optional[List['CPPN._Link']] = field(default_factory=list)

        @classmethod
        def from_string(cls, func: str, name):
            n = cls(func=_functions.functions[func])
            n.name = name
            return n

        def value(self, depth=0):
            if self.func is not None and self.data is None:
                self.data = self.func(np.sum([
                    link.weight * link.node.value(depth+1) for link in self.links
                ]))
            return self.data

    def __init__(self, genome: Genome):
        self._outputs_set = [CPPN.Output.WEIGHT, CPPN.Output.LEO, CPPN.Output.BIAS]
        self._n_inputs = len(Config.cppnInputNames)
        self._n_outputs = len(Config.cppnOutputNames)
        self._inputs = [self._Node(name=f"I{i}") for i in range(self._n_inputs)]
        self._outputs = [self._Node.from_string(func=Config.outputFunctions[o], name=f"O{o}")
                         for o in range(self._n_outputs)]
        self._hidden = [self._Node.from_string(func=n.func, name=f"H{n.id}") for n in genome.nodes]

        nodes = {i: n for i, n in enumerate(self._inputs + self._outputs)}
        nodes.update({n.id: h for h, n in zip(self._hidden, genome.nodes)})

        for link in genome.links:
            nodes[link.dst].links.append(
                self._Link(weight=link.weight, node=nodes[link.src]))

    def outputs(self): return np.zeros(len(self._outputs))

    def __call__(self, *args):
        if len(args) == 3:
            if isinstance(args[2], np.ndarray):
                return self._call_all(*args)
            else:
                return self._call_one(*args)
        else:
            return self._call_subset(*args)

    def _call_all(self, src: Point, dst: Point, outputs: np.ndarray) -> None:
        self._pre_evaluation(src, dst)
        for i in range(self._n_outputs):
            outputs[i] = self._outputs[i].value()

    def _call_one(self, src: Point, dst: Point, output: 'CPPN.Output') -> float:
        self._pre_evaluation(src, dst)
        return self._outputs[output.value].value()

    def _call_subset(self, src: Point, dst: Point, outputs: np.ndarray,
                     oset: Set['CPPN.Output']) -> None:
        self._pre_evaluation(src, dst)
        for o in oset:
            outputs[o.value] = self._outputs[o.value].value()

    __DIST_NORM = 2 * np.sqrt(2)

    def _pre_evaluation(self, src: Point, dst: Point):
        for i, c in enumerate(itertools.chain(src.__iter__(), dst.__iter__())):
            self._inputs[i].data = c
        self._inputs[-2].data = (src - dst).length() / CPPN.__DIST_NORM
        self._inputs[-1].data = 1

        for n in itertools.chain(self._hidden, self._outputs):
            n.data = None
