import collections
import itertools
import pprint
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set

import numpy as np

from .config import NeuronType as T
from .cppn import CPPN
from .point import Point
from ..core.config import Config

O = CPPN.Output


@dataclass
class Connection:
    src: Point
    dst: Point
    weight: float

    def __hash__(self): return hash((self.src, self.dst))
    def __eq__(self, other): return self.src == other.src and self.dst == other.dst


Points = Set[Point]
Connections = Set[Connection]


def connect(cppn: CPPN, inputs: List[Point], outputs: List[Point]) \
        -> Tuple[List[Point], List[Connection], int]:

    assert len(inputs[0]._data) == 3, f"2D implementation is not yet ready"

    sio = collections.Counter(inputs + outputs)
    assert all(v == 1 for v in sio.values()), f"Duplicate coordinates: {pprint.pformat(sio)}"

    s_hidden = set()
    connections = set()

    # Get neurons/connections from the input neurons
    for p in inputs:
        tree = _division_and_initialisation(cppn, p, True)
        new_hiddens, new_connections = set(), set()
        _prune_and_extract(cppn, p, tree, True, new_connections)
        _collect(new_connections, connections, new_hiddens, s_hidden)

    converged = False
    unexplored_hidden = set(s_hidden)

    iteration = 0
    while iteration < Config.iterations and not converged:
        new_hiddens, new_connections = set(), set()
        for p in unexplored_hidden:
            tree = _division_and_initialisation(cppn, p, True)
            _prune_and_extract(cppn, p, tree, True, new_connections)
            _collect(new_connections, connections, new_hiddens, s_hidden)

        unexplored_hidden = new_hiddens
        converged = (len(unexplored_hidden) == 0)

        iteration += 1

    for p in outputs:
        new_connections = set()
        tree = _division_and_initialisation(cppn, p, False)
        _prune_and_extract(cppn, p, tree, False, new_connections)
        connections.update(new_connections)

    hiddens = _remove_unconnected_neurons(inputs, outputs, s_hidden, connections)

    if len(hiddens) == 0 and Config.allowPerceptrons:
        _generate_perceptron(cppn, inputs, outputs, connections)

    return list(hiddens), list(connections), iteration


@dataclass
class _QOTreeNode:
    center: Point
    radius: float
    level: int
    weight: Optional[float]
    cs: list['_QOTreeNode']

    def __init__(self, p: Point, r: float, lvl: int, w: float):
        self.center = p
        self.radius = r
        self.level = lvl
        self.weight = w
        self.cs = []

    def variance(self):
        return 0 if len(self.cs) == 0 else np.var([c.weight for c in self.cs])


def _division_and_initialisation(cppn: CPPN, point: Point, out: bool) -> _QOTreeNode:
    dimension = point.DIMENSIONS
    initial_depth = Config.initialDepth
    max_depth = Config.maxDepth
    div_thr = Config.divThr

    root = _QOTreeNode(Point.null(), 1, 1, np.nan)
    queue = collections.deque()
    queue.append(root)

    def _weight(p0, p1): return cppn(p0, p1, O.WEIGHT)
    def weight(p): return _weight(point, p) if out else _weight(p, point)

    while len(queue) > 0:
        node: _QOTreeNode = queue.popleft()
        center = node.center
        hr = .5 * node.radius
        nl = node.level + 1

        node.cs = [
            _QOTreeNode(c, hr, nl, weight(c))
            for d in itertools.product([-1, 1], repeat=dimension)
            if (c := center + Point(d) * hr) is not None
        ]

        if node.level < initial_depth or \
                (node.level < max_depth and node.variance() > div_thr):
            for c in node.cs:
                queue.append(c)

    return root


def _prune_and_extract(cppn: CPPN, point: Point,
                       tree: _QOTreeNode, out: bool,
                       con: Connections):
    dimension = point.DIMENSIONS
    var_thr = Config.varThr
    bnd_thr = Config.bndThr

    def _leo(p0, p1): return cppn(p0, p1, O.LEO) > .5
    def leo(p): return _leo(point, p) if out else _leo(point, p)

    for c in tree.cs:
        if c.variance() >= var_thr:
            # More information at lower resolution -> explore
            _prune_and_extract(cppn, point, c, out, con)
        else:
            # Not enough information at lower resolution -> test if part of band
            r = c.radius

            def d_weight(*coords):
                src = point if out else Point(coords)
                dst = Point(coords) if out else point
                return np.fabs(c.weight - cppn(src, dst, O.WEIGHT))

            if dimension == 2:
                cx, cy = c.center
                bnd = max(
                    min(d_weight(cx-r, cy), d_weight(cx+r, cy)),
                    min(d_weight(cx, cy-r), d_weight(cx, cy+r))
                )
            else:
                cx, cy, cz = c.center
                bnd = max(
                    min(d_weight(cx-r, cy, cz), d_weight(cx+r, cy, cz)),
                    min(d_weight(cx, cy-r, cz), d_weight(cx, cy+r, cz)),
                    min(d_weight(cx, cy, cz-r), d_weight(cx, cy, cz+r))
                )

            if bnd > bnd_thr and leo(c.center) and c.weight != 0:
                con.add(Connection(
                    point if out else c.center,
                    c.center if out else point,
                    c.weight
                ))


def _collect(new_connections: Connections, connections: Connections,
             new_hiddens: Points, hiddens: Points):
    for c in new_connections:
        s = len(hiddens)
        hiddens.add(c.dst)
        if len(hiddens) == s + 1:
            new_hiddens.add(c.dst)
    connections.update(new_connections)


def _generate_perceptron(cppn: CPPN, inputs: List[Point], outputs: List[Point],
                         connections: Connections):
    os = [O.WEIGHT, O.LEO]
    res = cppn.outputs()

    connections.clear()
    for i in inputs:
        for o in outputs:
            cppn(i, o, res, os)
            if res[O.LEO.value]:
                connections.add(Connection(i, o, res[O.WEIGHT.value]))


def _remove_unconnected_neurons(inputs: List[Point], outputs: List[Point],
                                hiddens: Points,
                                connections: Connections) -> Points:
    @dataclass
    class N:
        p: Point
        t: T = T.HIDDEN
        i: List['L'] = field(init=False, repr=False, default_factory=list)
        o: List['L'] = field(init=False, repr=False, default_factory=list)

        def __hash__(self): return p.__hash__()
        def __eq__(self, other): return self.p == other.p

    @dataclass
    class L:
        n: N
        w: float

    nodes = dict()
    i_nodes, o_nodes = set(), set()
    for i_lst, o_lst, t in [(inputs, i_nodes, T.INPUT), (outputs, o_nodes, T.OUTPUT)]:
        for p in i_lst:
            n = N(p, t)
            o_lst.add(n)
            nodes[p] = n

    def get(_p: Point):
        if (_n := nodes.get(_p, None)) is None:
            _n = N(_p)
            nodes[_p] = _n
        return _n

    for c in connections:
        src, dst = get(c.src), get(c.dst)
        src.o.append(L(dst, c.weight))
        dst.i.append(L(src, c.weight))

    def bfs(_nodes: Set[N], n_field: str) -> Set[N]:
        queue = collections.deque()
        seen, dest = set(), set()
        for _n in _nodes:
            queue.append(_n)
            seen.add(_n)

        while len(queue) > 0:
            _n = queue.popleft()

            for _l in getattr(_n, n_field):
                n__ = _l.n
                if n__ not in seen:
                    if n__.t == T.HIDDEN:
                        dest.add(n__)
                    seen.add(n__)
                    queue.append(n__)

        return dest

    i_seen, o_seen = bfs(i_nodes, "o"), bfs(o_nodes, "i")
    connected_hiddens = i_seen.intersection(o_seen)

    # Re-generate to clean connections (sometimes mandatory, when?)
    connections.clear()
    hiddens.clear()
    for n in connected_hiddens:
        hiddens.add(n.p)
        for link in n.i:
            connections.add(Connection(link.n.p, n.p, link.w))
        for link in n.o:
            if link.n.t == T.OUTPUT:
                connections.add(Connection(n.p, link.n.p, link.w))

    # Clean-up the memory (tons of self-references)
    for n in nodes.values():
        for l_ in n.i + n.o:
            l_.n = None
        n.i = None
        n.o = None

    return hiddens
