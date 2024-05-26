"""
Test documentation for genome file (module?)
"""
import logging
import pathlib
from collections import namedtuple
from collections.abc import Iterable
from random import Random
from shutil import which
from typing import (Dict, List, Any, Optional, Union,
                    Collection, Callable, Tuple)

from graphviz import Digraph
from importlib_resources import files
from pyrecord import Record

from .._cpp.config import Config
from .._cpp.genotype import CPPNData as _CPPNData, Innovations

ESHNOutputs = Config.ESHNOutputs

logger = logging.getLogger(__name__ + ".mutations")

dot_found = (which("dot") is not None)


def _string_to_labels(string: str):
    label = ""
    labels = []
    for c in string:
        if c == '_' or c.isalnum():
            label += c
        else:
            if label:
                labels.append(label)
            label = ""
    if label:
        labels.append(label)
    return labels


class IDManager:
    """ Very simple monotonic ID generator for nodes and links
    """

    def __init__(self):
        """ Create a node/link ID manager (starting from 0)"""
        self._nextNodeID, self._nextLinkID = 0, 0

    def initialize(self, next_node_id: int):
        """ Reset the manager to `next_node_id`
        for the nodes and 0 for the links"""
        self._nextNodeID = next_node_id
        self._nextLinkID = 0

    def get_node_id(self, *_):
        """ Request a new node id.

        Same signature as :meth:`~abrain._cpp.genotype.Innovations.get_node_id`
        but does not use the arguments and, instead, provides monotonously
        increasing ids
        """
        value = self._nextNodeID
        self._nextNodeID += 1
        return value

    def get_link_id(self, *_):
        """ Request a new link id.

        Same signature as :meth:`~abrain._cpp.genotype.Innovations.get_link_id`
        but does not use the arguments and, instead, provides monotonously
        increasing ids
        """
        value = self._nextLinkID
        self._nextLinkID += 1
        return value

    def next_node_id(self):
        """ The next id that would be assigned to a node.

        Mostly for debugging purposes
        """
        return self._nextNodeID

    def next_link_id(self):
        """ The next id that would be assigned to a link.

        Mostly for debugging purposes
        """
        return self._nextLinkID

    link_id = None
    node_id = None
    new_node_id = get_node_id
    new_link_id = get_link_id


class GIDManager:
    """Simple integer-producing class for unique genome identifier
    """
    def __init__(self):
        """Assign 0 as the next value
        """
        self._next_value = 0

    def __call__(self) -> int:
        """Return a new value and increment internal counter
        """
        value = self._next_value
        self._next_value += 1
        return value


class Genome(_CPPNData):
    """Genome class encoding a CPPN either for ES-HyperNEAT or for direct use.

    A simple collection of Node/Link.
    Can only be created via :meth:`random` init, :meth:`crossover` or
    :meth:`copy` (with the help of the nested :class:`Data` structure)
    """

    class Data:
        """ Long-lived data structure encapsulating the various elements
        needed throughout evolution """

        __private_key = object()
        """Private key for preventing default-constructed CPPNs
        """

        cdata: dict
        """ CPPN creation data (shape, functions, ...) """

        labels: Optional[List[str]]
        """ (Optional) Labels for pretty-printing the input/output nodes"""

        rng: Random
        """ Source of randomness """

        id_manager: Union[Innovations, IDManager]
        """ Either an innovations database (allowing crossover
         but slightly more expensive) or a monotonic index-based allocator
         (for mutations only) """

        gid_manager: Optional[GIDManager]
        """ (Optional) Manager for genome identifiers and lineage """

        def __init__(self,
                     seed: int,
                     cdata: dict,
                     labels: Optional[Union[str, Collection[str]]],
                     genome_ids: bool = True,
                     innovations: bool = True,
                     key=None):

            assert (key == Genome.Data.__private_key), \
                "Must with created dedicated create_for_* functions"

            self.cdata = cdata
            assert "i" in cdata and "o" in cdata
            i, o = cdata["i"], cdata["o"]

            self.rng = Random(seed)
            self.id_manager = Innovations() if innovations else IDManager()
            self.gid_manager = GIDManager() if genome_ids else None

            self.id_manager.initialize(i+o)

            if labels is not None:
                if (not isinstance(labels, str)
                        and isinstance(labels, Collection)):
                    tokens = labels
                else:
                    tokens = _string_to_labels(labels)
                if cdata["input_bias"]:
                    tokens = tokens[:i-1] + ["b"] + tokens[i-1:]
                if len(tokens) != i + o:
                    raise ValueError(f"Wrong number of labels. Expected"
                                     f" {i+o} got {len(tokens)}")
                self.labels = tokens
            else:
                self.labels = ([f"I{j}" for j in range(i)]
                               + [f"I{j+i}" for j in range(o)])

        @staticmethod
        def create_for_generic_cppn(
                inputs: int, outputs: Union[int, Collection[str]],
                labels: Optional[Union[str, Collection[str]]] = None,
                seed: Optional[int] = None,
                with_input_bias: bool = True,
                with_innovations: bool = True,
                with_lineage: bool = True
        ):
            """ Create a structure handling CPPNs for generic use

                :param inputs: Number of inputs for the CPPN
                :param outputs: Number of outputs for the CPPN
                :param labels: (Optional) list of i/o labels for
                 pretty-printing
                :param with_input_bias: Whether to use an input bias
                :param seed: Seed for the random number generator
                :param with_innovations: Whether to use NEAT historical
                 markings
                :param with_lineage: Whether to store lineage information
            """

            if isinstance(outputs, int):
                functions = [Config.defaultOutputFunction
                             for _ in range(outputs)]
            else:
                functions = outputs
                outputs = len(outputs)

            d = Genome.Data(
                cdata=dict(
                    i=inputs + int(with_input_bias),
                    o=outputs,
                    input_bias=with_input_bias,
                    functions=functions
                ),
                labels=labels,
                seed=seed,
                genome_ids=with_lineage,
                innovations=with_innovations,
                key=Genome.Data.__private_key)
            return d

        @staticmethod
        def create_for_eshn_cppn(
                dimension: int,
                with_input_bias: Optional[bool] = True,
                with_input_length: Optional[bool] = True,
                with_leo: Optional[bool] = True,
                with_output_bias: Optional[bool] = True,
                seed: Optional[int] = None,
                with_innovations: bool = True,
                with_lineage: bool = True
        ):
            """ Create a structure handling CPPNs for use with ES-HyperNEAT

                :param dimension: The substrate dimension (2- or 3-D)
                :param with_input_bias: Whether to use an input bias
                :param with_input_length: Whether to directly provide the
                 distance between points
                :param with_leo: Whether to use a Level of Expression Output
                :param with_output_bias: Whether to use the CPPN to generate
                 per-neuron biases
                :param seed: Seed for the random number generator
                :param with_innovations: Whether to use NEAT historical
                 markings
                :param with_lineage: Whether to store lineage information
            """

            if __debug__:  # pragma: no branch
                if dimension not in [2, 3]:
                    raise ValueError("This ES-HyperNEAT implementation only"
                                     " works with 2D or 3D ANNs")
                for i in [with_input_bias, with_input_length, with_leo,
                          with_output_bias]:
                    if not isinstance(i, bool):
                        raise ValueError("Wrong argument type")

            coordinates = "xy" if dimension == 2 else "xyz"
            inputs = 2*dimension
            labels = [f"{c}_{i}" for c in coordinates for i in range(2)]
            if with_input_length:
                inputs += 1
                labels.append("l")

            outputs = [Config.eshnOutputFunctions[ESHNOutputs.Weight]]
            labels.append("W")
            if with_leo:
                outputs.append(Config.eshnOutputFunctions[ESHNOutputs.LEO])
                labels.append("L")
            if with_output_bias:
                outputs.append(Config.eshnOutputFunctions[ESHNOutputs.Bias])
                labels.append("B")

            return Genome.Data.create_for_generic_cppn(
                inputs=inputs,
                outputs=outputs,
                labels=labels,
                seed=seed,
                with_input_bias=with_input_bias,
                with_innovations=with_innovations,
                with_lineage=with_lineage
            )

    __private_key = object()
    """Private key for preventing default-constructed CPPNs
    """

    __id_field = "_id"
    """Name of the private field in which to store the optional identifier
    """

    __parents_field = "_parents"
    """Name of the private field in which to store the optional genealogy
    """

    def __init__(self, key=None):
        """ Create a genome directly. Only available internally.
        """
        _CPPNData.__init__(self)
        assert (key == Genome.__private_key), \
            "Genome objects must be created via random, deepcopy or" \
            " reproduction methods"

    def __repr__(self):
        return (f"CPPN:{self.inputs}:{self.outputs}"
                f"([{len(self.nodes)}, {len(self.links)}])")

    def id(self) -> Optional[int]:
        """Return the genome id if one was generated"""
        return getattr(self, self.__id_field, None)

    def parents(self) -> Optional[int]:
        """Return the genome's parent(s) if possible"""
        return getattr(self, self.__parents_field, None)

    ###########################################################################
    # Public manipulation interface
    ###########################################################################

    @classmethod
    def random(cls, data: Data) -> 'Genome':
        """Create a random CPPN with boolean initialization

        :param data: Evolution-wide shared genomic data
        :return: A random CPPN genome
        """

        g = cls(cls.__private_key)
        g.inputs = data.cdata["i"]
        g.outputs = data.cdata["o"]
        g.bias = data.cdata["input_bias"]

        functions = data.cdata["functions"]
        for j in range(g.outputs):
            g._add_node(g.inputs + j, functions[j])

        rng = data.rng
        for i in range(g.inputs):
            for j in range(g.outputs):
                if rng.random() < .5:
                    g._add_link(data,
                                i, j + g.inputs,
                                cls.__random_link_weight(rng))

        if data.gid_manager is not None:
            setattr(g, g.__id_field, data.gid_manager())
            setattr(g, g.__parents_field, [])

        g._sort_by_id()

        return g

    def mutate(self, data: Data, n=1) -> None:
        """Mutate (in-place) this genome
        :param data: Evolution-wide shared genomic data
        :param n: Number of mutations to perform
        """

        for _ in range(n):
            self._mutate(data)
        self._sort_by_id()

    def mutated(self, data: Data, n=1) -> 'Genome':
        """Return a mutated (copied) version of this genome
        :param data: Evolution-wide shared genomic data
        :param n: Number of mutations to perform
        """
        copy = self.copy()
        copy.mutate(data, n)

        if data.gid_manager is not None:
            setattr(copy, self.__id_field, data.gid_manager())
            setattr(copy, self.__parents_field, [self.id()])
        return copy

    @staticmethod
    def crossover(lhs: 'Genome', rhs: 'Genome', data: Data,
                  bias: Union[int, str] = "length") -> 'Genome':
        """Create a new genome from two parents using historical markings

        Bias specifies which parent to take disjoint and excess genes from:
            * 0: lhs
            * 1: rhs
            * "length": from the smallest
              (or random if both have the same size)

        :param lhs: A parent
        :param rhs: Another parent
        :param data: Evolution-wide shared genomic data
        :param bias: Which parent to take disjoint and excess genes
         (see description)

        :return: The child
        """

        if bias not in [0, 1, "length"]:
            raise ValueError(f"Invalid argument {bias=}")
        if bias == "length":
            lhs_l_len, rhs_l_len = len(lhs.links), len(rhs.links)
            lhs_n_len, rhs_n_len = len(lhs.nodes), len(rhs.nodes)
            if lhs_l_len != rhs_l_len:
                bias = 0 if lhs_l_len < rhs_l_len else 1
            elif lhs_n_len != rhs_n_len:
                bias = 0 if lhs_n_len < rhs_n_len else 1
            else:
                bias = data.rng.choice([0, 1])

        return Genome._crossover(lhs, rhs, data, bias)

    @staticmethod
    def distance(lhs: 'Genome', rhs: 'Genome', weights=None) -> float:
        """ Compute the genetic distance between `lhs` and `rhs`

        :param lhs: The first genome
        :param rhs: The first genome
        :param weights: How to weight the different terms:

        * Node excess
        * Node disjoint
        * Node matched
        * Link excess
        * Link disjoint
        * Link matched

        :return: the weighted genetic distance
        """

        # Genome._debug_align(lhs, rhs)
        if weights is None:
            weights = [1]*6
        nc1, nc2, nc3, lc1, lc2, lc3 = weights
        nn = max(len(lhs.nodes), len(rhs.nodes))
        nl = max(len(lhs.links), len(rhs.links))
        stats = Genome._distance(lhs, rhs)
        # pprint.pprint(stats)

        d = 0
        if nn > 0:  # pragma: no cover
            d += (
                nc1 * stats["nodes"]["excess"] / nn
                + nc2 * stats["nodes"]["disjoint"] / nn
            )
        if (nm := stats["nodes"]["matched"]) > 0:  # pragma: no cover
            d += nc3 * stats["nodes"]["diff"] / nm
        if nl > 0:  # pragma: no cover
            d += (
                lc1 * stats["links"]["excess"] / nl
                + lc2 * stats["links"]["disjoint"] / nl
            )
        if (lm := stats["links"]["matched"]) > 0:  # pragma: no cover
            d += lc3 * stats["links"]["diff"] / lm

        return d

    def copy(self) -> 'Genome':
        """Return a perfect (deep)copy of this genome"""
        copy = Genome(Genome.__private_key)

        copy.inputs = self.inputs
        copy.outputs = self.outputs
        copy.bias = self.bias
        copy.labels = self.labels
        copy.nodes = self.nodes
        copy.links = self.links

        if hasattr(self, self.__id_field):
            setattr(copy, copy.__id_field, self.id())
            setattr(copy, copy.__parents_field, self.parents())

        return copy

    def update_lineage(self, data: Data, parents: List['Genome']):
        """Update lineage fields

        :param data: Evolution-wide shared genomic data
        :param parents: list (potentially empty) of this genome's parents
        """
        assert data.gid_manager
        setattr(self, self.__id_field, data.gid_manager())
        setattr(self, self.__parents_field, [p.id() for p in parents])

    ###########################################################################
    # Public copy/deepcopy interface
    ###########################################################################

    def __copy__(self):
        """Return a perfect (deep)copy of this genome

        Implements the shallow copy interface
        """
        return self.copy()

    def __deepcopy__(self, _):
        """Return a perfect (deep)copy of this genome

        Implements the deep copy interface
        """
        return self.copy()

    ###########################################################################
    # Public pickle interface
    ###########################################################################

    def __getstate__(self):
        """Pickle the genome

        Also store the id/parents if present
        """
        dct = _CPPNData.__getstate__(self)
        if hasattr(self, self.__id_field):
            dct[self.__id_field] = self.id()
            dct[self.__parents_field] = self.parents()
        return dct

    def __setstate__(self, state):
        """Unpickle the genome

        Also restore the id/parents if present
        """
        _CPPNData.__setstate__(self, state)
        if self.__id_field in state:
            setattr(self, self.__id_field, state[self.__id_field])
            setattr(self, self.__parents_field, state[self.__parents_field])

    ###########################################################################
    # Public json/binary interface
    ###########################################################################

    def to_json(self) -> Dict[Any, Any]:
        """Return a json (dict) representation of this object"""
        dct = dict(
            inputs=self.inputs,
            outputs=self.outputs,
            bias=self.bias,
            labels=self.labels,
            nodes=[(n.id, n.func) for n in self.nodes],
            links=[(l_.id, l_.src, l_.dst, l_.weight) for l_ in self.links],
        )
        if (gid := self.id()) is not None:
            dct[self.__id_field] = gid
            dct[self.__parents_field] = self.parents()
        return dct

    @staticmethod
    def from_json(data: Dict[Any, Any]) -> 'Genome':
        """Recreate a Genome from a string json representation
        """
        g = Genome(Genome.__private_key)
        g.inputs = data["inputs"]
        g.outputs = data["outputs"]
        g.bias = data["bias"]
        g.labels = data["labels"]
        g.nodes = _CPPNData.Nodes([
            _CPPNData.Node(n[0], n[1]) for n in data["nodes"]
        ])
        g.links = _CPPNData.Links([
            _CPPNData.Link(*[d[0], d[1], d[2], d[3]]) for d in data["links"]
        ])

        if (gid := data.get(g.__id_field, None)) is not None:
            setattr(g, g.__id_field, gid)
            setattr(g, g.__parents_field, data[g.__parents_field])

        return g

    ###########################################################################
    # Public graphviz/dot interface
    ###########################################################################

    @staticmethod
    def from_dot(path: str, rng: Random) -> 'Genome':
        """Produce a Genome by parsing a simplified graph description

            .. warning:: Unimplemented

        :param path: the file to load
        :param rng: random pick unspecified functions
        :return: The user-specified CPPN
        """
        raise NotImplementedError  # pragma: no cover

    def to_dot(self, data: Data, path: str, ext: str = "pdf",
               title: Optional[str] = None,
               debug: Optional[Union[str, Iterable]] = None) -> str:
        """Produce a graphical representation of this genome

        .. note:: Missing functions images in nodes and/or lots of
            warning message are likely caused by misplaced image files.
            In doubt perform a full reinstall

        Args:
            data: Genome's shared data (for labels)
            path: The path to write to
            ext: The rendering format to use
            title: Optional title for the graph (e.g. for generational info)
            debug: Print more information on the graph.
                Special values (either as a single string or a list):

                - 'depth' will display every nodes' depth
                - 'links' will display every links' id
                - 'keepdot' will keep the intermediate dot file
                - 'all' all of the above

        Raises:
            OSError: if the `dot` program is not available (not installed, on
                the path and executable)
        """
        if not dot_found:  # pragma: no cover
            raise OSError("""
                dot program not found. Make sure it is installed before using
                 this function.

                [ubuntu] sudo apt install graphviz
            """)

        def _should_debug(flag: str):
            return isinstance(debug, Iterable) and flag in debug

        if _should_debug("all"):  # pragma: no cover
            debug = "depth;links;keepdot"

        path = pathlib.Path(path)
        if path.suffix != "":
            ext = path.suffix.replace('.', '')
            path = path.with_suffix('')

        vectorial = ext in ["pdf", "svg", "eps"]

        dot = Digraph('cppn')
        if title is not None:
            dot.attr(labelloc="t")
            dot.attr(label=title)

        g_i, g_h, g_o, g_ol = [Digraph(n) for n in
                               ['inputs', 'hidden', 'outputs', 'o_labels']]

        dot.attr(rankdir='BT')
        g_i.attr(rank="source")
        g_o.attr(rank="same")
        g_ol.attr(rank="sink")
        ix_to_name = {}

        depths = None
        if _should_debug("depth"):
            depths = self._compute_node_depths(self.links)

        def x_label(_data):
            return (f"<<font color='grey'>{_data}</font>>"
                    if _data is not None else '')

        def node_x_label(_nid):
            return x_label(depths[_nid] if depths is not None else None)

        # input nodes
        for i in range(self.inputs):
            label = data.labels[i]
            name = "I" + label.upper().replace("_", "")
            ix_to_name[i] = name
            if '_' in label:
                j = label.find('_')
                label = label[:j] + "<SUB>" + label[j + 1:] + "</SUB>"
            g_i.node(name, f"<{label}>", shape='plaintext',
                     xlabel=node_x_label(i))

        img_format = "svg" if vectorial else "png"

        def img_file(_func):
            p = (files('abrain') / f"core/functions/{_func}.{img_format}")
            return str(p.resolve())

        for i, n in enumerate(self.nodes):
            nid, func = n.id, n.func
            if self._is_output(nid):
                label = data.labels[i+self.inputs]
                name = "O" + label.upper().replace("_", "")
                ix_to_name[i + self.inputs] = name
                g_o.node(name, "",
                         width="0.5in", height="0.5in", margin="0",
                         fixedsize="shape",
                         image=img_file(func),
                         xlabel=node_x_label(i + self.inputs))
                g_ol.node(name + "_l", shape="plaintext", label=label)
                dot.edge(name, name + "_l")

            else:
                name = f"H{nid}"
                ix_to_name[nid] = name
                label = name if debug is not None else ""
                g_h.node(name, label,
                         width="0.5in", height="0.5in", margin="0",
                         fixedsize="shape",
                         image=img_file(func),
                         xlabel=node_x_label(nid))

        for i, l in enumerate(self.links):
            if l.weight == 0:  # pragma: no cover
                continue  # Highly unlikely

            w = abs(l.weight) / Config.cppnWeightBounds.max
            dot.edge(ix_to_name[l.src], ix_to_name[l.dst],
                     color="red" if l.weight < 0 else "black",
                     penwidth=str(3.75 * w + .25),
                     xlabel=x_label(
                         l.id if _should_debug("links") else None))

        # enforce i/o order
        for i in range(self.inputs - 1):
            dot.edge(ix_to_name[i], ix_to_name[i + 1], style="invis")
        for i in range(self.inputs, self.inputs + self.outputs - 1):
            dot.edge(ix_to_name[i], ix_to_name[i + 1], style="invis")

        # Register sub-graphs
        for g in [g_i, g_h, g_o, g_ol]:
            dot.subgraph(g)

        cleanup = False if _should_debug("keepdot") else True

        return dot.render(path, format=ext, cleanup=cleanup)

    ###########################################################################
    # Private mutation interface
    ###########################################################################

    def _is_input(self, nid: int):
        return nid < self.inputs

    def _is_output(self, nid: int):
        return self.inputs <= nid < self.inputs + self.outputs

    def _is_hidden(self, nid: int):
        return self.inputs + self.outputs <= nid

    @staticmethod
    def __random_node_function(rng: Random):
        return rng.choice(Config.functionSet)

    def _add_node(self, nid: int, func: str) -> None:
        self.nodes.append(_CPPNData.Node(nid, func))

    @staticmethod
    def __random_link_weight(rng: Random):
        return rng.uniform(Config.cppnWeightBounds.rndMin,
                           Config.cppnWeightBounds.rndMax)

    def __add_link_with_id(self, lid: int, src: int, dst: int, weight: float):
        self.links.append(_CPPNData.Link(lid, src, dst, weight))

    def _add_link(self, data: Data,
                  src: int, dst: int, weight: float,
                  is_mutation: bool = True) -> int:
        """Add a link to the genome

        :param data: Evolution-wide shared genomic data
        :param src: index of the source node in [0,
         :py:attr:`core.genome.Genome.inputs`]
        :param dst: index of the destination node in
         [Genome.inputs,Genome.outputs]
        :param weight: connection weight in [-1,1]
        """
        lid = data.id_manager.get_link_id(src, dst)
        if self._has_link(lid):  # pragma: no cover
            # (fail-safe) History no longer makes sense. Burn it down!
            lid = data.id_manager.new_link_id(src, dst)

        self.__add_link_with_id(lid, src, dst, weight)
        if is_mutation:
            logger.debug(f"[add_l:{lid}] {src} -> {dst}")
        return lid

    def __random_add_node(self, data: Data) -> None:
        i = data.rng.randrange(0, len(self.links))
        link: _CPPNData.Link = self.links[i]
        src, dst, weight = link.src, link.dst, link.weight

        nid = data.id_manager.get_node_id(src, dst)
        if self._has_node(nid):  # pragma: no cover
            # (fail-safe) History no longer makes sense. Burn it down!
            nid = data.id_manager.new_node_id(src, dst)
        self._add_node(nid, Genome.__random_node_function(data.rng))

        logger.debug(f"[add_n:{nid}] {src} -> {dst} >> {src}"
                     f" -> {nid} -> {dst}")

        self._add_link(data, src, nid, 1, is_mutation=False)
        self._add_link(data, nid, dst, weight, is_mutation=False)
        self.links.pop(i)

    def __random_del_node(self, data: Data, candidates) -> None:
        nid = data.rng.choice(candidates)
        n_ix, n = next((n_ix, n) for n_ix, n in  # pragma: no branch
                       enumerate(self.nodes) if n.id == nid)
        i_links = [(j, link) for j, link in enumerate(self.links)
                   if link.dst == n.id]
        o_links = [(j, link) for j, link in enumerate(self.links)
                   if link.src == n.id]
        assert len(i_links) == 1 and len(o_links) == 1

        j_i, i_l = i_links[0][0], i_links[0][1]
        j_o, o_l = o_links[0][0], o_links[0][1]
        logger.debug(f"[del_n] {i_l.src} -> {n.id} -> {o_l.dst} >>"
                     f" {i_l.src} -> {o_l.dst}")

        if sum(li.src == i_l.src and li.dst == o_l.dst for li in
               self.links) == 0 \
                and i_l.src != o_l.dst:
            # Link does not exist already and is not auto-recurrent -> add it
            self._add_link(data, i_l.src, o_l.dst, o_l.weight,
                           is_mutation=False)
        self.nodes.pop(n_ix)
        self.links.pop(j_i)
        if j_i < j_o:  # index may have been made invalid
            j_o -= 1
        self.links.pop(j_o)

    def __random_add_link(self, data: Data, candidates) -> None:
        link = data.rng.choice(list(candidates))
        weight = Genome.__random_link_weight(data.rng)

        lid = self._add_link(data, link.src, link.dst, weight,
                             is_mutation=False)
        logger.debug(f"[add_l:{lid}] {link.src} -({weight})-> {link.dst}")

    def __random_del_link(self, data: Data, candidates) -> None:
        i = candidates[data.rng.randrange(0, len(candidates))]
        logger.debug(f"[del_l] {self.links[i].src} -> {self.links[i].dst}")
        self.links.pop(i)

    def __random_mutate_weight(self, data: Data) -> None:
        link = data.rng.choice(self.links)
        bounds = Config.cppnWeightBounds
        delta = 0
        while delta == 0:
            delta = data.rng.gauss(0, bounds.stddev)
        w = max(bounds.min, min(link.weight + delta, bounds.max))

        logger.debug(f"[mut_w] {link.src} -({link.weight})->"
                     f" {link.dst} >> {w}")

        link.weight = w

    def __random_mutate_func(self, data: Data) -> None:
        # Cannot be used as slices return new objects
        # n = data.rng.choice(self.nodes[self.outputs:])
        # ===
        i = data.rng.randint(self.outputs, len(self.nodes)-1)
        n = self.nodes[i]
        f = data.rng.choice([f for f in Config.functionSet if not f == n.func])
        logger.debug(f"[mut_f] N({n.id}, {n.func}) -> {f}")
        n.func = f

    def _mutate(self, data: Data):
        # Get the list of all nodes
        N = namedtuple('Node', 'id type')
        offset = 0
        all_nodes: List[N] = []
        for i in range(self.inputs):
            all_nodes.append(N(i + offset, "I"))
        offset += self.inputs
        for i in range(self.outputs):
            all_nodes.append(N(i + offset, "O"))
        offset += self.outputs
        for i, n in enumerate(self.nodes[self.outputs:]):
            all_nodes.append(N(n.id, "H"))

        depths = self._compute_node_depths(self.links)

        # For now assume that all valid links can be created.
        # Existing ones are pruned afterward
        L = namedtuple('Link', 'src dst')
        potential_links = set()
        for l_id, l_type in all_nodes:
            if l_type == "O":
                continue

            l_depth = depths[l_id]
            for r_id, r_type in all_nodes:
                r_depth = depths[r_id]
                if (l_id != r_id
                        and r_type != "I"
                        and (r_type == "O" or l_depth <= r_depth)):
                    potential_links.add(L(l_id, r_id))

        for link in self.links:
            potential_links.discard(
                L(link.src, link.dst))  # not a candidate: already exists

        # Get inputs/outputs for each node.
        # Refine strict definition to detect loops
        node_degrees = self._compute_node_degrees()

        # Ensure that link removal does not produce in-/output less nodes
        # (except for I/O nodes, obviously)
        non_essential_links: List[int] = []
        for i, link in enumerate(self.links):
            if (self._is_input(link.src)
                and self._is_output(link.dst)) \
                    or (node_degrees[link.src].o > 1
                        and node_degrees[link.dst].i > 1):
                non_essential_links.append(i)

        # Simple nodes are (hidden) nodes of the form:
        # A -> X -> B which can be replaced by A -> B
        simples_nodes: List[int] = []
        for nid, d in node_degrees.items():
            if not self._is_output(nid) and d.i == 1 and d.o == 1:
                simples_nodes.append(nid)

        # Copy the rates and update based on possible mutations
        rates = {k: v for k, v in Config.mutationRates.items()}

        def update(key, predicate):
            predicate() or rates.pop(key)

        update("add_n", lambda: len(self.links) > 0)
        update("add_l", lambda: len(potential_links) > 0)
        update("del_l", lambda: len(non_essential_links) > 0)
        update("del_n", lambda: len(simples_nodes) > 0)
        update("mut_f", lambda: len(self.nodes) - self.outputs > 0)
        update("mut_w", lambda: len(self.links) > 0)
        assert len(rates) > 0

        actions = {
            "add_n": [self.__random_add_node],
            "add_l": [self.__random_add_link, potential_links],
            "del_n": [self.__random_del_node, simples_nodes],
            "del_l": [self.__random_del_link, non_essential_links],
            "mut_f": [self.__random_mutate_func],
            "mut_w": [self.__random_mutate_weight]
        }

        assert sum(rates.values()) > 0  # Should never occur outside tests

        choice = data.rng.choices(list(rates.keys()), list(rates.values()))[0]
        actions[choice][0](data, *actions[choice][1:])

    ###########################################################################
    # Private crossover interface
    ###########################################################################

    @staticmethod
    def _crossover(lhs: 'Genome', rhs: 'Genome', data: 'Genome.Data',
                   bias: int) -> 'Genome':
        child = Genome(key=Genome.__private_key)

        child.inputs = lhs.inputs
        assert child.inputs == rhs.inputs
        child.outputs = lhs.outputs
        assert child.outputs == rhs.outputs
        child.bias = lhs.bias
        assert child.bias == rhs.bias

        stats = dict(nodes=dict(matched=0, disjoint=0, excess=0),
                     links=dict(matched=0, disjoint=0, excess=0))
        nodes: dict[int, Tuple[_CPPNData.Node, bool]] = {}

        def add_node(_n: _CPPNData.Node): child._add_node(_n.id, _n.func)

        def add_link(_l: _CPPNData.Link):
            child.__add_link_with_id(
                lid=_l.id, src=_l.src, dst=_l.dst, weight=_l.weight)
            for nid in [_l.src, _l.dst]:
                if child._is_input(nid):
                    continue
                n, inserted = nodes[nid]
                if not inserted:
                    add_node(n)
                    nodes[nid] = (n, True)

        def collect_nodes(_ln: _CPPNData.Node,
                          _rn: _CPPNData.Node,
                          _excess: bool):
            key = "excess" if _excess else "disjoint"
            if _rn is None:
                nodes[_ln.id] = (_ln, False)
            elif _ln is None:
                nodes[_rn.id] = (_rn, False)
            else:
                inserted = False
                n = data.rng.choice([_ln, _rn])  # Pick one
                if child._is_output(_ln.id):     # Always add output nodes
                    add_node(n)
                    inserted = True
                nodes[_ln.id] = (n, inserted)    # Register
                key = "matched"
            stats["nodes"][key] += 1

        def process_links(_ll: _CPPNData.Link,
                          _rl: _CPPNData.Link,
                          _excess: bool):
            key = "excess" if _excess else "disjoint"
            if _rl is None:    # Maybe add link from lhs
                if bias == 0:
                    add_link(_ll)
            elif _ll is None:  # Maybe add link from lhs
                if bias == 1:
                    add_link(_rl)
            else:              # Matching links -> add one
                add_link(data.rng.choice([_ll, _rl]))
                key = "matched"
            stats["links"][key] += 1

        # Collect nodes
        Genome._iter(lhs, rhs, 'nodes', collect_nodes)

        # Search for matching/disjoint/excess links
        Genome._iter(lhs, rhs, 'links', process_links)

        child._sort_by_id()
        child.update_lineage(data, [lhs, rhs])
        return child

    @staticmethod
    def _distance(lhs: 'Genome', rhs: 'Genome') -> dict:
        stats = dict(nodes=dict(matched=0, disjoint=0, excess=0, diff=0),
                     links=dict(matched=0, disjoint=0, excess=0, diff=0))

        def process(_l: Union[_CPPNData.Node, _CPPNData.Link],
                    _r: Union[_CPPNData.Node, _CPPNData.Link],
                    _excess: bool, category: str, _fn: Callable):
            key = "excess" if _excess else "disjoint"
            if _l is not None and _r is not None:
                key = "matched"
                stats[category]["diff"] += _fn(_l, _r)
            stats[category][key] += 1

        def n_distance(_l: _CPPNData.Node, _r: _CPPNData.Node) -> float:
            return int(_l.func != _r.func)

        def l_distance(_l: _CPPNData.Link, _r: _CPPNData.Link) -> float:
            return abs(_l.weight - _r.weight)

        for (field, fn) in [("nodes", n_distance), ("links", l_distance)]:
            Genome._iter(lhs, rhs, field,
                         lambda _l, _r, _e: process(_l, _r, _e, field, fn))

        return stats

    ###########################################################################
    # Private helpers
    ###########################################################################

    # noinspection PyProtectedMember
    _sort_by_id = _CPPNData._sort_by_id

    def _has_node(self, nid: int):
        return any(n.id == nid for n in self.nodes)

    def _has_link(self, lid: int):
        return any(_l.id == lid for _l in self.links)

    def _compute_node_depths(self, links_original):
        depths = {}
        links = [l_ for l_ in links_original]

        def recurse(nid: int, depth: int):
            if nid < self.inputs:
                depths[nid] = 0
                return 0

            current_depth = depths.get(nid, 0)
            if current_depth == 0:
                nonlocal links
                head, tail = [], []
                for l_ in links:
                    (head if l_.dst == nid else tail).append(l_)
                links = tail

                current_depth = 0
                for l_ in head:
                    current_depth = max(current_depth,
                                        recurse(l_.src, depth + 1) + 1)

                depths[nid] = current_depth

            return current_depth

        for i in range(self.inputs + self.outputs):
            depths.setdefault(i, 0)

        for i in range(self.outputs):
            recurse(i + self.inputs, 0)

        return depths

    def _compute_node_degrees(self):
        d = Record.create_type("D", "i", "o", i=0, o=0)

        node_degrees: Dict[int, d] = {}
        for node in self.nodes:
            node_degrees[node.id] = d(0, 0)

        for link in self.links:
            node_degrees.setdefault(link.src, d()).o += 1
            node_degrees.setdefault(link.dst, d()).i += 1

        return node_degrees

    @staticmethod
    def _iter(lhs: 'Genome', rhs: 'Genome', field: str,
              fn: Callable[[Any, Any, bool], Any]):
        lhs_i, rhs_i = 0, 0
        lhs_items, rhs_items = getattr(lhs, field), getattr(rhs, field)
        lhs_len, rhs_len = len(lhs_items), len(rhs_items)
        while lhs_i < lhs_len and rhs_i < rhs_len:
            l_item, r_item = lhs_items[lhs_i], rhs_items[rhs_i]
            if l_item.id < r_item.id:
                fn(l_item, None, False)
                lhs_i += 1

            elif l_item.id > r_item.id:
                fn(None, r_item, False)
                rhs_i += 1

            else:
                fn(l_item, r_item, False)
                lhs_i += 1
                rhs_i += 1

        items = lhs_items
        while lhs_i < lhs_len:
            fn(items[lhs_i], None, True)
            lhs_i += 1

        items = rhs_items
        while rhs_i < rhs_len:
            fn(None, items[rhs_i], True)
            rhs_i += 1

    @staticmethod
    def _debug_align(lhs: 'Genome', rhs: 'Genome', w=30):  # pragma: no cover
        print(f"[kgd-debug-align] Aligning {lhs.id()}={rhs.id()}")

        l_fmt, r_fmt = f"{{:>{w}s}}", f"{{:<{w}s}}"

        def print_item(_l, _r, _):
            if _r is None:
                print(l_fmt.format(str(_l)), "<")
            elif _l is None:
                print(l_fmt.format(""), "  >", r_fmt.format(str(_r)))
            else:
                print(l_fmt.format(str(_l)), "<=>", r_fmt.format(str(_r)))

        Genome._iter(lhs, rhs, 'nodes', print_item)
        Genome._iter(lhs, rhs, 'links', print_item)
