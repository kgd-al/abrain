"""
Test documentation for genome file (module?)
"""
import logging
import pathlib
import pprint
from collections import namedtuple
from collections.abc import Iterable
from random import Random
from shutil import which
from typing import Dict, List, Any, Optional, Union, Collection

from graphviz import Digraph
from importlib_resources import files
from pyrecord import Record

from .._cpp.config import Config
from .._cpp.genotype import CPPNData as _CPPNData

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
    Can only be created via random init or copy
    """

    __private_key = object()
    """Private key for preventing default-constructed CPPNs
    """

    __id_field = "_id"
    """Name of the private field in which to store the optional identifier
    """

    __parents_field = "_parents"
    """Name of the private field in which to store the optional genealogy
    """

    __labels_sep = ","

    def __init__(self, key=None):
        _CPPNData.__init__(self)
        assert (key == Genome.__private_key), \
            "Genome objects must be created via random, deepcopy or" \
            " reproduction methods"

    def __repr__(self):
        return f"CPPN:{self.inputs}:{self.outputs}([{len(self.nodes)}," \
               f" {self.nextNodeID}], [{len(self.links)},{self.nextLinkID}])"

    def id(self) -> Optional[int]:
        """Return the genome id if one was generated"""
        return getattr(self, self.__id_field, None)

    def parents(self) -> Optional[int]:
        """Return the genome's parent(s) if possible"""
        return getattr(self, self.__parents_field, None)

    ###########################################################################
    # Public manipulation interface
    ###########################################################################

    @staticmethod
    def random(rng: Random,
               inputs: int,
               outputs: Union[int, Collection[str]],
               with_input_bias: bool = True,
               labels: Optional[Union[str, Collection[str]]] = None,
               id_manager: Optional[GIDManager] = None) \
            -> 'Genome':
        """Create a random CPPN with boolean initialization

        :param rng: The source of randomness
        :param inputs: Number of inputs
        :param outputs: Number of outputs or specific functions
        :param with_input_bias: Whether to use an input bias
        :param labels: Labels for the inputs/outputs (in that order, single characters)
        :param id_manager: an optional manager providing unique identifiers

        :return: A random CPPN genome
        """
        g = Genome(Genome.__private_key)
        inputs += int(with_input_bias)
        g.inputs = inputs
        g.outputs = outputs if isinstance(outputs, int) else len(outputs)
        g.bias = with_input_bias

        if labels is not None:
            if not isinstance(labels, str) and isinstance(labels, Collection):
                tokens = labels
            else:
                tokens = _string_to_labels(labels)
            if with_input_bias:
                tokens = tokens[:g.inputs-1] + ["b"] + tokens[g.inputs-1:]
            if len(tokens) != g.inputs + g.outputs:
                raise ValueError(f"Wrong number of labels. Expected {g.inputs+g.outputs}"
                                 f" got {len(tokens)}")
            g.labels = Genome.__labels_sep.join(tokens)

        def output_func(_i):
            if isinstance(outputs, Iterable):
                return outputs[_i]
            else:
                return Config.defaultOutputFunction

        for j in range(g.outputs):
            g._add_node(output_func(j))

        for i in range(g.inputs):
            for j in range(g.outputs):
                if rng.random() < .5:
                    g._add_link(i, j + inputs,
                                Genome.__random_link_weight(rng))

        if id_manager is not None:
            setattr(g, g.__id_field, id_manager())
            setattr(g, g.__parents_field, [])

        return g

    @staticmethod
    def eshn_random(rng: Random,
                    dimension: int,
                    with_input_bias: Optional[bool] = True,
                    with_input_length: Optional[bool] = True,
                    with_leo: Optional[bool] = True,
                    with_output_bias: Optional[bool] = True,
                    id_manager: Optional[GIDManager] = None):
        """Create a random CPPN with boolean initialization for use with ES-HyperNEAT

        :param rng: The source of randomness
        :param dimension: The substrate dimension (2- or 3-D)
        :param with_input_bias: Whether to use an input bias
        :param with_input_length: Whether to directly provide the distance between points
        :param with_leo: Whether to use a Level of Expression Output
        :param with_output_bias: Whether to use the CPPN to generate per-neuron biases
        :param id_manager: an optional manager providing unique identifiers

        :return: A random CPPN genome for use with ES-HyperNEAT
        """

        if __debug__:  # pragma: no branch
            if dimension not in [2, 3]:
                raise ValueError("This ES-HyperNEAT implementation only works with 2D or 3D ANNs")
            for i in [with_input_bias, with_input_length, with_leo, with_output_bias]:
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

        return Genome.random(rng, inputs, outputs,
                             with_input_bias=with_input_bias,
                             labels=labels,
                             id_manager=id_manager)

    def mutate(self, rng: Random) -> None:
        """Mutate (in-place) this genome

        :param rng: The source of randomness
        """

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
                if l_id != r_id \
                        and r_type != "I" \
                        and (r_type == "O" or l_depth <= r_depth):
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
        update("mut_f", lambda: len(self.nodes) > 0)
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

        choice = rng.choices(list(rates.keys()), list(rates.values()))[0]
        actions[choice][0](rng, *actions[choice][1:])

    def mutated(self, rng: Random, id_manager: Optional[GIDManager] = None) \
            -> 'Genome':
        """Return a mutated (copied) version of this genome

        :param rng: the source of randomness
        :param id_manager: an optional manager providing unique identifiers
        """
        copy = self.copy()
        copy.mutate(rng)

        if __debug__:  # pragma: no branch
            should_have_id = hasattr(self, self.__id_field)
            can_have_id = (id_manager is not None)
            assert not should_have_id or can_have_id, \
                "Current genome has an id, but no ID manager provided for child"
            assert should_have_id or not can_have_id, \
                "ID manager provided but parent has no id"

        if id_manager is not None:
            setattr(copy, self.__id_field, id_manager())
            setattr(copy, self.__parents_field, [self.id()])
        return copy

    def copy(self) -> 'Genome':
        """Return a perfect (deep)copy of this genome"""
        copy = Genome(Genome.__private_key)

        copy.inputs = self.inputs
        copy.outputs = self.outputs
        copy.bias = self.bias
        copy.labels = self.labels
        copy.nodes = self.nodes
        copy.links = self.links
        copy.nextNodeID = self.nextNodeID
        copy.nextLinkID = self.nextLinkID

        if hasattr(self, self.__id_field):
            setattr(copy, copy.__id_field, self.id())
            setattr(copy, copy.__parents_field, self.parents())

        return copy

    def update_lineage(self, id_manager: GIDManager, parents: List['Genome']):
        """Update lineage fields

        :param id_manager: generator of unique identifiers
        :param parents: list (potentially empty) of this genome's parents
        """
        setattr(self, self.__id_field, id_manager())
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
            NID=self.nextNodeID,
            LID=self.nextLinkID
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
        g.nextNodeID = data["NID"]
        g.nextLinkID = data["LID"]

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

    def to_dot(self, path: str, ext: str = "pdf",
               title: Optional[str] = None,
               debug: Optional[Union[str, Iterable]] = None) -> str:
        """Produce a graphical representation of this genome

        .. note:: Missing functions images in nodes and/or lots of
            warning message are likely caused by misplaced image files.
            In doubt perform a full reinstall

        Args:
            path: The path to write to
            ext: The rendering format to use
            title: Optional title for the graph (e.g. for generational info)
            debug: Print more information on the graph.
                Special values (either as a single string or a list):

                - 'depth' will display every nodes' depth
                - 'links' will display every links' id
                - 'keepdot' will keep the intermediate dot file

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
        if isinstance(debug, Iterable) and "depth" in debug:
            depths = self._compute_node_depths(self.links)

        def x_label(data):
            return f"<<font color='grey'>{data}</font>>" if data is not None \
                else ''

        def node_x_label(nid):
            return x_label(depths[nid] if depths is not None else None)

        if self.labels:
            labels = self.labels.split(self.__labels_sep)
        else:
            labels = ([f"I{i}" for i in range(self.inputs)]
                      + [f"I{i+self.inputs}" for i in range(self.outputs)])

        # input nodes
        for i in range(self.inputs):
            label = labels[i]
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
                label = labels[i+self.inputs]
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
                         l.id if isinstance(debug, Iterable) and "links" in debug
                         else None))

        # enforce i/o order
        for i in range(self.inputs - 1):
            dot.edge(ix_to_name[i], ix_to_name[i + 1], style="invis")
        for i in range(self.inputs, self.inputs + self.outputs - 1):
            dot.edge(ix_to_name[i], ix_to_name[i + 1], style="invis")

        # Register sub-graphs
        for g in [g_i, g_h, g_o, g_ol]:
            dot.subgraph(g)

        cleanup = False if isinstance(debug, Iterable) and "keepdot" in debug else True

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

    def _add_node(self, func: str) -> int:
        nid = self.nextNodeID + self.inputs
        self.nodes.append(_CPPNData.Node(nid, func))
        self.nextNodeID += 1
        return nid

    @staticmethod
    def __random_link_weight(rng: Random):
        return rng.uniform(Config.cppnWeightBounds.rndMin,
                           Config.cppnWeightBounds.rndMax)

    def _add_link(self, src: int, dst: int, weight: float, spontaneous: bool = True) -> int:
        """Add a link to the genome

        :param src: index of the source node in [0,
         :py:attr:`core.genome.Genome.inputs`]
        :param dst: index of the destination node in
         [Genome.inputs,Genome.outputs]
        :param weight: connection weight in [-1,1]
        """
        lid = self.nextLinkID
        self.links.append(_CPPNData.Link(lid, src, dst, weight))
        self.nextLinkID += 1
        if spontaneous:
            logger.debug(f"[add_l] {src} -> {dst}")
        return lid

    def __random_add_node(self, rng: Random) -> None:
        i = rng.randrange(0, len(self.links))
        link: _CPPNData.Link = self.links[i]
        nid = self._add_node(Genome.__random_node_function(rng))

        src, dst, weight = link.src, link.dst, link.weight
        logger.debug(f"[add_n] {src} -> {dst} >> {src} -> {nid} -> {dst}")

        self._add_link(src, nid, 1, spontaneous=False)
        self._add_link(nid, dst, weight, spontaneous=False)
        self.links.pop(i)

    def __random_del_node(self, rng: Random, candidates) -> None:
        nid = rng.choice(candidates)
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
            self._add_link(i_l.src, o_l.dst, o_l.weight, spontaneous=False)
        self.nodes.pop(n_ix)
        self.links.pop(j_i)
        if j_i < j_o:  # index may have been made invalid
            j_o -= 1
        self.links.pop(j_o)

    def __random_add_link(self, rng: Random, candidates) -> None:
        link = rng.choice(list(candidates))
        weight = Genome.__random_link_weight(rng)

        logger.debug(f"[add_l] {link.src} -({weight})-> {link.dst}")

        self._add_link(link.src, link.dst, weight, spontaneous=False)

    def __random_del_link(self, rng: Random, candidates) -> None:
        i = candidates[rng.randrange(0, len(candidates))]
        logger.debug(f"[del_l] {self.links[i].src} -> {self.links[i].dst}")
        self.links.pop(i)

    def __random_mutate_weight(self, rng: Random) -> None:
        link = rng.choice(self.links)
        bounds = Config.cppnWeightBounds
        delta = 0
        while delta == 0:
            delta = rng.gauss(0, bounds.stddev)
        w = max(bounds.min, min(link.weight + delta, bounds.max))

        logger.debug(f"[mut_w] {link.src} -({link.weight})->"
                     f" {link.dst} >> {w}")

        link.weight = w

    def __random_mutate_func(self, rng: Random) -> None:
        n = rng.choice(self.nodes)
        f = rng.choice([f for f in Config.functionSet if not f == n.func])
        logger.debug(f"[mut_f] N({n.id}, {n.func}) -> {f}")
        n.func = f

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
