"""
Test documentation for genome file (module?)
"""
import json
import logging
import pathlib
from collections import namedtuple
from collections.abc import Iterable
from random import Random
from shutil import which
from typing import Dict, List

from graphviz import Digraph
from importlib_resources import files
from pyrecord import Record

from .._cpp.config import Config
from .._cpp.genotype import CPPNData as _CPPNData

logger = logging.getLogger(__name__ + ".mutations")


dot_found = (which("dot") is not None)


class Genome(_CPPNData):
    """Genome class for ES-HyperNEAT.

    A simple collection of Node/Link.
    Can only be created via random init or copy
    """

    __private_key = object()
    """Private key for preventing default-constructed CPPNs
    """

    def __init__(self, key=None):
        _CPPNData.__init__(self)
        assert (key == Genome.__private_key), \
            "Genome objects must be created via random, deepcopy or" \
            " reproduction methods"

    def __repr__(self):
        return f"CPPN:{Genome.INPUTS}:{Genome.OUTPUTS}([{len(self.nodes)}," \
               f" {self.nextNodeID}], [{len(self.links)},{self.nextLinkID}])"

    @staticmethod
    def _is_input(nid: int):
        return nid < Genome.INPUTS

    @staticmethod
    def _is_output(nid: int):
        return Genome.INPUTS <= nid < Genome.INPUTS + Genome.OUTPUTS

    @staticmethod
    def _is_hidden(nid: int):
        return Genome.INPUTS + Genome.OUTPUTS <= nid

    ###########################################################################
    # Public mutation interface
    ###########################################################################

    def mutate(self, rng: Random) -> None:
        """Mutate (in-place) this genome

        :param rng: The source of randomness
        """

        # Get the list of all nodes
        N = namedtuple('Node', 'id type')
        offset = 0
        all_nodes: List[N] = []
        for i in range(self.INPUTS):
            all_nodes.append(N(i + offset, "I"))
        offset += self.INPUTS
        for i in range(self.OUTPUTS):
            all_nodes.append(N(i + offset, "O"))
        offset += self.OUTPUTS
        for i, n in enumerate(self.nodes):
            all_nodes.append(N(n.id, "H"))

        depths = Genome._compute_node_depths(self.links)

        # For now assume that all valid links can be created.
        # Existing ones are pruned afterwards
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
        node_degrees = Genome._compute_node_degrees(self.links)

        # Ensure that link removal does not produce in-/output less nodes
        # (except for I/O nodes, obviously)
        non_essential_links: List[int] = []
        for i, link in enumerate(self.links):
            if (Genome._is_input(link.src)
                and Genome._is_output(link.dst)) \
                    or (node_degrees[link.src].o > 1
                        and node_degrees[link.dst].i > 1):
                non_essential_links.append(i)

        # Simple nodes are (hidden) nodes of the form:
        # A -> X -> B which can be replaced by A -> B
        simples_nodes: List[int] = []
        for nid, d in node_degrees.items():
            if d.i == 1 and d.o == 1:
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

    def mutated(self, rng: Random) -> 'Genome':
        """Return a mutated (copied) version of this genome

        :param rng: the source of randomness
        """
        copy = self.copy()
        copy.mutate(rng)
        return copy

    def copy(self) -> 'Genome':
        """Return a perfect (deep)copy of this genome"""
        copy = Genome(Genome.__private_key)

        copy.nodes = self.nodes
        copy.links = self.links
        copy.nextNodeID = self.nextNodeID
        copy.nextLinkID = self.nextLinkID

        return copy

    @staticmethod
    def random(rng: Random) -> 'Genome':
        """Create a random CPPN with boolean initialization

        :param rng: The source of randomness
        :return: A random CPPN genome
        """
        g = Genome(Genome.__private_key)

        for i in range(_CPPNData.INPUTS):
            for j in range(_CPPNData.OUTPUTS):
                if rng.random() < .5:
                    g._add_link(i, j + _CPPNData.INPUTS,
                                Genome.__random_link_weight(rng))

        return g

    ###########################################################################
    # Public json/binary interface
    ###########################################################################

    def to_json(self) -> str:
        """Return a string json representation of this object"""
        return json.dumps(dict(
            nodes=[(n.id, n.func) for n in self.nodes],
            links=[(l_.id, l_.src, l_.dst, l_.weight) for l_ in self.links],
            NID=self.nextNodeID,
            LID=self.nextLinkID
        ))

    @staticmethod
    def from_json(json_str: str) -> 'Genome':
        """Recreate a Genome from a string json representation
        """
        data = json.loads(json_str)
        g = Genome(Genome.__private_key)
        g.nodes = _CPPNData.Nodes([
            _CPPNData.Node(n[0], n[1]) for n in data["nodes"]
        ])
        g.links = _CPPNData.Links([
            _CPPNData.Link(*[d[0], d[1], d[2], d[3]]) for d in data["links"]
        ])
        g.nextNodeID = data["NID"]
        g.nextLinkID = data["LID"]

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

    def to_dot(self, path: str, ext: str = "pdf", title: str = None,
               debug=None) -> str:
        """Produce a graphical representation of this genome

        .. note:: Missing functions images in nodes and/or lots of
            warning message are likely caused by misplaced image files.
            In doubt perform a full reinstall

        Args:
            path: The path to write to
            ext: The rendering format to use
            title: Optional title for the graph (e.g. for generational info)
            debug: Print more information on the graph.
                Special values:

                - 'depth' will display every nodes' depth

        Raises:
            OSError: if the `dot` program is not available (not installed, on
                the path and executable)
        """
        if not dot_found:
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

        # input nodes
        i_labels = Config.cppnInputNames
        for i in range(self.INPUTS):
            name = f"I{i}"
            ix_to_name[i] = name
            label = i_labels[i]
            if '_' in label:
                j = label.find('_')
                label = label[:j] + "<SUB>" + label[j + 1:] + "</SUB>"
            g_i.node(name, f"<{label}>", shape='plaintext',
                     xlabel=node_x_label(i))

        img_format = "svg" if vectorial else "png"

        def img_file(func):
            p = (files('abrain') / f"core/functions/{func}.{img_format}")
            return str(p.resolve())

        o_labels = Config.cppnOutputNames
        for i in range(self.OUTPUTS):
            name = f"O{i}"
            ix_to_name[i + self.INPUTS] = name
            f = Config.outputFunctions[i]
            g_o.node(name, "",
                     width="0.5in", height="0.5in", margin="0",
                     fixedsize="shape",
                     image=img_file(f),
                     xlabel=node_x_label(i + self.INPUTS))
            g_ol.node(name + "_l", shape="plaintext", label=o_labels[i])
            dot.edge(name, name + "_l")

        for i, n in enumerate(self.nodes):
            name = f"H{n.id}"
            ix_to_name[n.id] = name
            label = name if debug is not None else ""
            g_h.node(name, label,
                     width="0.5in", height="0.5in", margin="0",
                     fixedsize="shape",
                     image=img_file(n.func),
                     xlabel=node_x_label(n.id))

        for i, l in enumerate(self.links):
            if l.weight == 0:  # pragma: no cover
                continue  # Highly unlikely

            w = abs(l.weight) / Config.cppnWeightBounds.max
            dot.edge(ix_to_name[l.src], ix_to_name[l.dst],
                     color="red" if l.weight < 0 else "black",
                     penwidth=str(3.75 * w + .25),
                     xlabel=x_label(
                         l.id if debug is not None and "links" in debug
                         else None))

        # enforce i/o order
        for i in range(self.INPUTS - 1):
            dot.edge(ix_to_name[i], ix_to_name[i + 1], style="invis")
        for i in range(self.INPUTS, self.INPUTS + self.OUTPUTS - 1):
            dot.edge(ix_to_name[i], ix_to_name[i + 1], style="invis")

        # Register sub-graphs
        for g in [g_i, g_h, g_o, g_ol]:
            dot.subgraph(g)

        cleanup = False if debug is not None and "keepdot" in debug else True

        return dot.render(path, format=ext, cleanup=cleanup)

    ###########################################################################
    # Private mutation interface
    ###########################################################################

    @staticmethod
    def __random_node_function(rng: Random):
        return rng.choice(Config.functionSet)

    def _add_node(self, func: str) -> int:
        nid = self.nextNodeID + self.INPUTS + self.OUTPUTS
        self.nodes.append(_CPPNData.Node(nid, func))
        self.nextNodeID += 1
        return nid

    @staticmethod
    def __random_link_weight(rng: Random):
        return rng.uniform(Config.cppnWeightBounds.rndMin,
                           Config.cppnWeightBounds.rndMax)

    def _add_link(self, src: int, dst: int, weight: float) -> int:
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
        return lid

    def __random_add_node(self, rng: Random) -> None:
        i = rng.randrange(0, len(self.links))
        link: _CPPNData.Link = self.links[i]
        nid = self._add_node(Genome.__random_node_function(rng))

        src, dst, weight = link.src, link.dst, link.weight
        logger.info(f"[add_n] {src} -> {dst} >> {src} -> {nid} -> {dst}")

        self._add_link(src, nid, 1)
        self._add_link(nid, dst, weight)
        self.links.pop(i)

    def __random_del_node(self, rng: Random, candidates) -> None:
        nid = rng.choice(candidates)
        n_ix, n = next((n_ix, n) for n_ix, n in   # pragma: no branch
                       enumerate(self.nodes) if n.id == nid)
        i_links = [(j, link) for j, link in enumerate(self.links)
                   if link.dst == n.id]
        o_links = [(j, link) for j, link in enumerate(self.links)
                   if link.src == n.id]
        assert len(i_links) == 1 and len(o_links) == 1

        j_i, i_l = i_links[0][0], i_links[0][1]
        j_o, o_l = o_links[0][0], o_links[0][1]
        logger.info(f"[del_n] {i_l.src} -> {n.id} -> {o_l.dst} >>"
                    f" {i_l.src} -> {o_l.dst}")

        if sum(li.src == i_l.src and li.dst == o_l.dst for li in
               self.links) == 0 \
                and i_l.src != o_l.dst:
            # Link does not exist already and is not auto-recurrent -> add it
            self._add_link(i_l.src, o_l.dst, o_l.weight)
        self.nodes.pop(n_ix)
        self.links.pop(j_i)
        if j_i < j_o:  # index may have been made invalid
            j_o -= 1
        self.links.pop(j_o)

    def __random_add_link(self, rng: Random, candidates) -> None:
        link = rng.choice(list(candidates))
        weight = Genome.__random_link_weight(rng)

        logger.info(f"[add_l] {link.src} -({weight})-> {link.dst}")

        self._add_link(link.src, link.dst, weight)

    def __random_del_link(self, rng: Random, candidates) -> None:
        i = candidates[rng.randrange(0, len(candidates))]
        logger.info(f"[del_l] {self.links[i].src} -> {self.links[i].dst}")
        self.links.pop(i)

    def __random_mutate_weight(self, rng: Random) -> None:
        link = rng.choice(self.links)
        bounds = Config.cppnWeightBounds
        delta = 0
        while delta == 0:
            delta = rng.gauss(0, bounds.stddev)
        w = max(bounds.min, min(link.weight + delta, bounds.max))

        logger.info(f"[mut_w] {link.src} -({link.weight})-> {link.dst} >> {w}")

        link.weight = w

    def __random_mutate_func(self, rng: Random) -> None:
        n = rng.choice(self.nodes)
        f = rng.choice([f for f in Config.functionSet if not f == n.func])
        logger.info(f"[mut_f] N({n.id}, {n.func}) -> {f}")
        n.func = f

    @staticmethod
    def _compute_node_depths(links_original):
        depths = {}
        links = [l_ for l_ in links_original]

        def recurse(nid: int, depth: int):
            if nid < Genome.INPUTS:
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

        for i in range(Genome.OUTPUTS):
            recurse(i + Genome.INPUTS, 0)

        for i in range(Genome.INPUTS):
            depths.setdefault(i, 0)

        return depths

    @staticmethod
    def _compute_node_degrees(links):
        d = Record.create_type("D", "i", "o", i=0, o=0)

        node_degrees: Dict[int, d] = {}
        for link in links:
            node_degrees.setdefault(link.src, d()).o += 1
            node_degrees.setdefault(link.dst, d()).i += 1

        return node_degrees
