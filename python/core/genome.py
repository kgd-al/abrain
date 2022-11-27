"""
Test documentation for genome file (module?)
"""

from inspect import getsourcefile
from os.path import abspath, dirname
from random import Random

from graphviz import Digraph
from pyne_cpp.config import Config
from pyne_cpp.genotype import CPPNData


class Genome(CPPNData):
    """Genome class for ES-HyperNEAT.

    A simple collection of Node/Link.
    Can only be created via random init or copy
    """

    __private_key = object()
    """Private key for preventing default-constructed CPPNs
    """

    nextNodeID: int
    """ID of the next node to be created
    """

    def __init__(self, key=None):
        CPPNData.__init__(self)
        assert (key == Genome.__private_key), \
            "Genome objects must be created via random, deepcopy or reproduction methods"
        self.nextNodeID = CPPNData.INPUTS + CPPNData.OUTPUTS;

    def __repr__(self):
        return f"CPPN:{Genome.INPUTS}:{Genome.OUTPUTS}({len(self.nodes)}, {len(self.links)})"

    def _add_link(self, src: int, dst: int, weight: float):
        """Add a link to the genome

        :param src: index of the source node in [0, :py:attr:`core.genome.Genome.inputs`]
        :param dst: index of the destination node in [Genome.inputs,Genome.outputs]
        :param weight: connection weight in [-1,1]
        """
        self.links.append(CPPNData.Link(src, dst, weight))

    @staticmethod
    def random(rng: Random):
        """Create a random CPPN with boolean initialization

        :param rng: The source of randomness
        :return: A random CPPN genome
        """
        g = Genome(Genome.__private_key)

        wb = Config.weightBounds
        for i in range(CPPNData.INPUTS):
            for j in range(CPPNData.OUTPUTS):
                if rng.random() < .5:
                    w = rng.random() * (wb.rndMax - wb.rndMin) + wb.rndMin
                    g._add_link(i, j+CPPNData.INPUTS, w)

        return g

    def to_dot(self, path: str, ext: str = "pdf") -> str:
        dot = Digraph('cppn')

        g_i, g_h, g_o, g_ol = [Digraph(n) for n in ['inputs', 'hidden', 'outputs', 'o_labels']]

        dot.attr(rankdir='BT')
        g_i.attr(rank="source")
        g_o.attr(rank="same")
        g_ol.attr(rank="sink")
        ix_to_name = {}

        # input nodes
        i_labels = Config.cppnInputNames
        for i in range(self.INPUTS):
            name = f"I{i}"
            ix_to_name[i] = name
            label=i_labels[i]
            if '_' in label:
                j = label.find('_')
                label = label[:j] + "<SUB>" + label[j+1:] + "</SUB>"
            g_i.node(name, f"<{label}>", shape='plain')

        ps_path = abspath(dirname(getsourcefile(lambda: 0)) + "/../../ps")
        o_labels = Config.cppnOutputNames
        for i in range(self.OUTPUTS):
            name = f"O{i}"
            ix_to_name[i+self.INPUTS] = name
            f = Config.outputFunctions[i]
            g_o.node(name, "",
                     width="0.5in", height="0.5in", margin="0", fixedsize="shape",
                     image=f"{ps_path}/{f}.png")
            g_ol.node(name + "_l", shape="plain", label=o_labels[i])
            dot.edge(name, name + "_l")

        for i, n in enumerate(self.nodes):
            name = f"H{i}"
            ix_to_name[i+self.INPUTS+self.OUTPUTS] = name
            g_h.node(name, "",
                     width="0.5in", height="0.5in", margin="0", fixedsize="shape",
                     image=f"{ps_path}/{n.func}.png")

        for i, l in enumerate(self.links):
            if l.weight == 0:
                continue

            w = abs(l.weight) / Config.weightBounds.max
            dot.edge(ix_to_name[l.src], ix_to_name[l.dst],
                     color="red" if l.weight < 0 else "black",
                     penwidth=str(3.75*w + .25))

        # enforce i/o order
        for i in range(self.INPUTS-1):
            dot.edge(ix_to_name[i], ix_to_name[i+1], style="invis")
        for i in range(self.INPUTS, self.INPUTS+self.OUTPUTS-1):
            dot.edge(ix_to_name[i], ix_to_name[i + 1], style="invis")

        # Register sub-graphs
        for g in [g_i, g_h, g_o, g_ol]:
            dot.subgraph(g)

        return dot.render(path, format=ext, cleanup=True)
