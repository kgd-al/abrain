from random import Random

from abrain import Genome


def assert_genomes_equal(lhs: Genome, rhs: Genome):
    assert lhs is not rhs
    assert lhs.nodes is not rhs.nodes
    assert lhs.links is not rhs.nodes

    assert len(lhs.nodes) == len(rhs.nodes)
    assert len(lhs.links) == len(rhs.links)

    for lhs_n, rhs_n in zip(lhs.nodes, rhs.nodes):
        assert lhs_n is not rhs_n
        assert lhs_n.id == rhs_n.id
        assert lhs_n.func == rhs_n.func

    for lhs_l, rhs_l in zip(lhs.links, rhs.links):
        assert lhs_l is not rhs_l
        assert lhs_l.id == rhs_l.id
        assert lhs_l.src == rhs_l.src
        assert lhs_l.dst == rhs_l.dst
        assert lhs_l.weight == rhs_l.weight

    assert lhs.nextNodeID == rhs.nextNodeID
    assert lhs.nextLinkID == rhs.nextLinkID


def genome_factory(seed, eshn: bool, shape=None, labels=None, **kwargs):
    rng = Random(seed)
    if eshn:
        kwargs.setdefault("dimension", 3)
        g = Genome.eshn_random(rng, **kwargs)
        return rng, g.inputs, g.outputs, g
    else:
        i, o = shape or (5, 3)
        return rng, i, o, Genome.random(rng, i, o, labels=labels, **kwargs)
