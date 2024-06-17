import copy
import pickle

import pytest
from abrain import Genome


def assert_genomes_equal(lhs: Genome, rhs: Genome):
    assert lhs is not rhs
    assert lhs.nodes is not rhs.nodes
    assert lhs.links is not rhs.nodes

    assert lhs.inputs == rhs.inputs
    assert lhs.outputs == rhs.outputs
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


def genome_factory(seed, eshn: bool, shape=None, labels=None, n=1, **kwargs):
    if eshn:
        kwargs.setdefault("dimension", 3)
        data = Genome.Data.create_for_eshn_cppn(seed=seed, **kwargs)
    else:
        i, o = shape or (5, 3)
        data = Genome.Data.create_for_generic_cppn(
            i, o, labels=labels, seed=seed, **kwargs
        )
    g = [Genome.random(data) for _ in range(n)]
    return data, g[0].inputs, g[0].outputs, g if n > 1 else g[0]


SERIALIZER_FUNCTIONS = [
    pytest.param(lambda o: o.__class__.from_json(o.to_json()), id="json"),
    pytest.param(lambda o: pickle.loads(pickle.dumps(o)), id="pickle"),
    pytest.param(lambda o: o.copy(), id="copy"),
    pytest.param(lambda o: copy.copy(o), id="__copy__"),
    pytest.param(lambda o: copy.deepcopy(o), id="__deepcopy__")
]
