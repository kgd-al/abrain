import copy
import pickle
import pprint
import pydoc

import pytest

from _utils import genome_factory
from abrain import Genome
from abrain.core.genome import Innovations


def __debug_print_innov(i: Innovations):  # pragma: no cover
    pprint.pprint(i.to_json())


def __debug_print_genome(g: Genome):  # pragma: no cover
    print(g)
    print("Nodes:")
    for n in g.nodes:
        print(f"> N{n.id}: {n.func}")
    print("Links:")
    for _l in g.links:
        print(f"> L{_l.id}: {_l.src} -> {_l.dst} ({_l.weight})")


###############################################################################
# Generic tests
###############################################################################


def test_exists():
    i = Innovations()
    print(i)
    print(pydoc.render_doc(i))


def test_create():
    i = Innovations()
    assert i.empty()

    nn, nl = i.size()
    assert nn == 0
    assert nl == 0
    assert i.next_node_id() == 0
    assert i.next_link_id() == 0


def test_ids():
    i = Innovations()
    i.initialize(0)
    assert i.empty()

    steps = 10
    for _ in range(steps):
        assert i.get_node_id(0, 0) == 0
        assert i.get_link_id(0, 0) == 0

    assert i.node_id(0, 0) == 0
    assert i.link_id(0, 0) == 0
    assert i.size() == (1, 1)

    i.initialize(0)
    assert i.empty()

    steps = 10
    for j in range(steps):
        assert i.next_node_id() == j
        assert i.new_node_id(0, 0) == j
        assert i.next_link_id() == j
        assert i.new_link_id(0, 0) == j
    assert i.node_id(0, 0) == steps-1
    assert i.link_id(0, 0) == steps-1

    assert i.node_id(1, 1) == i.NOT_FOUND
    assert i.link_id(1, 1) == i.NOT_FOUND


@pytest.mark.parametrize('with_bias', [True, False])
def test_create_single_genome(seed, with_bias, cppn_shape):
    i, o = cppn_shape

    data, _, _, g = genome_factory(
        seed=seed, eshn=False, shape=(i, o),
        with_input_bias=with_bias, with_innovations=True)

    innovations = data.id_manager

    g_links = len(g.links)
    nn, nl = innovations.size()
    assert nn == 0 and nl == g_links
    assert innovations.next_node_id() == i + int(with_bias) + o
    assert innovations.next_link_id() == g_links

    i_json = innovations.to_json()
    assert "nodes" in i_json
    assert len(i_json["nodes"]) == nn
    assert "links" in i_json
    assert len(i_json["links"]) == nl
    assert "nextNode" in i_json
    assert i_json["nextNode"] == innovations.next_node_id()
    assert "nextLink" in i_json
    assert i_json["nextLink"] == innovations.next_link_id()


@pytest.mark.parametrize('with_bias', [True, False])
def test_create_multiple_genomes(seed, with_bias, cppn_shape):
    i, o = cppn_shape

    data, _, _, genomes = genome_factory(
        n=10, seed=seed, eshn=False, shape=(i, o),
        with_input_bias=with_bias, with_innovations=True)

    innovations = data.id_manager

    g_links = len(set(_l.id for _g in genomes for _l in _g.links))
    nn, nl = innovations.size()
    assert nn == 0 and nl == g_links
    assert innovations.next_node_id() == i + int(with_bias) + o
    assert innovations.next_link_id() == g_links

    i_json = innovations.to_json()
    assert "nodes" in i_json
    assert len(i_json["nodes"]) == nn
    assert "links" in i_json
    assert len(i_json["links"]) == nl
    assert "nextNode" in i_json
    assert i_json["nextNode"] == innovations.next_node_id()
    assert "nextLink" in i_json
    assert i_json["nextLink"] == innovations.next_link_id()


def test_mutate_multiple_genomes(seed, mutations, cppn_shape):
    i, o = cppn_shape

    data, _, _, genomes = genome_factory(
        n=10, seed=seed, eshn=False, shape=(i, o),
        with_innovations=True)

    for g in genomes:
        for _ in range(mutations):
            g.mutate(data)

    innovations = data.id_manager
    g_links = len(set(_l.id for _g in genomes for _l in _g.links))
    g_nodes = len(set(n.id for _g in genomes for n in _g.nodes
                      if _g._is_hidden(n.id)))
    nn, nl = innovations.size()
    assert nn >= g_nodes
    assert nl >= g_links
    assert innovations.next_node_id() >= i + o + g_nodes
    assert innovations.next_link_id() >= g_links

    i_json = innovations.to_json()
    assert "nodes" in i_json
    assert len(i_json["nodes"]) == nn
    assert "links" in i_json
    assert len(i_json["links"]) == nl
    assert "nextNode" in i_json
    assert i_json["nextNode"] == innovations.next_node_id()
    assert "nextLink" in i_json
    assert i_json["nextLink"] == innovations.next_link_id()


# TODO: Test with specific mutations

###############################################################################
# Serialization tests
###############################################################################

def _simple_innovation(seed):
    data, _, _, genomes = genome_factory(
        n=10, seed=seed, eshn=False, shape=(5, 3),
        with_innovations=True)

    for g in genomes:
        for _ in range(10):
            g.mutate(data)

    return data.id_manager


def assert_equal(lhs: Innovations, rhs: Innovations):
    assert lhs.next_node_id() == rhs.next_node_id()
    assert lhs.next_link_id() == rhs.next_link_id()
    assert lhs.__repr__() == rhs.__repr__()
    assert lhs.empty() == rhs.empty()
    assert lhs.size() == rhs.size()
    assert lhs.to_json() == rhs.to_json()


def test_pickle_genome(seed):
    innovations = _simple_innovation(seed)
    roundabout = pickle.loads(pickle.dumps(innovations))
    assert_equal(innovations, roundabout)


def test_json_genome(seed):
    innovations = _simple_innovation(seed)
    roundabout = Innovations.from_json(innovations.to_json())
    assert_equal(innovations, roundabout)


def test_copy_genome(seed):
    innovations = _simple_innovation(seed)
    copied = innovations.copy()
    assert_equal(innovations, copied)


def test___copy___genome(seed):
    innovations = _simple_innovation(seed)
    copied = copy.copy(innovations)
    assert_equal(innovations, copied)


def test___deepcopy___genome(seed):
    innovations = _simple_innovation(seed)
    copied = copy.deepcopy(innovations)
    assert_equal(innovations, copied)
