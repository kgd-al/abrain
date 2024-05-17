import copy
import logging
import math
import pickle
import pprint
import pydoc
from pathlib import Path
from random import Random
from typing import Optional

import pytest

from _utils import genome_factory
from abrain import Config, Genome, GIDManager, Innovations
from abrain.core.genome import logger as genome_logger


def __debug_print_innov(i: Innovations):
    pprint.pprint(i.to_json())


def __debug_print_genome(g: Genome):
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
    assert i.nextNodeID() == 0
    assert i.nextLinkID() == 0


@pytest.mark.parametrize('with_bias', [True, False])
def test_create_single_genome(seed, with_bias, cppn_shape):
    innovations = Innovations()
    assert innovations.empty()
    nn, nl = innovations.size()
    assert nn == 0 and nl == 0
    assert innovations.nextNodeID() == 0
    assert innovations.nextLinkID() == 0

    i, o = cppn_shape

    _, _, _, g = genome_factory(seed, eshn=False, shape=(i, o),
                                with_input_bias=with_bias,
                                innovations=innovations)

    g_links = len(g.links)
    nn, nl = innovations.size()
    assert nn == 0 and nl == g_links
    assert innovations.nextNodeID() == i + int(with_bias) + o
    assert innovations.nextLinkID() == g_links

    i_json = innovations.to_json()
    assert "nodes" in i_json
    assert len(i_json["nodes"]) == nn
    assert "links" in i_json
    assert len(i_json["links"]) == nl
    assert "nextNode" in i_json
    assert i_json["nextNode"] == innovations.nextNodeID()
    assert "nextLink" in i_json
    assert i_json["nextLink"] == innovations.nextLinkID()


@pytest.mark.parametrize('with_bias', [True, False])
def test_create_multiple_genomes(seed, with_bias, cppn_shape):
    innovations = Innovations()

    i, o = cppn_shape

    genomes = []
    for j in range(10):
        _, _, _, g = genome_factory(seed+j, eshn=False, shape=(i, o),
                                    with_input_bias=with_bias,
                                    innovations=innovations)
        genomes.append(g)

    g_links = len(set(_l.id for _g in genomes for _l in _g.links))
    nn, nl = innovations.size()
    assert nn == 0 and nl == g_links
    assert innovations.nextNodeID() == i + int(with_bias) + o
    assert innovations.nextLinkID() == g_links

    i_json = innovations.to_json()
    assert "nodes" in i_json
    assert len(i_json["nodes"]) == nn
    assert "links" in i_json
    assert len(i_json["links"]) == nl
    assert "nextNode" in i_json
    assert i_json["nextNode"] == innovations.nextNodeID()
    assert "nextLink" in i_json
    assert i_json["nextLink"] == innovations.nextLinkID()


def test_mutate_multiple_genomes(seed, mutations, cppn_shape):
    innovations = Innovations()

    i, o = cppn_shape

    genomes = []
    for j in range(10):
        rng, _, _, g = genome_factory(
            seed+j, eshn=False, shape=(i, o),
            innovations=innovations)
        for _ in range(mutations):
            g.mutate(rng, innovations)
        genomes.append(g)

    g_links = len(set(_l.id for _g in genomes for _l in _g.links ))
    g_nodes = len(set(n.id for _g in genomes for n in _g.nodes
                      if _g._is_hidden(n.id)))
    nn, nl = innovations.size()
    assert nn >= g_nodes
    assert nl >= g_links
    assert innovations.nextNodeID() >= i + o + g_nodes
    assert innovations.nextLinkID() >= g_links

    i_json = innovations.to_json()
    assert "nodes" in i_json
    assert len(i_json["nodes"]) == nn
    assert "links" in i_json
    assert len(i_json["links"]) == nl
    assert "nextNode" in i_json
    assert i_json["nextNode"] == innovations.nextNodeID()
    assert "nextLink" in i_json
    assert i_json["nextLink"] == innovations.nextLinkID()


# TODO: Test with specific mutations

###############################################################################
# Serialization tests
###############################################################################

def _simple_innovation(seed):
    innovations = Innovations()
    for j in range(10):
        rng, _, _, g = genome_factory(
            seed+j, eshn=False, shape=(5, 3),
            innovations=innovations)
        for _ in range(10):
            g.mutate(rng, innovations)
    return innovations


def assert_equal(lhs: Innovations, rhs: Innovations):
    assert lhs.nextNodeID() == rhs.nextNodeID()
    assert lhs.nextLinkID() == rhs.nextLinkID()
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
