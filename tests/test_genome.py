import copy
import itertools
import logging
import math
import pickle
import pprint
import pydoc
import string
from pathlib import Path
from random import Random
from typing import Optional

import pytest

from _utils import assert_equal
from abrain import Config, Genome, GIDManager
from abrain.core.genome import logger as genome_logger

logging.root.setLevel(logging.NOTSET)
logging.getLogger('graphviz').setLevel(logging.WARNING)


def __genome(seed, eshn: bool, shape=None, labels=None, **kwargs):
    rng = Random(seed)
    if eshn:
        kwargs.setdefault("dimension", 3)
        g = Genome.eshn_random(rng, **kwargs)
        return rng, g.inputs, g.outputs, g
    else:
        i, o = shape or (5, 3)
        return rng, i, o, Genome.random(rng, i, o, labels=labels, **kwargs)


###############################################################################
# Generic tests
###############################################################################


def test_exists():
    rng = Random(16)
    g = Genome.random(rng, 5, 3)
    print(g)
    print(pydoc.render_doc(g))


def test_default_genome_fails():
    with pytest.raises(AssertionError):
        Genome()


@pytest.mark.parametrize('with_bias', [True, False])
def test_create_genome(seed, with_bias):
    _, i, o, g = __genome(seed, eshn=False, with_input_bias=with_bias)
    assert g.inputs == i + int(with_bias)
    assert g.outputs == o
    assert g.nextNodeID == o

    print(g)
    assert "CPPN" in g.__repr__(), \
        f"Wrong output format for CPPN genome description. Got {g.__repr__()}"

    size_before = len(g.links)
    g._add_link(0, 0, 0)
    assert len(g.links) == size_before + 1, "Failed to append a link"

    assert len(g.nodes) == o
    g._add_node("id")
    assert len(g.nodes) == o+1

    print(f"Full details of {g}")
    print("\tNodes:")
    for node in g.nodes:
        print(f"\t\t{node}")
    print("\tLinks:")
    for link in g.links:
        print(f"\t\t{link}")


@pytest.mark.parametrize('with_bias', [True, False])
@pytest.mark.parametrize('shape_l', [
    (5, 3, "a,b,c,d,e,A,B,C"),
    (3, 5, "a,b,c,A,B,C,D,E"),
    (5, 3, "a,b,c,d,e,,,A,B,C"),
    (5, 3, "a,b,c,d,e,A,B,C,"),
])
def test_create_genome_create_with_labels(seed, with_bias, shape_l):
    i, o, labels = shape_l

    _, _, _, g = __genome(seed, eshn=False, shape=(i, o), labels=labels,
                          with_input_bias=with_bias)
    assert g.inputs == i + int(with_bias)
    assert g.outputs == o
    assert g.nextNodeID == o

@pytest.mark.parametrize('with_bias', [True, False])
@pytest.mark.parametrize('shape_l', [
    (2, 2, ""),
    (2, 2, "a,b,c,A,B,C"),
])
def test_create_genome_create_with_labels_error(seed, with_bias, shape_l):
    i, o, labels = shape_l

    with pytest.raises(ValueError):
        __genome(seed, eshn=False, shape=(i, o), labels=labels,
                 with_input_bias=with_bias)


@pytest.mark.parametrize('dimension', [2, 3])
@pytest.mark.parametrize('input_bias', [True, False])
@pytest.mark.parametrize('input_length', [True, False])
@pytest.mark.parametrize('leo', [True, False])
@pytest.mark.parametrize('output_bias', [True, False])
def test_create_genome_eshn(seed,
                            dimension,
                            input_bias, input_length,
                            leo, output_bias):
    _, _, _, g = __genome(
        seed,
        eshn=True,
        dimension=dimension,
        with_input_length=input_length,
        with_input_bias=input_bias,
        with_leo=leo,
        with_output_bias=output_bias)
    assert g.inputs == 2*dimension + input_length + input_bias
    assert g.outputs == 1 + leo + output_bias
    assert g.nextNodeID == 1 + leo + output_bias


@pytest.mark.parametrize('args', [
    (4, True, True, True, True),
    (3, "foo", True, True, True),
    (3, True, "foo", True, True),
    (3, True, True, "foo", True),
    (3, True, True, True, "foo"),
])
def test_create_genome_eshn_error(seed,
                                  args):
    with pytest.raises(ValueError):
        __genome(
            seed,
            eshn=True,
            dimension=args[0],
            with_input_length=args[1],
            with_input_bias=args[2],
            with_leo=args[3],
            with_output_bias=args[4])


###############################################################################
# Mutation tests
###############################################################################

class RatesGuard:
    def __init__(self, rates: dict):
        self.override = rates

    def __enter__(self):
        # print("Backed mutation rates:", Config.mutationRates)
        self.backup = {k: v for k, v in Config.mutationRates.items()}
        for k in Config.mutationRates.keys():
            Config.mutationRates[k] = 0
        for k, v in self.override.items():
            Config.mutationRates[k] = v
        # print("Replaced mutation rates with:", Config.mutationRates)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for k, v in self.backup.items():
            Config.mutationRates[k] = v
        # print("Restored mutation rates:", Config.mutationRates)


def save_function(g: Genome, max_gen: int, path: Path, capfd):
    def helper(gen: Optional[int] = None, title: Optional[str] = None):
        if gen is not None:
            setattr(helper, 'gen', gen)
        if not hasattr(helper, 'fmt'):
            setattr(helper, "fmt", f"{path}/gen{{:0{math.ceil(math.log10(max_gen+1))}}}")

        g.to_dot(path=getattr(helper, "fmt").format(getattr(helper, 'gen')),
                 ext="png", debug="depths;keepdot", title=title)

        if capfd is not None:  # pragma: no branch
            captured = capfd.readouterr()
            assert len(captured.err) == 0

        setattr(helper, 'gen', getattr(helper, 'gen') + 1)
    return helper


def test_mutate_genome_add(seed, eshn_genome, tmp_path, capfd):
    rng, _, o, g = __genome(seed, eshn=eshn_genome)
    steps = 10

    save = save_function(g, 2*steps, tmp_path, capfd)
    with RatesGuard({"add_n": 1}):
        save(0)
        for i in range(steps):
            g.mutate(rng)
            save()
        assert len(g.nodes) == steps + o
        assert g.nextNodeID == steps + o

    with RatesGuard({"add_l": 1}):
        nl = len(g.links)
        save()
        for i in range(steps):
            g.mutate(rng)
            save()
        assert len(g.links) == nl + steps


def test_mutate_genome_del_n(seed, eshn_genome, tmp_path, capfd):
    rng, _, o, g = __genome(seed, eshn=eshn_genome)
    steps = 10

    save = save_function(g, 2*steps, tmp_path, capfd)
    with RatesGuard({"add_n": 1}):
        save(0)
        for i in range(steps):
            g.mutate(rng)
            save()
    with RatesGuard({"del_n": 1}):
        save()
        for i in range(steps):
            g.mutate(rng)
            save()
        assert len(g.nodes) == o
        assert g.nextNodeID == steps + o


def test_mutate_genome_del_l(seed, eshn_genome, tmp_path, capfd):
    steps = 10
    rng, _, _, g = __genome(seed, eshn=eshn_genome)

    save = save_function(g, 3*steps, tmp_path, capfd)
    save(0)

    with RatesGuard({"add_n": 1}):
        for i in range(steps):
            g.mutate(rng)
            save()

    init_len = len(g.links)
    init_lid = g.nextLinkID
    with RatesGuard({"add_l": 1}):
        for i in range(steps):
            g.mutate(rng)
            save()
    assert len(g.links) == init_len+steps
    assert g.nextLinkID == init_lid+steps

    with RatesGuard({"del_l": 1}):
        for i in range(steps):
            g.mutate(rng)
            save()
    assert len(g.links) == init_len
    assert g.nextLinkID == init_lid+steps


def test_mutate_genome_mut(seed, eshn_genome, tmp_path, capfd):
    rng, _, o, g = __genome(seed, eshn=eshn_genome)
    steps = 10

    save = save_function(g, 2*steps, tmp_path, capfd)

    save(0)
    with RatesGuard({"add_n": 1}):
        g.mutate(rng)
        save()

    assert len(g.nodes) == 1 + o

    with RatesGuard({"mut_f": 1}):
        for i in range(steps):
            def _f(): return [str(n.func) for n in g.nodes]
            func = _f()
            g.mutate(rng)
            save()
            assert sum(a != b for a, b in zip(func, _f())) == 1

    with RatesGuard({"mut_w": 1}):
        steps = 10
        for i in range(steps):
            weights = [l_.weight for l_ in g.links]
            g.mutate(rng)
            save()
            assert any(b_w != a_w for b_w, a_w in
                       zip(weights, [l_.weight for l_ in g.links]))


def mutate_genome_topology(ad_rate, seed, eshn_genome, tmp_path, output, gens,
                           verbose, capfd) \
        -> Genome:
    rates = {k: 0 if k.startswith("mut") else 1 for k in Config.mutationRates}
    rates["add_l"] = ad_rate
    rates["add_n"] = ad_rate
    rates["del_l"] = 1/ad_rate
    rates["del_n"] = 1/ad_rate

    title = f"add/del: {ad_rate}, seed: {seed}"

    with RatesGuard(rates):
        rng, _, _, g = __genome(seed, eshn=eshn_genome)

        if output:
            save = save_function(g, gens, tmp_path, capfd)
            save(0, title)

        for j in range(gens):
            try:
                genome_logger.debug(f"Generation {j}")
                g.mutate(rng)
                assert len(g.nodes) == len(set([n.id for n in g.nodes]))
                assert len(g.links) == len(set([(li.src, li.dst) for li
                                                in g.links]))
                assert not any(li.src == li.dst for li in g.links)

                def valid_nid(nid): return nid < \
                    g.nextNodeID + g.inputs + g.outputs
                # noinspection PyProtectedMember
                degrees = g._compute_node_degrees()
                for node in g.nodes:
                    d = degrees[node.id]
                    # noinspection PyProtectedMember
                    assert not g._is_hidden(node.id) \
                           or (d.i > 0 and d.o > 0)
                    assert valid_nid(node.id)
                for link in g.links:
                    assert link.id < g.nextLinkID
                    assert valid_nid(link.src)
                    assert valid_nid(link.dst)

            except Exception as e:  # pragma: no cover
                genome_logger.debug("== Exception ==")
                genome_logger.debug(f">> {e}")
                genome_logger.debug("== Post fail details ==")
                genome_logger.debug("Nodes:")
                # noinspection PyProtectedMember
                degrees = g._compute_node_degrees()
                # noinspection PyProtectedMember
                depths = g._compute_node_depths(g.links)
                for node in sorted(g.nodes, key=lambda n_: n_.id):
                    d = degrees[node.id]
                    genome_logger.debug(f"\t{node} i:{d.i}, o:{d.o},"
                                        f" d:{depths[node.id]}")
                genome_logger.debug("Links:")
                for link in sorted(g.links, key=lambda l_: l_.id):
                    genome_logger.debug(f"\t{link}")

                g.to_dot(f"{tmp_path}/faulty_graph", "png", debug=True)
                genome_logger.info(f"Wrote faulty graph to {tmp_path}/"
                                   f"faulty_graph.png")
                genome_logger.info(f"Wrote log to {tmp_path}/log")

                raise e

            if output and verbose:  # pragma: no cover
                save(title=title)

        if output and not verbose:
            save(title=title)

        return g


def test_mutate_genome_topology_with_gvc_output(
        ad_rate, seed, eshn_genome, tmp_path, verbose, capfd):
    mutate_genome_topology(ad_rate, seed, eshn_genome, tmp_path, output=True, gens=100,
                           verbose=verbose, capfd=capfd)


def test_mutate_genome_topology(ad_rate, seed, eshn_genome, tmp_path, capfd):
    gens = 1000
    g = mutate_genome_topology(ad_rate, seed, eshn_genome, tmp_path, output=False,
                               gens=gens, verbose=False, capfd=None)
    save_function(g, gens, tmp_path, capfd)(
        gens-1, title=f"gen{gens-1}, add/del: {ad_rate}, seed: {seed}")


def test_mutate_genome_deepcopy(seed, eshn_genome):
    rng, _, o, parent = __genome(seed, eshn=eshn_genome)
    steps = 10

    with RatesGuard({"add_n": 1}):
        for _ in range(steps):
            parent.mutate(rng)
        child = parent.mutated(rng)

        assert_equal(parent, parent.copy())

    assert len(parent.nodes) == len(child.nodes) - 1

    with RatesGuard({"add_n": 1}):
        for _ in range(steps):
            parent.mutate(rng)

    with RatesGuard({"del_n": 1}):
        for _ in range(steps+1):
            child.mutate(rng)
        assert len(child.nodes) == o

    assert len(parent.nodes) == 2*steps + o
    assert len(child.nodes) == o


###############################################################################
# Serialization tests
###############################################################################

def _simple_genome(seed, with_id):
    rng = Random(seed)
    id_manager = GIDManager() if with_id else None
    genome = Genome.random(rng, 5, 3, id_manager=id_manager)
    for _ in range(10):
        genome.mutate(rng)
    return genome


@pytest.mark.parametrize('with_id', [True, False])
def test_pickle_genome(seed, with_id):
    genome = _simple_genome(seed, with_id)
    roundabout = pickle.loads(pickle.dumps(genome))
    assert_equal(genome, roundabout)


@pytest.mark.parametrize('with_id', [True, False])
def test_json_genome(seed, with_id):
    genome = _simple_genome(seed, with_id)
    roundabout = Genome.from_json(genome.to_json())
    assert_equal(genome, roundabout)


@pytest.mark.parametrize('with_id', [True, False])
def test_copy_genome(seed, with_id):
    genome = _simple_genome(seed, with_id)
    copied = genome.copy()
    assert_equal(genome, copied)


@pytest.mark.parametrize('with_id', [True, False])
def test___copy___genome(seed, with_id):
    genome = _simple_genome(seed, with_id)
    copied = copy.copy(genome)
    assert_equal(genome, copied)


@pytest.mark.parametrize('with_id', [True, False])
def test___deepcopy___genome(seed, with_id):
    genome = _simple_genome(seed, with_id)
    copied = copy.deepcopy(genome)
    assert_equal(genome, copied)
