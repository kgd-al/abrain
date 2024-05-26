import copy
import logging
import math
import pickle
import pydoc
import re
from pathlib import Path
from typing import Optional, Tuple

import pytest
from abrain import Config, Genome
from abrain.core.genome import logger as genome_logger

from _utils import genome_factory

logging.root.setLevel(logging.NOTSET)
logging.getLogger('graphviz').setLevel(logging.WARNING)
#
# print("[kgd-debug] Muted genome logger")
# genome_logger.setLevel(logging.CRITICAL)


###############################################################################
# Helpers
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


def _assert_valid_genome(g: Genome, data: Genome.Data, path: Path):
    try:
        n_ids = set([n.id for n in g.nodes])
        assert len(g.nodes) == len(n_ids), "Duplicate node ids!"
        assert len(g.links) == len(set([li.id for li in g.links])), \
            "Duplicate links ids!"
        assert not any(li.src == li.dst for li in g.links), \
            "Self-recurrent links!"

        # noinspection PyProtectedMember
        assert (g.outputs ==
                len([n.id for n in g.nodes if g._is_output(n.id)])), \
            "Missing output node(s)!"

        # noinspection PyProtectedMember
        linked_ids = set([nid for _l in g.links for nid in [_l.src, _l.dst]
                          if g._is_hidden(nid)])
        # noinspection PyProtectedMember
        hidden_ids = set([n.id for n in g.nodes if g._is_hidden(n.id)])
        assert len(hidden_ids) == len(linked_ids), "Missing nodes!"

        # Check that nodes/links are sorted
        for lst in [g.nodes, g.links]:
            for i in range(len(lst)-1):
                assert lst[i].id < lst[i+1].id, "Disordered ids!"

        def valid_nid(nid): return nid < data.id_manager.next_node_id()

        # noinspection PyProtectedMember
        degrees = g._compute_node_degrees()
        for node in g.nodes:
            d = degrees[node.id]
            # noinspection PyProtectedMember
            assert not g._is_hidden(node.id) \
                   or (d.i > 0 and d.o > 0), "Null degree for hidden node!"
            # noinspection PyProtectedMember
            assert not g._is_input(node.id) or d.i == 0, \
                "Incoming link for input node!"
            # noinspection PyProtectedMember
            assert not g._is_output(node.id) or d.o == 0, \
                "Outgoing link for output node!"
            assert valid_nid(node.id), "Invalid node id"
        for link in g.links:
            assert link.id < data.id_manager.next_link_id(), "ID too high!"
            assert valid_nid(link.src), "Invalid source id"
            assert valid_nid(link.dst), "Invalid destination id!"

        seen = set([n.id for n in g.nodes[:g.outputs]])
        links = {}
        queue = []
        for _l in g.links:
            links.setdefault(_l.dst, [])
            links[_l.dst].append(_l)
            # noinspection PyProtectedMember
            if g._is_output(_l.dst):
                queue.append(_l)
        while len(queue) > 0:
            _l = queue.pop(0)
            seen.add(_l.dst)
            # noinspection PyProtectedMember
            if not g._is_input(_l.src) and _l.src not in seen:
                for __l in links[_l.src]:
                    queue.append(__l)
        assert len(g.nodes) == len(seen), "Disconnected nodes!"

    except Exception as e:  # pragma: no cover
        genome_logger.debug("== Exception ==")
        genome_logger.debug(f">> {e.__class__}: {e}")
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

        g.to_dot(data, f"{path}/faulty_graph", "png", debug="all")
        genome_logger.info(f"Wrote faulty graph to {path}/"
                           f"faulty_graph.png")
        genome_logger.info(f"Wrote log to {path}/log")

        raise e


###############################################################################
# Generic tests
###############################################################################


def test_exists():
    data = Genome.Data.create_for_generic_cppn(5, 3, seed=16)
    g = Genome.random(data)
    print(g)
    print(pydoc.render_doc(g))


def test_default_genome_fails():
    with pytest.raises(AssertionError):
        Genome()


@pytest.mark.parametrize('with_bias', [True, False])
def test_create_genome(seed, with_bias):
    i, o = 5, 3
    data, _, _, g = genome_factory(seed, eshn=False, shape=(i, o),
                                   with_input_bias=with_bias)
    assert g.inputs == i + int(with_bias)
    assert g.outputs == o
    assert data.id_manager.next_node_id() == i + int(with_bias) + o

    print(g)
    assert "CPPN" in g.__repr__(), \
        f"Wrong output format for CPPN genome description. Got {g.__repr__()}"

    size_before = len(g.links)
    g._add_link(data, 0, 0, 0)
    assert len(g.links) == size_before + 1, "Failed to append a link"

    assert len(g.nodes) == o
    g._add_node(data.id_manager.next_node_id(), "id")
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

    data, _, _, g = genome_factory(
        seed, eshn=False, shape=(i, o), labels=labels,
        with_input_bias=with_bias)
    assert g.inputs == i + int(with_bias)
    assert g.outputs == o
    assert data.id_manager.next_node_id() == i + int(with_bias) + o


@pytest.mark.parametrize('with_bias', [True, False])
@pytest.mark.parametrize('shape_l', [
    (2, 2, ""),
    (2, 2, "a,b,c,A,B,C"),
])
def test_create_genome_create_with_labels_error(seed, with_bias, shape_l):
    i, o, labels = shape_l

    with pytest.raises(ValueError):
        genome_factory(seed, eshn=False, shape=(i, o), labels=labels,
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
    data, _, _, g = genome_factory(
        seed,
        eshn=True,
        dimension=dimension,
        with_input_length=input_length,
        with_input_bias=input_bias,
        with_leo=leo,
        with_output_bias=output_bias)
    assert g.inputs == 2 * dimension + input_length + input_bias
    assert g.outputs == 1 + leo + output_bias
    assert data.id_manager.next_node_id() == g.inputs + 1 + leo + output_bias


@pytest.mark.parametrize('args', [
    (4, True, True, True, True),
    (3, "foo", True, True, True),
    (3, True, "foo", True, True),
    (3, True, True, "foo", True),
    (3, True, True, True, "foo"),
])
def test_create_genome_eshn_error(seed,
                                  args):
    d, il, ib, leo, ob = args
    with pytest.raises(ValueError):
        genome_factory(
            seed=seed, eshn=True,
            dimension=d,
            with_input_length=il, with_input_bias=ib,
            with_leo=leo, with_output_bias=ob)


###############################################################################
# Mutation tests
###############################################################################

def save_function(data: Genome.Data, g: Genome, max_gen: int, path: Path,
                  capfd):
    def helper(gen: Optional[int] = None, title: Optional[str] = None):
        if gen is not None:
            setattr(helper, 'gen', gen)
        if not hasattr(helper, 'fmt'):
            setattr(helper, "fmt",
                    f"{path}/gen{{:0{math.ceil(math.log10(max_gen+1))}}}")

        g.to_dot(data=data,
                 path=getattr(helper, "fmt").format(getattr(helper, 'gen')),
                 ext="png", debug="depths;keepdot", title=title)

        if capfd is not None:  # pragma: no branch
            err = capfd.readouterr().err
            err = re.sub(
                r"dot: graph is too large for cairo-renderer bitmaps[^\n]*\n",
                "", err
            )
            assert len(err) == 0

        setattr(helper, 'gen', getattr(helper, 'gen') + 1)
    return helper


def test_mutate_genome_add(seed, eshn_genome, tmp_path, capfd):
    data, i, o, g = genome_factory(seed, eshn=eshn_genome)
    steps = 10

    save = save_function(data, g, 2*steps, tmp_path, capfd)
    with RatesGuard({"add_n": 1}):
        save(0)
        for _ in range(steps):
            g.mutate(data)
            save()
        assert len(g.nodes) == steps + o
        assert data.id_manager.next_node_id() == steps + o + i

    with RatesGuard({"add_l": 1}):
        nl = len(g.links)
        save()
        for _ in range(steps):
            g.mutate(data)
            save()
        assert len(g.links) == nl + steps


def test_mutate_genome_del_n(seed, eshn_genome, tmp_path, capfd):
    data, i, o, g = genome_factory(seed, eshn=eshn_genome)
    steps = 10

    save = save_function(data, g, 2*steps, tmp_path, capfd)
    with RatesGuard({"add_n": 1}):
        save(0)
        for _ in range(steps):
            g.mutate(data)
            save()
    with RatesGuard({"del_n": 1}):
        save()
        for _ in range(steps):
            g.mutate(data)
            save()
        assert len(g.nodes) == o
        assert data.id_manager.next_node_id() == steps + o + i


def test_mutate_genome_del_l(seed, eshn_genome, tmp_path, capfd):
    steps = 10
    data, _, _, g = genome_factory(seed, eshn=eshn_genome)

    save = save_function(data, g, 3*steps, tmp_path, capfd)
    save(0)

    with RatesGuard({"add_n": 1}):
        for i in range(steps):
            g.mutate(data)
            save()

    init_len = len(g.links)
    init_lid = data.id_manager.next_link_id()
    with RatesGuard({"add_l": 1}):
        for i in range(steps):
            g.mutate(data)
            save()
    assert len(g.links) == init_len+steps
    assert data.id_manager.next_link_id() <= init_lid+steps

    with RatesGuard({"del_l": 1}):
        for i in range(steps):
            g.mutate(data)
            save()
    assert len(g.links) == init_len
    assert data.id_manager.next_link_id() <= init_lid+steps


def test_mutate_genome_mut(seed, eshn_genome, tmp_path, capfd):
    data, _, o, g = genome_factory(seed, eshn=eshn_genome)
    steps = 10

    save = save_function(data, g, 2*steps, tmp_path, capfd)

    save(0)
    with RatesGuard({"add_n": 1}):
        g.mutate(data)
        save()

    assert len(g.nodes) == 1 + o

    with RatesGuard({"mut_f": 1}):
        for i in range(steps):
            def _f(): return [str(n.func) for n in g.nodes]
            func = _f()
            g.mutate(data)
            save()
            assert sum(a != b for a, b in zip(func, _f())) == 1

    with RatesGuard({"mut_w": 1}):
        steps = 10
        for i in range(steps):
            weights = [l_.weight for l_ in g.links]
            g.mutate(data)
            save()
            assert 1 == sum(b_w != a_w for b_w, a_w in
                            zip(weights, [l_.weight for l_ in g.links]))


def mutate_genome_topology(ad_rate, seed, eshn_genome, tmp_path, output, gens,
                           with_innovations, with_lineage, verbose, capfd) \
        -> Tuple[Genome.Data, Genome]:
    rates = {k: 0 if k.startswith("mut") else 1 for k in Config.mutationRates}
    rates["add_l"] = ad_rate
    rates["add_n"] = ad_rate
    rates["del_l"] = 1/ad_rate
    rates["del_n"] = 1/ad_rate

    title = f"add/del: {ad_rate}, seed: {seed}"

    with RatesGuard(rates):
        data, _, _, g = genome_factory(
            seed, eshn=eshn_genome,
            with_innovations=with_innovations, with_lineage=with_lineage)

        if output:
            save = save_function(data, g, gens, tmp_path, capfd)
            save(0, title)

        for j in range(gens):
            genome_logger.debug(f"Generation {j}")
            g.mutate(data)

            _assert_valid_genome(g, data, tmp_path)

            if output and verbose:  # pragma: no cover
                save(title=title)

        if output and not verbose:
            save(title=title)

        return data, g


def test_mutate_genome_topology_with_gvc_output(
        ad_rate, seed, generations, eshn_genome, tmp_path, verbose, capfd):
    mutate_genome_topology(ad_rate, seed, eshn_genome, tmp_path, output=True,
                           with_innovations=True, with_lineage=False,
                           gens=generations, verbose=verbose, capfd=capfd)


@pytest.mark.parametrize("with_innovations", [True, False])
@pytest.mark.parametrize("with_lineage", [True, False])
def test_mutate_genome_topology(ad_rate, seed, generations, eshn_genome,
                                with_innovations, with_lineage,
                                tmp_path, capfd):
    gens = 1000
    data, g = mutate_genome_topology(
        ad_rate, seed, eshn_genome, tmp_path,
        with_innovations=with_innovations,
        with_lineage=with_lineage,
        output=False, gens=generations, verbose=False,
        capfd=None)
    save_function(data, g, gens, tmp_path, capfd)(
        gens-1, title=f"gen{gens-1}, add/del: {ad_rate}, seed: {seed}")


@pytest.mark.parametrize("with_lineage", [True, False])
def test_mutate_genome_deepcopy(seed, eshn_genome, with_lineage):
    data, _, o, parent = genome_factory(seed, eshn=eshn_genome,
                                        with_lineage=with_lineage)
    steps = 10

    with RatesGuard({"add_n": 1}):
        for _ in range(steps):
            parent.mutate(data)
        child = parent.mutated(data)

        assert_equal(parent, parent.copy())

    assert len(parent.nodes) == len(child.nodes) - 1

    with RatesGuard({"add_n": 1}):
        for _ in range(steps):
            parent.mutate(data)

    with RatesGuard({"del_n": 1}):
        for _ in range(steps+1):
            child.mutate(data)
        assert len(child.nodes) == o

    assert len(parent.nodes) == 2*steps + o
    assert len(child.nodes) == o


###############################################################################
# Crossover tests
###############################################################################


def _test_crossover(data, parent_mutations, child_mutations,
                    tmp_path, *_):
    pop_size, gens = 10, 10

    with RatesGuard({"add_n": 1, "add_l": 1, "del_n": .5, "del_l": .5}):
        population = [Genome.random(data) for _ in range(pop_size)]
        for _ in range(parent_mutations):
            for g in population:
                g.mutate(data, n=1)

        for i in range(pop_size):
            p = population[i]

            for j in range(i+1, pop_size):
                p_ = population[j]
                print(i, j)
                assert Genome.distance(p, p_) > 0
                assert Genome.distance(p, p_, [0]*6) == 0

            assert Genome.distance(p, p) == 0

        for gen in range(gens):
            new_pop = []

            while len(new_pop) < len(population):
                parents = data.rng.sample(population, 2)
                child = Genome.crossover(
                    *parents, data=data,
                    bias=data.rng.choice([0, 1, "length"]))
                for _ in range(child_mutations):
                    child.mutate(data)
                _assert_valid_genome(child, data, tmp_path)
                new_pop.append(child)
            population = new_pop

    with pytest.raises(ValueError):
        Genome.crossover(*data.rng.sample(population, 2), data=data,
                         bias="wrong")


@pytest.mark.parametrize("child_mutations", [0, 2])
def test_generic_cppn_crossover(seed, cppn_shape, mutations, child_mutations,
                                tmp_path, verbose):
    i, o = cppn_shape
    data = Genome.Data.create_for_generic_cppn(
        inputs=i, outputs=o,
        seed=seed+1,
        with_innovations=True,
        with_lineage=True
    )

    _test_crossover(data, mutations, child_mutations,
                    tmp_path, verbose)


@pytest.mark.parametrize("child_mutations", [0, 2])
def test_eshn_cppn_crossover(seed, dimension, mutations, child_mutations,
                             tmp_path, verbose):
    data = Genome.Data.create_for_eshn_cppn(
        dimension=dimension,
        seed=seed,
        with_innovations=True,
        with_lineage=True
    )

    _test_crossover(data, mutations, child_mutations,
                    tmp_path, verbose)


###############################################################################
# Serialization tests
###############################################################################


def assert_equal(lhs: Genome, rhs: Genome):
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


def _simple_genome(seed, with_id):
    data = Genome.Data.create_for_generic_cppn(5, 3,
                                               seed=seed,
                                               with_lineage=with_id)
    genome = Genome.random(data)
    for _ in range(10):
        genome.mutate(data)
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
def test___copy_genome_factory(seed, with_id):
    genome = _simple_genome(seed, with_id)
    copied = copy.copy(genome)
    assert_equal(genome, copied)


@pytest.mark.parametrize('with_id', [True, False])
def test___deepcopy_genome_factory(seed, with_id):
    genome = _simple_genome(seed, with_id)
    copied = copy.deepcopy(genome)
    assert_equal(genome, copied)
