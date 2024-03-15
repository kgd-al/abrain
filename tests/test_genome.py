import copy
import logging
import pickle
import pydoc
from pathlib import Path
from random import Random
from typing import Optional

import pytest

from _utils import assert_equal
from abrain.core.config import Config
from abrain.core.genome import Genome, logger as genome_logger, GIDManager

logging.root.setLevel(logging.NOTSET)
logging.getLogger('graphviz').setLevel(logging.WARNING)

###############################################################################
# Generic tests
###############################################################################


def test_exists():
    rng = Random(16)
    g = Genome.random(rng)
    print(g)
    print(pydoc.render_doc(g))


def test_default_genome_fails():
    with pytest.raises(AssertionError):
        Genome()


def test_random_genome():
    g = Genome.random(Random(16))
    print(g)
    assert "CPPN" in g.__repr__(), \
        f"Wrong output format for CPPN genome description. Got {g.__repr__()}"

    size_before = len(g.links)
    g._add_link(0, 0, 0)
    assert len(g.links) == size_before + 1, "Failed to append a link"

    g._add_node("id")
    assert len(g.nodes) == 1

    print(f"Full details of {g}")
    print("\tNodes:")
    for node in g.nodes:
        print(f"\t\t{node}")
    print("\tLinks:")
    for link in g.links:
        print(f"\t\t{link}")


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


def test_mutate_genome_create(tmp_path):
    g = Genome.random(Random(16))
    assert g.nextNodeID == 0


def save_function(g: Genome, path: Path, capfd):
    def helper(gen: Optional[int] = None, title: Optional[str] = None):
        if gen is not None:
            setattr(helper, 'gen', gen)

        g.to_dot(path=f"{path}/gen{getattr(helper, 'gen'):02}",
                 ext="png", debug="depths", title=title)

        captured = capfd.readouterr()
        assert len(captured.err) == 0

        setattr(helper, 'gen', getattr(helper, 'gen') + 1)
    return helper


def test_mutate_genome_add(tmp_path, capfd):
    rng = Random(16)
    g = Genome.random(rng)
    save = save_function(g, tmp_path, capfd)
    with RatesGuard({"add_n": 1}):
        save(0)
        steps = 10
        for i in range(steps):
            g.mutate(rng)
            save()
        assert len(g.nodes) == steps
        assert g.nextNodeID == steps

    with RatesGuard({"add_l": 1}):
        nl = len(g.links)
        save()
        steps = 10
        for i in range(steps):
            g.mutate(rng)
            save()
        assert len(g.links) == nl + steps


def test_mutate_genome_del_n(tmp_path, capfd):
    steps = 10
    rng = Random(16)
    g = Genome.random(rng)
    save = save_function(g, tmp_path, capfd)
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
        assert len(g.nodes) == 0
        assert g.nextNodeID == steps


def test_mutate_genome_del_l(tmp_path, capfd):
    steps = 10
    rng = Random(16)
    g = Genome.random(rng)
    save = save_function(g, tmp_path, capfd)
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


def test_mutate_genome_mut(tmp_path, capfd):
    rng = Random(16)
    g = Genome.random(rng)
    save = save_function(g, tmp_path, capfd)

    save(0)
    with RatesGuard({"add_n": 1}):
        g.mutate(rng)
        save()

    assert len(g.nodes) == 1

    with RatesGuard({"mut_f": 1}):
        steps = 10
        for i in range(steps):
            func = str(next(iter(g.nodes)).func)
            g.mutate(rng)
            save()
            assert str(next(iter(g.nodes)).func) != func

    with RatesGuard({"mut_w": 1}):
        steps = 10
        for i in range(steps):
            weights = [l_.weight for l_ in g.links]
            g.mutate(rng)
            save()
            assert any(b_w != a_w for b_w, a_w in
                       zip(weights, [l_.weight for l_ in g.links]))


def mutate_genome_topology(ad_rate, seed, tmp_path, output, gens, capfd) \
        -> Genome:
    rates = {k: 0 if k.startswith("mut") else 1 for k in Config.mutationRates}
    rates["add_l"] = ad_rate
    rates["add_n"] = ad_rate
    rates["del_l"] = 1/ad_rate
    rates["del_n"] = 1/ad_rate

    title = f"add/del: {ad_rate}, seed: {seed}"

    with RatesGuard(rates):
        rng = Random(seed)
        g = Genome.random(rng)

        if output:
            save = save_function(g, tmp_path, capfd)
            save(0, title)

        for j in range(gens):
            try:
                g.mutate(rng)
                assert len(g.nodes) == len(set([n.id for n in g.nodes]))
                assert len(g.links) == len(set([(li.src, li.dst) for li
                                                in g.links]))
                assert not any(li.src == li.dst for li in g.links)

                def valid_nid(nid): return nid < \
                    g.nextNodeID + g.INPUTS + g.OUTPUTS
                # noinspection PyProtectedMember
                degrees = Genome._compute_node_degrees(g.links)
                for node in g.nodes:
                    d = degrees[node.id]
                    assert not Genome._is_hidden(node.id) \
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
                degrees = Genome._compute_node_degrees(g.links)
                # noinspection PyProtectedMember
                depths = Genome._compute_node_depths(g.links)
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

            if output:
                save(title=title)

        return g


def test_mutate_genome_topology_with_gvc_output(ad_rate, seed, tmp_path,
                                                capfd):
    mutate_genome_topology(ad_rate, seed, tmp_path, output=True, gens=100,
                           capfd=capfd)


def test_mutate_genome_topology(ad_rate, seed, tmp_path, capfd):
    gens = 1000
    g = mutate_genome_topology(ad_rate, seed, tmp_path, output=False,
                               gens=gens, capfd=None)
    save_function(g, tmp_path, capfd)(gens-1,
                                      title=f"gen{gens-1}, add/del: {ad_rate},"
                                            f" seed: {seed}")


def test_mutate_genome_deepcopy(seed):
    rng = Random(seed)
    parent = Genome.random(rng)

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
        assert len(child.nodes) == 0

    assert len(parent.nodes) == 2*steps
    assert len(child.nodes) == 0


###############################################################################
# Serialization tests
###############################################################################

def _simple_genome(seed, with_id):
    rng = Random(seed)
    id_manager = GIDManager() if with_id else None
    genome = Genome.random(rng, id_manager)
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
