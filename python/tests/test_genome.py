from random import Random

import pytest

from core.genome import Genome


def test_default_genome_fails():
    with pytest.raises(AssertionError):
        Genome()


def test_random_genome():
    rng = Random()
    rng.seed(16)
    g = Genome.random(rng)
    print(g)
    assert "CPPN" in g.__repr__(), \
        f"Wrong output format for CPPN genome description. Got {g.__repr__()}"

    size_before = len(g.links)
    g._add_link(0, 0, 0)
    assert len(g.links) == size_before + 1, \
        f"Failed to append a link"
