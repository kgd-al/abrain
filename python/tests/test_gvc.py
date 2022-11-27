import os.path
from random import Random

from core.genome import Genome


def test_random_graph(tmp_path):
    rng = Random(16)
    g = Genome.random(rng)
    output = g.to_dot(f"{tmp_path}/cppn_test")

    assert output.endswith(".pdf")  # Default extension pdf
    assert os.path.exists(output)   # File should exist
    print("Generated", output)

    for i in range(16):
        rng = Random(i)
        g = Genome.random(rng)
        output = g.to_dot(f"{tmp_path}/cppn_test_{i}", "png")
        assert output.endswith(".png")
        assert os.path.exists(output)   # File should exist
        print("Generated", output)
