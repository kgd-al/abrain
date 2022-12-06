import math
from random import Random
from time import perf_counter

import pytest

from core.ann import ANN, Point, plotly_render
from core.genome import Genome


def test_default_is_empty():
    ann = ANN()
    assert ann.empty()
    assert len(ann.inputs()) == 0
    assert len(ann.outputs()) == 0

    stats = ann.stats()
    assert stats.depth == 0
    assert stats.edges == 0
    assert stats.axons == 0


def _make_ann(mutations, rng):
    genome = Genome.random(rng)
    for _ in range(mutations):
        genome.mutate(rng)

    def d(): return rng.randint(10, 20)
    def c(): return rng.uniform(-1, 1)
    def p(y): return Point(c(), y, c())
    inputs = [p(-1) for _ in range(d())]
    outputs = [p(1) for _ in range(d())]

    return ANN.build(inputs, outputs, genome), inputs, outputs


def test_random_eval(mutations, seed):
    rng = Random(seed)
    ann, inputs, outputs = _make_ann(mutations, rng)

    assert len(ann.inputs()) == len(inputs)
    assert len(ann.outputs()) == len(outputs)
    assert ann.empty() or ann.stats().edges > 0

    avg_output = 0
    for i in range(1000):
        inputs, outputs = ann.inputs(), ann.outputs()
        for j in range(len(inputs)):
            inputs[j] = rng.uniform(-1, 1)
        for j in range(len(outputs)):
            outputs[j] = float('nan')

        ann(inputs, outputs, 1)

        avg_output += sum(outputs)

        assert ann.inputs() != inputs
        assert ann.empty() or not any([math.isnan(o) for o in outputs])

    assert ann.empty() or sum != 0


def test_view_neurons_png(mutations, seed, tmp_path):
    rng = Random(seed)
    ann, _, _ = _make_ann(mutations, rng)

    file = f"{tmp_path}/ann.png"
    fig = plotly_render(ann)
    fig.write_image(file)
    print("Generated", file)


@pytest.mark.parametrize('mutations', [10])
@pytest.mark.parametrize('seed', [1])
def test_view_neurons_interactive(mutations, seed, tmp_path):
    start = perf_counter()

    def time():
        nonlocal start
        duration = perf_counter() - start
        start = perf_counter()
        return duration

    rng = Random(seed)
    ann, _, _ = _make_ann(mutations, rng)
    print(f"Generating ANN(gen={mutations}, seed={seed}): {time()}s")

    fig = plotly_render(ann)
    print(f"Preparing rendering: {time()}s")

    fig.show()
    print(f"Showing: {time()}s")
