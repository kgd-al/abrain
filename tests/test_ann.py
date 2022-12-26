import math
from random import Random
from time import perf_counter
from typing import Tuple, List

import pytest

from abrain.core.ann import ANN, Point, plotly_render
from abrain.core.genome import Genome


def test_default_is_empty():
    ann = ANN()
    assert ann.empty()
    assert len(ann.ibuffer()) == 0
    assert len(ann.obuffer()) == 0

    stats = ann.stats()
    assert stats.depth == 0
    assert stats.edges == 0
    assert stats.axons == 0

    with pytest.raises(ValueError):
        p0 = Point(0, 0, 0)
        print("Testing no point at", p0)
        ann.neuronAt(p0)


def _make_ann(mutations, rng, other_inputs=None, other_outputs=None) -> Tuple[ANN, List[Point], List[Point]]:
    genome = Genome.random(rng)
    for _ in range(mutations):
        genome.mutate(rng)

    def d(): return rng.randint(10, 20)
    def c(): return rng.uniform(-1, 1)
    def p(y): return Point(c(), y, c())

    def append(lhs, rhs):
        if rhs is not None:
            lhs += [Point(x, y, z) for x, y, z in rhs]

    inputs = [p(-1) for _ in range(d())]
    append(inputs, other_inputs)

    outputs = [p(1) for _ in range(d())]
    append(outputs, other_outputs)

    return ANN.build(inputs, outputs, genome), inputs, outputs


@pytest.mark.parametrize(
    'inputs, outputs',
    [
        pytest.param([(0, 0, 0), (0, 0, 0)], [(1, 1, 1)], id="in"),
        pytest.param([(1, 1, 1)], [(0, 0, 0), (0, 0, 0)], id="out"),
        pytest.param([(0, 0, 0)], [(0, 0, 0)], id="io"),
    ])
def test_invalid_duplicates(inputs, outputs):
    with pytest.raises(ValueError):
        print(f"ANN.build({inputs}, {outputs}, genome)")
        _make_ann(0, Random(0), inputs, outputs)


# .. todo:: implement (code exists in ann.cpp)
# def test_deepcopy():
#     original = _make_ann(1000, 0)
#     original.cop


def test_random_eval(mutations, seed):
    rng = Random(seed)
    ann, inputs, outputs = _make_ann(mutations, rng)

    assert len(ann.ibuffer()) == len(inputs)
    assert len(ann.obuffer()) == len(outputs)
    assert ann.empty() or ann.stats().edges > 0

    assert ann.ibuffer() == ann.ibuffer()  # same objects
    assert ann.obuffer() == ann.obuffer()  #

    avg_output = 0
    for _ in range(1000):
        # inputs, outputs = ann.buffers()
        inputs, outputs = ann.ibuffer(), ann.obuffer()
        for j in range(len(inputs)):
            inputs[j] = rng.uniform(-1, 1)
        if hasattr(outputs, "set_to_nan"):  # pragma: no cover
            outputs.set_to_nan()

        ann(inputs, outputs, 1)

        for i in range(len(outputs)):
            avg_output += outputs[i]

        if hasattr(outputs, "set_to_nan"):  # pragma: no cover
            assert ann.empty() or not \
                any([math.isnan(outputs[i]) for i in range(len(outputs))])

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

    fig.write_html(f"{tmp_path}/interactive.ann.html")
    print(f"Writing: {time()}s")
