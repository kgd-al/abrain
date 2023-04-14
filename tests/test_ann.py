import logging
import math
from random import Random
from time import perf_counter
from typing import Tuple, List, Dict

import pytest

from abrain import Point, Config, Genome, ANN, plotly_render
from abrain.core.genome import logger as genome_logger


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


def _make_ann(mutations, rng, other_inputs=None, other_outputs=None) -> \
        Tuple[ANN, List[Point], List[Point]]:
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


def test_empty_perceptrons(mutations, seed):
    n = 10

    def generate_stats():
        rng = Random(seed)
        l_stats: Dict = {key: 0 for key in ['empty', 'perceptron', 'ann']}
        for _ in range(n):
            ann, _, _ = _make_ann(mutations, rng)
            l_stats['empty'] += ann.empty()
            l_stats['perceptron'] += ann.perceptron()
            l_stats['ann'] += (not ann.empty(strict=True))
            assert not ann.empty() or ann.empty(strict=True)
            assert not ann.perceptron() or ann.empty(strict=True)
            assert not (ann.empty() and ann.perceptron())
            assert not ann.perceptron() or Config.allowPerceptrons

        return l_stats

    genome_logger.setLevel(logging.CRITICAL)
    allow_perceptrons = bool(Config.allowPerceptrons)
    Config.allowPerceptrons = True
    stats_t = generate_stats()

    Config.allowPerceptrons = False
    stats_f = generate_stats()
    Config.allowPerceptrons = allow_perceptrons

    print(f"{'-':11s} {'True':10s} {'False':10s}")
    for k in stats_t:
        print(f"{k:10s}", end='')
        for stats in [stats_t, stats_f]:
            print(f" {100*stats[k]/n:10g}%", end='')
        print()

    for s in [stats_t, stats_f]:
        assert (sum(s.values()) == n)
    assert stats_t['empty'] <= stats_f['empty']
    assert stats_f['perceptron'] == 0


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

    try:
        fig.write_image(file)
        print("Generated", file)    # pragma: no cover
    except Exception as e:  # pragma: no cover
        pytest.skip(f"Ignoring exceptions from unstable kaleido: "
                    f"{e=}, {type(e)=}")


@pytest.mark.parametrize('mutations', [10])
@pytest.mark.parametrize('seed', [1])
@pytest.mark.parametrize('with_labels', [True, False])
def test_view_neurons_interactive(mutations, seed, with_labels, tmp_path):
    start = perf_counter()

    def time():
        nonlocal start
        duration = perf_counter() - start
        start = perf_counter()
        return duration

    rng = Random(seed)
    ann, inputs, _ = _make_ann(mutations, rng)
    print(f"Generating ANN(gen={mutations}, seed={seed}): {time()}s")

    labels = None
    if with_labels:
        labels = {}
        for i in inputs:
            labels[i] = f"Input{len(labels)}"

    fig = plotly_render(ann, labels)
    print(f"Preparing rendering: {time()}s")

    fig.write_html(f"{tmp_path}/interactive.ann.html")
    print(f"Writing: {time()}s")
