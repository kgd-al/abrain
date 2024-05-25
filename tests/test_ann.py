import logging
import math
from random import Random
from time import perf_counter
from typing import Tuple, List, Dict, Union, Callable

import pytest

from abrain import Config, Genome, Point2D, Point3D, ANN2D, ANN3D
from abrain.core.ann import ANNMonitor
from abrain.core.genome import logger as genome_logger

ANN = Union[ANN2D, ANN3D]
Point = Union[Point2D, Point3D]


def _ann_type(dimension):
    if dimension == 2:
        return ANN2D
    else:
        return ANN3D


def test_default_is_empty(dimension):
    # No default constructor
    with pytest.raises(TypeError):
        _ann_type(dimension)()

    ann, _, _ = _make_ann(dimension, 0, 0)
    with pytest.raises(ValueError):
        ann.neuronAt(ann.Point.null())


def _make_ann(dimension, mutations, seed,
              other_inputs=None, other_outputs=None) -> \
        Tuple[ANN, List[Point], List[Point]]:

    data = Genome.Data.create_for_eshn_cppn(dimension=dimension,
                                            seed=seed)
    genome = Genome.random(data)
    for _ in range(mutations):
        genome.mutate(data)

    rng = data.rng
    ann_t = _ann_type(dimension)

    def d(): return rng.randint(10, 20)
    def c(): return rng.uniform(-1, 1)

    if dimension == 2:
        def p(y): return ann_t.Point(c(), rng.uniform(.8*y, y))
    else:
        def p(y): return ann_t.Point(c(), y, c())

    def append(lhs, rhs):
        if rhs is not None:
            lhs.extend(rhs)

    inputs = [p(-1) for _ in range(d())]
    append(inputs, other_inputs)

    outputs = [p(1) for _ in range(d())]
    append(outputs, other_outputs)

    return ann_t.build(inputs, outputs, genome), inputs, outputs


# Force both dimensions to be tested even with small scale test
@pytest.mark.parametrize('_dimension', [2, 3])
def test_empty_perceptrons(_dimension, mutations, seed):
    n = 10

    def generate_stats():
        l_stats: Dict = {key: 0 for key in ['empty', 'perceptron', 'ann']}
        for _ in range(n):
            ann, _, _ = _make_ann(_dimension, mutations, seed)
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


@pytest.mark.parametrize('_dimension', [2, 3])
def test_invalid_genome(_dimension):
    ann_t = _ann_type(_dimension)
    genome = Genome.random(Genome.Data.create_for_generic_cppn(2, 2))
    with pytest.raises(ValueError):
        ann_t.build([], [], genome)

    genome = Genome.random(Genome.Data.create_for_eshn_cppn(5 - _dimension))
    with pytest.raises(ValueError):
        print()
        ann_t.build([], [], genome)


@pytest.mark.parametrize(
    'i, o',
    [
        pytest.param([0, 0], [1], id="in"),
        pytest.param([1], [0, 0], id="out"),
        pytest.param([0], [0], id="io"),
    ])
def test_invalid_duplicates(dimension, i, o):
    p_t = _ann_type(dimension).Point
    def p(c): return p_t(*[c for _ in range(dimension)])
    inputs = [p(c) for c in i]
    outputs = [p(c) for c in o]
    with pytest.raises(ValueError):
        print(f"ANN.build({inputs}, {outputs}, genome)")
        _make_ann(dimension, 0, 0, inputs, outputs)


def test_inspect_neurons(dimension, mutations, seed):
    ann, _, _ = _make_ann(dimension, mutations, seed)
    attrs = [k for k in ann.Neuron.__dict__.keys()
             if k[0].islower() and k[0].isalpha() and k != "links"]
    data = []
    widths = [len(k) for k in attrs]
    for i, n in enumerate(ann.neurons()):
        n: ANN.Neuron
        data.append([])
        for j, k in enumerate(attrs):
            attr = getattr(n, k)
            if isinstance(attr, Callable):
                s = str(attr())
            else:
                s = str(attr)
            data[i].append(s)
            widths[j] = max(widths[j], len(s))
    fmts = " ".join(f"{{:{w}}}" for w in widths)
    print(fmts.format(*attrs))
    for i in range(len(data)):
        print(fmts.format(*data[i]))


# .. todo:: implement (code exists in ann.cpp)
# def test_deepcopy():
#     original = _make_ann(1000, 0)
#     original.cop


def test_stats(dimension, mutations, seed):
    ann, _, _ = _make_ann(dimension, mutations, seed)
    ann_t = type(ann)

    stats = ann.stats()

    print("==")
    print(stats.hidden)
    print(len(ann.ibuffer()))
    print(len(ann.obuffer()))

    assert isinstance(stats.dict(), dict), f"{type(stats)=}"
    assert all([k in stats.dict().keys() for k in
                "depth hidden edges axons density utility iterations".split()])
    assert 0 <= stats.depth <= Config.iterations
    assert ann_t.max_hidden_neurons() == 2**(ann.DIMENSIONS * Config.maxDepth)
    assert 0 <= stats.hidden <= ann_t.max_hidden_neurons()
    assert ((ann.max_edges() == (len(ann.ibuffer()) * len(ann.obuffer()))) or
            (ann.max_edges() == (len(ann.ibuffer()) * stats.hidden
                                 + stats.hidden * stats.hidden
                                 + len(ann.obuffer()) * stats.hidden)))
    assert 0 <= stats.edges <= ann.max_edges()
    assert 0 <= stats.axons <= ann.max_edges() * math.sqrt(12)
    assert 0 <= stats.density <= 1
    assert 0 <= stats.utility <= 1
    assert 1 <= stats.iterations <= Config.iterations


def test_random_eval(dimension, mutations, seed):
    ann, inputs, outputs = _make_ann(dimension, mutations, seed)

    assert len(ann.ibuffer()) == len(inputs)
    assert len(ann.obuffer()) == len(outputs)
    assert ann.empty() or ann.stats().edges > 0

    assert ann.ibuffer() == ann.ibuffer()  # same objects
    assert ann.obuffer() == ann.obuffer()  #

    rng = Random(seed)
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


def _random_step(ann: ANN, rng: Random):
    inputs, outputs = ann.ibuffer(), ann.obuffer()
    for j in range(len(inputs)):
        inputs[j] = rng.uniform(-1, 1)

    ann(inputs, outputs, 1)
    return outputs


def test_reset(dimension, mutations, seed):
    ann, _, _ = _make_ann(dimension, mutations, seed)

    n = 1000
    all_outputs = [[], []]
    for i in range(2):
        ann.reset()
        rng = Random(seed)
        for _ in range(n):
            outputs = _random_step(ann, rng)
            all_outputs[i].append([outputs[i] for i in range(len(outputs))])

    assert all(all_outputs[0][i] == all_outputs[1][i]
               for i in range(n)), \
        "\n".join(f"{a} =?= {b}" for a, b
                  in zip(all_outputs[0], all_outputs[1]))


def test_view_neurons_png(mutations, seed, tmp_path):
    ann, _, _ = _make_ann(3, mutations, seed)

    file = f"{tmp_path}/ann.png"
    fig = ann.render3D()

    try:
        fig.write_image(file)
        print("Generated", file)    # pragma: no cover
    except Exception as e:  # pragma: no cover
        pytest.skip(f"Ignoring exceptions from unstable kaleido: "
                    f"{e=}, {type(e)=}")


def _time(_start=None):
    duration = perf_counter() - _start if _start else None
    _start = perf_counter()
    return duration, _start


@pytest.mark.parametrize('mutations', [10])
@pytest.mark.parametrize('seed', [1])
@pytest.mark.parametrize('with_labels', [True, False])
def test_view_neurons_interactive(mutations, seed, with_labels, tmp_path):
    _, start = _time()

    ann, inputs, _ = _make_ann(3, mutations, seed)
    duration, start = _time(start)
    print(f"Generating ANN(gen={mutations}, seed={seed}): {duration}s")

    labels = None
    if with_labels:
        labels = {}
        for i in inputs:
            labels[i] = f"Input{len(labels)}"

    fig = ann.render3D(labels)
    duration, start = _time(start)
    print(f"Preparing rendering: {duration}s")

    labels_str = "_with_labels" if with_labels else ""
    fig.write_html(f"{tmp_path}/interactive{labels_str}.ann.html")
    duration, start = _time(start)
    print(f"Writing: {duration}s")


@pytest.mark.parametrize('mutations', [10])
@pytest.mark.parametrize('seed', [1])
@pytest.mark.parametrize('with_labels', [True, False])
@pytest.mark.parametrize('with_neurons', [True, False])
@pytest.mark.parametrize('with_dynamics', [True, False])
@pytest.mark.parametrize('dt', [None, 0.1])
def test_view_neurons_dynamics(mutations, seed, with_labels,
                               with_neurons, with_dynamics, dt,
                               tmp_path):
    labels_str = "_with_labels" if with_labels else ""
    prefix = f"interactive{labels_str}"

    neurons_file = f"{prefix}.neurons.dat" if with_neurons else None
    dynamics_file = f"{prefix}.dynamics.dat" if with_dynamics else None

    _, start = _time()

    ann, inputs, _ = _make_ann(3, mutations, seed)
    duration, start = _time(start)
    print(f"Generating ANN(gen={mutations}, seed={seed}): {duration}s")

    labels = None
    if with_labels:
        labels = {}
        for i in inputs:
            labels[i] = f"Input{len(labels)}"

    n = 100
    print(f"Stepping {n} times")
    ann_monitor = ANNMonitor(
        ann=ann,
        labels=labels,
        folder=tmp_path,
        neurons_file=neurons_file,
        dynamics_file=dynamics_file,
        dt=dt
    )

    rng = Random(seed)
    for _ in range(n):
        _random_step(ann, rng)
        ann_monitor.step()

    ann_monitor.close()

    fig = ann.render3D(labels)
    duration, start = _time(start)
    print(f"Preparing rendering: {duration}s")

    ann_path = f"{tmp_path}/{prefix}.ann.html"
    fig.write_html(ann_path)
    duration, start = _time(start)
    print(f"Writing: {duration}s")
    print("Wrote to", ann_path)
