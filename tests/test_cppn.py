import array
import pydoc
import random
from itertools import chain, combinations
from random import Random
from typing import Callable, Tuple

import pytest
from pytest_steps import test_steps

from abrain import (
    Genome,
    CPPN as GenericCPPN, CPPN2D, CPPN3D,
)


def _make_cppn(cppn_type, seed, mutations=0):
    name = cppn_type.__name__
    if name == GenericCPPN.__name__:
        rng = Random(seed)
        data = Genome.Data.create_for_generic_cppn(
            seed=seed,
            inputs=rng.randint(2, 5),
            outputs=rng.randint(1, 3)
        )
    elif name == CPPN2D.__name__:
        data = Genome.Data.create_for_eshn_cppn(dimension=2, seed=seed)
    elif name == CPPN3D.__name__:
        data = Genome.Data.create_for_eshn_cppn(dimension=3, seed=seed)
    else:  # pragma: no cover
        raise ValueError(f"Invalid CPPN type {cppn_type}")

    genome = Genome.random(data)
    for _ in range(mutations):
        genome.mutate(data)
    try:
        cppn = cppn_type(genome)
    except Exception as e:  # pragma: no cover
        genome.to_dot(data, "foo.pdf", debug="depths,links")
        raise e

    return cppn, genome, data


def _uniform_point_factory(point_type):
    return lambda c: point_type(*[c for _ in range(point_type.DIMENSIONS)])


def _random_inputs(seed, cppn):
    rng = random.Random(seed)
    return [rng.uniform(-1, 1) for _ in range(cppn.n_inputs())]


def test_exists(cppn_type):
    cppn, _, _ = _make_cppn(cppn_type, 16)
    print(cppn)
    print(pydoc.render_doc(cppn))
    if hasattr(cppn_type, "Point"):
        print(pydoc.render_doc(cppn_type.Output))
        print(cppn_type.Output)
        print(cppn_type.Output.Weight)


def test_functions(cppn_type):
    for k, v in cppn_type.functions().items():
        print(f"{k}(0) = {v(0)}")


# Force all types to be tested even with small scale test
@pytest.mark.parametrize('_cppn_type', [GenericCPPN, CPPN2D, CPPN3D])
def test_create(_cppn_type, seed):
    cppn, _, _ = _make_cppn(_cppn_type, seed)
    print(cppn)
    assert cppn.n_hidden() == 0


def test_buffers(cppn_type, seed):
    cppn, _, _ = _make_cppn(cppn_type, seed)
    for buffer, size in [(cppn.ibuffer(), cppn.n_inputs()),
                         (cppn.obuffer(), cppn.n_outputs())]:
        print(pydoc.render_doc(buffer))
        assert len(buffer) == size
        print(buffer)
        print(buffer.__repr__())
        print([v for v in buffer])
        buffer[0] = 1
        buffer[1] += 2
        print([v for v in buffer])
        buffer[0] *= 4


# ===================================================
def __tester_generic_single_output(
        seed, mutations, args_maker: Callable[[GenericCPPN, list], Tuple]):
    cppn, _, _ = _make_cppn(GenericCPPN, seed, mutations)
    inputs = _random_inputs(seed, cppn)
    print(cppn.n_inputs(), cppn.n_inputs(True))
    print(len(inputs), inputs)

    with pytest.raises((ValueError, RuntimeError)):
        cppn(0, *args_maker(cppn, inputs + inputs))

    args = args_maker(cppn, inputs)
    for o in range(cppn.n_outputs()):
        values = set()
        for i in range(100):
            values.add(cppn(o, *args))
        assert len(values) == 1
# ===================================================


# ===================================================
def __tester_generic_all_outputs(
        seed, mutations, args_maker: Callable[[GenericCPPN, list], Tuple]):
    cppn, _, _ = _make_cppn(GenericCPPN, seed, mutations)
    inputs = _random_inputs(seed, cppn)
    args = args_maker(cppn, inputs)
    outputs = cppn.outputs()
    values = [set() for _ in outputs]

    with pytest.raises((ValueError, RuntimeError)):
        cppn(0, *args_maker(cppn, inputs + inputs))

    for i in range(100):
        cppn(outputs, *args)
        for j, o in enumerate(outputs):
            values[j].add(o)

    for v_set in values:
        assert len(v_set) == 1
# ===================================================


# ===================================================
def __list_to_input_buffer(cppn: GenericCPPN, lst: list):
    inputs = cppn.ibuffer()
    inputs[:] = lst
    return (inputs,)
# ===================================================


def test_generic_single_output_input_buffer(seed, mutations):
    __tester_generic_single_output(seed, mutations, __list_to_input_buffer)


def test_generic_single_output_input_list(seed, mutations):
    __tester_generic_single_output(seed, mutations, lambda _, lst: (lst,))


def test_generic_single_output_input_args(seed, mutations):
    __tester_generic_single_output(seed, mutations, lambda _, lst: tuple(lst))


def test_generic_all_outputs_input_buffer(seed, mutations):
    __tester_generic_all_outputs(seed, mutations, __list_to_input_buffer)


def test_generic_all_outputs_input_list(seed, mutations):
    __tester_generic_all_outputs(seed, mutations, lambda _, lst: (lst,))


def test_generic_all_outputs_input_args(seed, mutations):
    __tester_generic_all_outputs(seed, mutations, lambda _, lst: tuple(lst))


def test_nd_output_single(cppn_nd_type, seed):
    cppn, _, _ = _make_cppn(cppn_nd_type, seed)
    _p = _uniform_point_factory(cppn_nd_type.Point)
    p0, p1 = _p(0), _p(0)
    for o in cppn_nd_type.Output:
        values = set()
        for i in range(100):
            values.add(cppn(p0, p1, o))
        assert len(values) == 1


def test_nd_outputs_all(cppn_nd_type, seed):
    cppn, _, _ = _make_cppn(cppn_nd_type, seed)
    _p = _uniform_point_factory(cppn_nd_type.Point)
    p0, p1 = _p(0), _p(0)
    values = {k: set() for k in cppn_nd_type.Output}
    outputs = cppn.outputs()
    for i in range(100):
        cppn(p0, p1, outputs)
        for o in cppn_nd_type.Output:
            values[o].add(outputs[o])

    for o in cppn_nd_type.Output:
        assert len(values[o]) == 1


def test_nd_outputs_subset(cppn_nd_type, seed):
    cppn, _, _ = _make_cppn(cppn_nd_type, seed)
    _p = _uniform_point_factory(cppn_nd_type.Point)
    p0, p1 = _p(0), _p(0)
    values = {k: set() for k in cppn_nd_type.Output}
    subset = {cppn_nd_type.Output.Weight, cppn_nd_type.Output.LEO}
    outputs = cppn.outputs()
    for i in range(100):
        cppn(p0, p1, outputs, subset)
        for o in cppn_nd_type.Output:
            values[o].add(outputs[o])

    for o in cppn_nd_type.Output:
        assert len(values[o]) == 1


def test_outputs_equals(cppn_nd_type, seed):
    cppn, _, _ = _make_cppn(cppn_nd_type, seed)

    rng = Random(seed)

    def r():
        return rng.uniform(-1, 1)

    def p():
        return cppn_nd_type.Point(*[r() for _ in
                                    range(cppn_nd_type.DIMENSIONS)])

    def all_subsets(ss):
        return list(chain(*map(lambda x: combinations(ss, x),
                               range(0, len(ss)+1))))

    for _ in range(100):
        p0, p1 = p(), p()

        outputs_manual = []
        for o in cppn_nd_type.Output:
            outputs_manual.append(cppn(p0, p1, o))

        outputs = cppn.outputs()
        # Test all combinations
        outputs_subsets = []
        for subset in all_subsets(cppn_nd_type.Output):
            cppn(p0, p1, outputs, set(subset))
            outputs_subsets.append((subset,
                                    [outputs[i] for i in range(len(outputs))]))

        cppn(p0, p1, outputs)
        outputs_all = [outputs[i] for i in range(len(outputs))]

        def assert_near_equal(_lhs, _rhs):
            assert abs(_lhs - _rhs) < 1e-6

        for lhs, rhs in zip(outputs_manual, outputs_all):
            assert_near_equal(lhs, rhs)
        for subset, outputs_subset in outputs_subsets:
            for i in range(len(outputs_manual)):
                if cppn_nd_type.Output(i) in subset:
                    assert_near_equal(outputs_manual[i], outputs_subset[i])


sample_sizes = [10, 50, 100]


@test_steps('generate_cppn', *[f"sample_at_{s}" for s in sample_sizes])
def test_multiscale_sampling(cppn_nd_type, mutations, seed,
                             tmp_path_factory):  # pragma: no cover
    folder = tmp_path_factory.mktemp(
        numbered=False,
        basename=f"test_multiscale_sampling_"
                 f"s{seed}_m{mutations}_{cppn_nd_type.__name__}")

    cppn, genome, data = _make_cppn(cppn_nd_type, seed, mutations)

    genome.to_dot(data, f"{folder}/cppn.png")
    print('Generated', f"{folder}/cppn.png")

    yield 'generate_cppn'

    d = cppn_nd_type.Point.DIMENSIONS

    def _p(*args):
        return cppn_nd_type.Point(*(list(args)+[0 for _
                                                in range(d-len(args))]))

    dp = _p()
    for size in sample_sizes:
        def to_substrate_coord(index):
            return 2 * index / (size - 1) - 1

        data = array.array('B', [0, 0, 0] * size * size * 2)

        outputs = cppn.outputs()

        for i in range(size):
            x = to_substrate_coord(i)
            for j in range(size):
                y = to_substrate_coord(j)
                p = _p(x, y)
                for k, p0, p1 in [(0, p, dp), (1, dp, p)]:
                    cppn(p, dp, outputs)
                    ix = 3*(i+size*(j+size*k))
                    for o, v in enumerate(
                            [outputs[i] for i in range(len(outputs))]):
                        data[ix+o] = \
                            int(.5 * (max(-1.0, min(outputs[o], 1.0)) + 1)
                                * 255)

        for name, i in [('input', 0), ('output', 1)]:
            filename = f"{folder}/xy_{name}_plane_{size:03}x{size:03}.ppm"
            with open(filename, 'wb') as f:
                f.write(bytearray(f"P6\n{size} {size}\n255\n", 'ascii'))
                data.tofile(f)
            print('Generated', filename)

        yield f"sample_at_{size}"
