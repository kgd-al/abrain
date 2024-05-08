import array
import pydoc
from itertools import chain, combinations
from random import Random

from abrain._cpp.phenotype import (  # noqa
    CPPN as GenericCPPN, CPPN2D, CPPN3D,
    Point2D, Point3D,
)
from pytest_steps import test_steps

from abrain.core.genome import Genome


def _make_generic_cppn(cppn_type, seed, mutations=0):
    rng = Random(seed)
    genome = Genome.random(rng, 5, 3)
    for _ in range(mutations):
        genome.mutate(rng)
    try:
        cppn = cppn_type(genome)
    except Exception as e:  # pragma: no cover
        genome.to_dot("foo.pdf", debug="depths,links")
        raise e
    return cppn, genome


def _make_eshn_cppn(cppn_type, seed, mutations=0):
    rng = Random(seed)
    genome = Genome.eshn_random(rng, cppn_type.DIMENSIONS)
    for _ in range(mutations):
        genome.mutate(rng)
    try:
        cppn = cppn_type(genome)
    except Exception as e:  # pragma: no cover
        genome.to_dot("foo.pdf", debug="depths,links")
        raise e
    return cppn, genome


def _make_cppn(cppn_type, seed, mutations=0):
    match cppn_type.__name__:
        case GenericCPPN.__name__: return _make_generic_cppn(cppn_type, seed, mutations)
        case CPPN2D.__name__: return _make_eshn_cppn(cppn_type, seed, mutations)
        case CPPN3D.__name__: return _make_eshn_cppn(cppn_type, seed, mutations)


def _uniform_point_factory(point_type):
    return lambda c: point_type(*[c for _ in range(point_type.DIMENSIONS)])


def test_exists(cppn_type):
    cppn, _ = _make_cppn(cppn_type, 16)
    print(cppn)
    print(pydoc.render_doc(cppn))
    if hasattr(cppn_type, "Point"):
        print(pydoc.render_doc(cppn_type.Output))
        print(cppn_type.Output.Weight)
        print(cppn_type.OUTPUTS_LIST)
        assert cppn_type.Output.Weight in cppn_type.OUTPUTS_LIST


def test_functions(cppn_type):
    for k, v in cppn_type.functions().items():
        print(f"{k}(0) = {v(0)}")


def test_create(cppn_type, seed):
    cppn = _make_cppn(cppn_type, seed)
    print(cppn)


def test_output_single(cppn_type, seed):
    cppn, _ = _make_cppn(cppn_type, seed)
    _p = _uniform_point_factory(cppn_type.Point)
    p0, p1 = _p(0), _p(0)
    for o in CPPN.OUTPUTS_LIST:
        values = set()
        for i in range(100):
            values.add(cppn(p0, p1, o))
        assert len(values) == 1


def test_outputs_all(cppn_type, seed):
    cppn, _ = _make_cppn(cppn_type, seed)
    _p = _uniform_point_factory(cppn_type.Point)
    p0, p1 = _p(0), _p(0)
    values = {k: set() for k in CPPN.OUTPUTS_LIST}
    outputs = CPPN.outputs()
    for i in range(100):
        cppn(p0, p1, outputs)
        for o in CPPN.OUTPUTS_LIST:
            values[o].add(outputs[o])

    for o in CPPN.OUTPUTS_LIST:
        assert len(values[o]) == 1


def test_outputs_subset(cppn_type, seed):
    cppn, _ = _make_cppn(cppn_type, seed)
    _p = _uniform_point_factory(cppn_type.Point)
    p0, p1 = _p(0), _p(0)
    values = {k: set() for k in CPPN.OUTPUTS_LIST}
    subset = {CPPN.Output.Weight, CPPN.Output.LEO}
    outputs = CPPN.outputs()
    for i in range(100):
        cppn(p0, p1, outputs, subset)
        for o in CPPN.OUTPUTS_LIST:
            values[o].add(outputs[o])

    for o in CPPN.OUTPUTS_LIST:
        assert len(values[o]) == 1


def test_outputs_equals(cppn_type, seed):
    cppn, _ = _make_cppn(cppn_type, seed)

    rng = Random(seed)

    def r():
        return rng.uniform(-1, 1)

    def p():
        return cppn_type.Point(*[r() for _ in range(cppn_type.DIMENSIONS)])

    def all_subsets(ss):
        return list(chain(*map(lambda x: combinations(ss, x),
                               range(0, len(ss)+1))))

    for _ in range(100):
        p0, p1 = p(), p()

        outputs_manual = []
        for o in CPPN.OUTPUTS_LIST:
            outputs_manual.append(cppn(p0, p1, o))

        outputs = CPPN.outputs()
        # Test all combinations
        outputs_subsets = []
        for subset in all_subsets(CPPN.OUTPUTS_LIST):
            cppn(p0, p1, outputs, set(subset))
            outputs_subsets.append((subset,
                                    [outputs[i] for i in range(len(outputs))]))

        cppn(p0, p1, outputs)
        outputs_all = [outputs[i] for i in range(len(outputs))]

        assert outputs_manual == outputs_all
        for subset, outputs_subset in outputs_subsets:
            for i in range(len(outputs_manual)):
                if CPPN.Output(i) in subset:
                    assert outputs_manual[i] == outputs_subset[i]


sample_sizes = [10, 50, 100]


@test_steps('generate_cppn', *[f"sample_at_{s}" for s in sample_sizes])
def test_multiscale_sampling(cppn_type, mutations, seed,
                             tmp_path_factory):  # pragma: no cover
    folder = tmp_path_factory.mktemp(numbered=False,
                                     basename=f"test_multiscale_sampling_"
                                              f"s{seed}_m{mutations}")

    cppn, genome = _make_cppn(cppn_type, seed, mutations)

    genome.to_dot(f"{folder}/cppn.png")
    print('Generated', f"{folder}/cppn.png")

    yield 'generate_cppn'

    d = cppn_type.Point.DIMENSION
    def _p(*args): return cppn_type.Point(*(list(args)+[0 for _ in range(len(args)-d)]))

    dp = _p()
    for size in sample_sizes:
        def to_substrate_coord(index):
            return 2 * index / (size - 1) - 1

        data = array.array('B', [0, 0, 0] * size * size * 2)

        outputs = CPPN.outputs()

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
