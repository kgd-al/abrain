import pytest
from enum import IntFlag


class TestSize(IntFlag):
    SMALL = 2
    NORMAL = 8
    LARGE = 32


flags = {
    TestSize.SMALL: "--fast",
    TestSize.NORMAL: "--normal-scale",
    TestSize.LARGE: "--full-scale",
}


def pytest_addoption(parser):
    parser.addoption(flags[TestSize.SMALL], "--small-scale", action='store_const',
                     const=TestSize.SMALL, dest='size',
                     help='Run very small test suite '
                          '(single mutation case, 4 repeats...)')
    parser.addoption(flags[TestSize.NORMAL], action='store_const',
                     const=TestSize.NORMAL, default=TestSize.NORMAL, dest='size',
                     help='Run moderate test suite '
                          '(two mutation cases, 8 repeats...)')
    parser.addoption(flags[TestSize.LARGE], action='store_const',
                     const=TestSize.LARGE, dest='size',
                     help='Run large test suite '
                          '(4 mutation cases, 16 repeats...). '
                          'Warning: While it ensures good coverage, it should take '
                          'long')


best_ad_rate = .75


kgd_config = dict(
    seed={k: range(k) for k in TestSize},
    mutations={
        TestSize.SMALL: [10],
        TestSize.NORMAL: [100],
        TestSize.LARGE: [0, 1000]
    },
    ad_rate={
        TestSize.SMALL: [best_ad_rate],
        TestSize.NORMAL: [1],
        TestSize.LARGE: [2, .5]
    }
)


def values_for(key: str):
    values = []
    for v in kgd_config[key].values():
        values += v
    return sorted(set(values))


def scale_for(key: str, value):
    for ts in TestSize:  # pragma: no cover
        if value in kgd_config[key][ts]:
            return ts


def max_scale_for(params):
    scale = TestSize.SMALL
    for k, v in params.items():
        if k in kgd_config:
            scale = max(scale, scale_for(k, v))
    return scale


def pytest_generate_tests(metafunc):
    def can_parametrize(name):
        if name not in metafunc.fixturenames:
            return False
        existing = [
            m for m in metafunc.definition.iter_markers('parametrize')
            if name in m.args[0]
        ]
        return len(existing) == 0

    def maybe_parametrize(name, short_name):
        if can_parametrize(name):
            # print(f"adding {name} = {values_for(name)}")
            metafunc.parametrize(name, values_for(name), ids=lambda val: f"{short_name}_{val}")

    # print("Configuring", metafunc.function)
    maybe_parametrize("mutations", "m")
    maybe_parametrize("seed", "s")
    maybe_parametrize("ad_rate", "ar")


def pytest_collection_modifyitems(config, items):
    scale = config.getoption("size")
    slow_marks = {}
    for s_ in TestSize:
        if s_ <= scale:
            continue
        flag = flags[s_]
        slow_marks[s_] = \
            pytest.mark.skip(reason=f"Test scale ({s_.name}) is larger than "
                                    f"current target ({scale.name}). Use {flag} to run")

    for item in items:
        if hasattr(item, 'callspec'):
            that_scale = max_scale_for(item.callspec.params)
            if scale < that_scale:
                item.add_marker(slow_marks[that_scale])
            # else:
            #     print(item, item.callspec)
