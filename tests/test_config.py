import shutil
from pathlib import Path

import pytest
from configobj import ConfigObj

from abrain.core.config import Config

default_config = None


def get_default_config():
    global default_config
    if default_config is None:
        default_config = {}
        # noinspection PyProtectedMember
        for section in Config._sections.values():
            for item in section:
                default_config[item] = str(getattr(Config, item))
    return default_config


@pytest.fixture
def restore_config_after():
    config = get_default_config()

    yield

    # noinspection PyProtectedMember
    for section in Config._sections.values():
        for item in section:
            # noinspection PyProtectedMember
            setattr(Config, item, Config._convert(item, config[item]))
    Config.show()


def test_config_exists():
    assert Config.functionSet
    assert Config.mutationRates
    assert len(Config._dict) > 0

    for k, v in Config._dict.items():
        print(f"\t{k}: {v}")


def test_config_read_write(tmp_path):
    Config.show()
    Config.write(tmp_path)
    Config.read(tmp_path)
    Config.show()


def test_round_loop(tmp_path):
    """
    Execute last in case configuration is screwed up for every one
    :param tmp_path:
    :return:
    """
    print(Config.mutationRates)
    print(str(Config.mutationRates))

    original = Config.write(None)
    path = Config.write(tmp_path)

    def header(msg):
        sw, _ = shutil.get_terminal_size((80, 20))
        print(f"{'_'*sw}\n== {msg} ==\n")

    def show(msg):
        header(msg)
        Config.show()
        print()

    show("Default file contents")

    header("On-disk config file")
    print(Path(path).read_text())
    print()

    header("Resetting values")
    for section in Config._sections.values():
        for item in section:
            attr = getattr(Config, item)
            default = type(attr)()
            print(f"\t{item}: {attr} of type {type(attr)} >> {default}")
            setattr(Config, item, default)
    print()

    show("Reset values")

    Config.read(tmp_path)

    show("Restored values")

    assert Config.write(None) == original


def test_failed_write(tmp_path):
    Config.write(tmp_path)
    with pytest.raises(IOError, match="exist"):
        Config.write(tmp_path)


def test_failed_read(tmp_path):
    with pytest.raises(IOError, match="not exist"):
        Config.read(Path(f"{tmp_path}/not-existing-sub-folder"))


def change_config_value(key, value, path):
    contents = Config.write(None)

    # print("--"*80)
    # print(f"Changing {key}: {value}")
    # print(contents)

    modified_contents = []
    for line in contents:
        tokens = line.split(" = ")
        if len(tokens) == 2:
            k, v = tokens
            if k == key:
                v = value
                # print(f"\t\t{k}:{type(k)}, {v}:{type(k)}")
            line = " = ".join([k, v])
            # print("\t", line)
        modified_contents.append(line)

    # print(modified_contents)

    with open(f"{path}/{Config.file}", 'wb') as outfile:
        ConfigObj(modified_contents).write(outfile)
        # print("Writing altered configuration to", outfile.name)
    Config.read(path)
    # Config.show()


@pytest.mark.usefixtures("restore_config_after")
@pytest.mark.parametrize(
    'key, value',
    [
        pytest.param("iterations", "15", id="int->int"),
        pytest.param("annWeightsRange", "1.5", id="float->float"),
        pytest.param("activationFunc", "id", id="str->str"),
        pytest.param("cppnWeightBounds", "Bounds(1,1,1,1,1)",
                     id="Bounds->Bounds"),
        pytest.param("functionSet", "[id,abs,ssgn]", id="Strings->Strings"),
        pytest.param("mutationRates",
                     str(get_default_config()["mutationRates"]),
                     id="MutationRates->MutationRates")
    ])
def test_correct_read_type(key, value, tmp_path):
    change_config_value(key, value, tmp_path)


@pytest.mark.usefixtures("restore_config_after")
@pytest.mark.parametrize(
    'key, value',
    [
        pytest.param("annWeightsRange", "error", id="str->float"),
        pytest.param("iterations", "4.2", id="float->int"),
        pytest.param("cppnWeightBounds", "foo", id="str->Bounds"),
        pytest.param("cppnWeightBounds", "Bounds(3, 1, -1, -3, .1)",
                     id="inv_bounds"),
        pytest.param("cppnWeightBounds", "Bounds(-3, -1, 1, 3, -.1)",
                     id="neg_bounds"),
        pytest.param("functionSet", "42", id="str->Strings"),
        pytest.param("mutationRates", "{}", id="empty_mr"),
        pytest.param("mutationRates", "[foo,bar]", id="bad_mr"),
    ])
def test_failed_read_type(key, value, tmp_path):
    with pytest.raises(TypeError):
        change_config_value(key, value, tmp_path)


@pytest.mark.usefixtures("restore_config_after")
@pytest.mark.parametrize(
    'key, value',
    [
        pytest.param("activationFunc", "sin", id="act"),
        pytest.param("functionSet", "[abs, id]", id="funcs"),
        pytest.param("outputFunctions", "[id,id,id]", id="outputs")
    ])
def test_correct_read_depends(key, value, tmp_path):
    change_config_value(key, value, tmp_path)


@pytest.mark.usefixtures("restore_config_after")
@pytest.mark.parametrize(
    'key, value',
    [
        pytest.param("activationFunc", "1", id="act"),
        pytest.param("functionSet", "[circle_quadrature]", id="funcs"),
        pytest.param("outputFunctions", "[id,id]", id="outputs"),
        pytest.param("outputFunctions", "[sa,la,mi]", id="outputs"),
        pytest.param("mutationRates", "{add_n: 0}", id="null_mr"),
        pytest.param("mutationRates", "{add_n: -1}", id="neg_mr"),
    ])
def test_failed_read_depends(key, value, tmp_path):
    with pytest.raises(ValueError):
        change_config_value(key, value, tmp_path)


def test_activation_function():
    assert Config.activationFunc == 'ssgn'
    print(Config.activationFunc)
