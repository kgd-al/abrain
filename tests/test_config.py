import json
import shutil
from pathlib import Path
from typing import Dict

import pytest

from abrain.core.config import Config

default_config = None


@pytest.fixture
def tmp_config_file(tmp_path):
    return tmp_path.joinpath("config.json")


def get_default_config():
    global default_config
    if default_config is None:
        default_config = Config.to_json()
    return default_config


@pytest.fixture
def restore_config_after():
    config = get_default_config()

    yield

    # noinspection PyProtectedMember
    Config.from_json(config)
    Config.show()


def test_config_exists():
    assert Config.functionSet
    assert Config.mutationRates
    assert len(Config._dict) > 0

    for k, v in Config._dict.items():
        print(f"\t{k}: {v}")


@pytest.mark.parametrize("key", ["cppnWeightBounds",
                                 "functionSet", "outputFunctions",
                                 "mutationRates"])
def test_json_conversions(key):
    value = getattr(Config, key)
    fake_json = value.toJson()
    roundtrip = type(value).fromJson(fake_json)

    print(value, fake_json, roundtrip)


def test_config_read_write(tmp_config_file):
    Config.show()
    Config.write(tmp_config_file)
    Config.read(tmp_config_file)
    Config.show()


def test_round_loop(tmp_config_file):
    """
    Execute last in case configuration is screwed up for every one
    :param tmp_config_file:
    :return:
    """
    print(Config.mutationRates)
    print(str(Config.mutationRates))

    original = Config.to_json()

    Config.write(tmp_config_file)

    def header(msg):
        sw, _ = shutil.get_terminal_size((80, 20))
        print(f"{'_'*sw}\n== {msg} ==\n")

    def show(msg):
        header(msg)
        Config.show()
        print()

    show("Default file contents")

    header("On-disk config file")
    print(tmp_config_file.read_text())
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

    Config.read(tmp_config_file)

    show("Restored values")

    assert Config.to_json() == original


def test_failed_write(tmp_config_file):
    Config.write(tmp_config_file)
    with pytest.raises(IOError, match="exist"):
        Config.write(tmp_config_file)


def test_failed_read(tmp_config_file):
    with pytest.raises(IOError, match="not exist"):
        Config.read(Path(f"{tmp_config_file}/not-existing-sub-folder"))


def nested_getter(key: str, j: Dict):  # pragma: no cover
    for section, items in Config._sections.items():
        for k in items:
            if k == key:
                return j[section][key]
    raise KeyError


def nested_setter(key: str, value, j: Dict):  # pragma: no cover
    for section, items in Config._sections.items():
        for k in items:
            if k == key:
                j[section][key] = value
                return
    raise KeyError


def change_config_value(key, value, path):
    contents = Config.to_json()

    # print("--"*80)
    # print(f"Changing {key}: {value}")
    # print(contents)

    nested_setter(key, value, contents)

    # print(contents)

    with open(path, 'w') as outfile:
        json.dump(contents, outfile)
        # print("Writing altered configuration to", outfile.name)
    Config.read(path)
    # Config.show()


@pytest.mark.usefixtures("restore_config_after")
@pytest.mark.parametrize(
    'key, value',
    [
        pytest.param("iterations", 15, id="int->int"),
        pytest.param("annWeightsRange", 1.5, id="float->float"),
        pytest.param("activationFunc", "id", id="str->str"),
        pytest.param("cppnWeightBounds", [1, 1, 1, 1, 1],
                     id="List->Bounds"),
        pytest.param("functionSet", ["id", "abs", "ssgn"], id="List->Strings"),
        pytest.param("mutationRates",
                     nested_getter("mutationRates", get_default_config()),
                     id="MutationRates->MutationRates")
    ])
def test_correct_read_type(key, value, tmp_config_file):
    change_config_value(key, value, tmp_config_file)


@pytest.mark.usefixtures("restore_config_after")
@pytest.mark.parametrize(
    'key, value',
    [
        pytest.param("annWeightsRange", "error", id="str->float"),
        pytest.param("iterations", 4.2, id="float->int"),
        pytest.param("cppnWeightBounds", "foo", id="str->Bounds"),
        pytest.param("functionSet", "42", id="str->Strings"),
        pytest.param("mutationRates", ["foo", "bar"], id="bad_mr"),
    ])
def test_failed_read_type(key, value, tmp_config_file):
    with pytest.raises(TypeError):
        change_config_value(key, value, tmp_config_file)


@pytest.mark.usefixtures("restore_config_after")
@pytest.mark.parametrize(
    'key, value',
    [
        pytest.param("activationFunc", "sin", id="act"),
        pytest.param("functionSet", ["abs", "id"], id="funcs"),
        pytest.param("outputFunctions", ["id", "id", "id"], id="outputs")
    ])
def test_correct_read_depends(key, value, tmp_config_file):
    change_config_value(key, value, tmp_config_file)


@pytest.mark.usefixtures("restore_config_after")
@pytest.mark.parametrize(
    'key, value',
    [
        pytest.param("activationFunc", "1", id="act"),
        pytest.param("functionSet", ["circle_quadrature"], id="funcs"),
        pytest.param("outputFunctions", ["id", "id"], id="outputs"),
        pytest.param("outputFunctions", ["sa", "la", "mi"], id="outputs"),
        pytest.param("cppnWeightBounds", [3, 1, -1, -3, .1],
                     id="inv_bounds"),
        pytest.param("cppnWeightBounds", [-3, -1, 1, 3, -.1],
                     id="neg_bounds"),
        pytest.param("mutationRates", {}, id="empty_mr"),
        pytest.param("mutationRates", {"add_n": 0}, id="null_mr"),
        pytest.param("mutationRates", {"add_n": -1}, id="neg_mr"),
    ])
def test_failed_read_depends(key, value, tmp_config_file):
    with pytest.raises(ValueError):
        change_config_value(key, value, tmp_config_file)


def test_activation_function():
    assert Config.activationFunc == 'ssgn'
    print(Config.activationFunc)
