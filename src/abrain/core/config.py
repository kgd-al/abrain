import json
import logging
import pprint
from pathlib import Path
from typing import Optional, Dict

# noinspection PyUnresolvedReferences
from .._cpp.config import (Config as CPPConfig,
                           OutputFunctions as _OutputFunctions,
                           MutationRates as _MutationRates,
                           )

logger = logging.getLogger(__name__)


class Config(CPPConfig):
    """
    Wrapper for the C++ configuration data

    Allows transparent access to the values used for CPPN structure,
    genome mutation (:func:`Genome.mutate`), and ANN/ES-HyperNEAT parameters
    """

    OutputFunctions = _OutputFunctions
    MutationRates = _MutationRates

    #: The set of all wrapped C++ values
    _dict = {
        k: v for k, v in CPPConfig.__dict__.items() if not k.startswith('_')
    }

    #: Private reference to the config sections
    _sections = CPPConfig.__dict__['_sections']

    @classmethod
    def to_json(cls) -> Dict:
        """
        Convert to a json-compliant Python dictionary
        """
        dct = {}
        for section, items in cls._sections.items():
            dct[section] = {}
            for k in items:
                attr = getattr(cls, k)
                if hasattr(attr, "toJson"):
                    attr = attr.toJson()
                dct[section][k] = attr

        return dct

    @classmethod
    def from_json(cls, j: Dict):
        """
        Restore values from a json-compliant Python dictionary

        :param j: the dictionary to parse values from
        """
        for section, items in cls._sections.items():
            for k in items:
                attr = j[section][k]
                this_attr = getattr(cls, k)
                if hasattr(this_attr, "fromJson"):
                    j_attr = attr
                    attr = type(this_attr).fromJson(j_attr)

                    if not attr.isValid():
                        raise ValueError(
                            f"{attr} is not a valid value for {k}")
                try:
                    setattr(cls, k, attr)
                except TypeError as e:
                    raise TypeError(f"Failed to convert {attr}:"
                                    f" {type(attr)} -> {type(this_attr)}\n{e}")

    @classmethod
    def write(cls, path: Optional[Path]):
        """
        Write the configuration to the specified file or stdout

        :param path: where to write or none to print to screen
        """

        if path is not None and path.exists():
            raise IOError(f"Will not overwrite existing "
                          f"file {path} with this one")

        json_cls = cls.to_json()

        if path is None:
            pprint.pprint(json_cls)
        else:
            with open(path, 'w') as f:
                json.dump(json_cls, f)
            assert path.exists()

    @classmethod
    def show(cls):
        """
        Write the configuration on standard output
        """

        cls.write(path=None)

    @classmethod
    def read(cls, path: Path):
        """
        Try to load data from provided path

        :param path: Filename
        """

        if not path.exists():
            raise IOError(f"Input path '{path}' does not exist")

        with open(path, 'r') as f:
            cls.from_json(json.load(f))

        try:
            # Delegate relationship tests to the C++
            Config.test_valid()
        except ValueError as e:
            raise ValueError("C++ layer exception when checking"
                             " the provided configuration") \
                from e

        return path
