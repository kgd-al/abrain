import logging
from pathlib import Path
from typing import Optional

from configobj import ConfigObj

from .._cpp.config import Config as CPPConfig
from .._cpp.phenotype import CPPN

logger = logging.getLogger(__name__)


class Config(CPPConfig):
    """
    Wrapper for the C++ configuration data

    Allows transparent access to the values used for CPPN structure,
    genome mutation (:func:`Genome.mutate`), and ANN/ES-HyperNEAT parameters
    """

    #: The name of this config file
    name = "ESHN-Config"

    #: The filename to write to / read from
    file = f"{name}.config"

    #: The set of all wrapped C++ values
    _dict = {
        k: v for k, v in CPPConfig.__dict__.items() if not k.startswith('_')
    }

    #: Private reference to the config sections
    _sections = CPPConfig.__dict__['_sections']

    @staticmethod
    def write(folder: Optional[Path]):
        """
        Write the configuration to a file under folder

        The specific name for this class is ESHN-Config.config

        Bullet points:
          - one
          - two

        :param folder: where to write
        :return: the path where this file was actually written to
        """

        path = None
        if folder is not None:
            path = f"{folder}/{Config.file}"
            if Path(path).exists():
                raise IOError(f"Will not overwrite existing "
                              f"file {path} with this one")

        config = ConfigObj(path)

        config[Config.name] = {}
        for section, items in Config._sections.items():
            config[Config.name][section] = {}
            for k in items:
                config[Config.name][section][k] = getattr(Config, k)

        if path is None:
            return config.write()
        else:
            config.write()
            assert Path(path).exists()
            return path

    @staticmethod
    def show():
        """
        Write the configuration on standard output
        """

        for section, items in Config._sections.items():
            print(f"\n[{section}]")
            for k in items:
                print(f"{k}: {getattr(Config, k)}")

    @staticmethod
    def read(folder: Path):
        """
        Try to load data from provided folder

        :param folder: Containing folder
        :return: The actual path used for reading
        """

        path = f"{folder}/{Config.file}"
        if not Path(path).exists():
            raise IOError(f"Input path '{path}' does not exist")

        config = ConfigObj(path, list_values=False)

        # First just read and convert (raise on incompatible types)
        for section, items in Config._sections.items():
            for item in items:
                logger.debug(f"Fetching config[{Config.name}][{section}]"
                             f"[{item}]: {config[Config.name][section][item]}")
                str_value = config[Config.name][section][item]
                new_value = Config._convert(item, str_value)
                setattr(Config, item, new_value)

        # Then check relationship assertions
        if not Config.known_function(Config.activationFunc):
            raise ValueError(f"Activation function {Config.activationFunc}"
                             f" is unknown")

        for func in Config.functionSet:
            if not Config.known_function(func):
                raise ValueError(f"CPPN internal function {func} is unknown")

        for func in Config.outputFunctions:
            if not Config.known_function(func):
                raise ValueError(f"CPPN output function {func} is unknown")

        if len(Config.outputFunctions) != CPPN.OUTPUTS:
            raise ValueError(f"Invalid number of output functions in"
                             f" {Config.outputFunctions}. CPPN requires"
                             f" {CPPN.OUTPUTS}")

        if sum(r for r in Config.mutationRates.values()) == 0:
            raise ValueError("Mutation rates require at least one positive"
                             " value")

        if any(r < 0 for r in Config.mutationRates.values()):
            raise ValueError("Mutation rates cannot be negative")

        # Let us assume I am careful enough for the other values

        return path

    @staticmethod
    def _convert(item, str_value):
        c_value = getattr(Config, item)
        c_type = type(c_value)

        logger.debug(f"\tConverting {str_value}:{type(str_value)} to {c_type}")
        if hasattr(c_value, 'ccpParseString'):
            new_value = c_type.ccpParseString(str_value)
            print(str_value, "->", new_value, c_type(), new_value == c_type())
            if not c_type.isValid(new_value):
                raise TypeError(f"Failed to parse config item {item} with"
                                f" expected type of {c_type} and value of"
                                f" {str_value}")
        else:
            try:
                new_value = type(c_value)(str_value)
            except Exception as e:
                raise TypeError(e)

        return new_value
