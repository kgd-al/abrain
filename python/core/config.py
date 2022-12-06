import logging
from pathlib import Path
from typing import Optional

from configobj import ConfigObj

from _cpp.config import Config as CPPConfig
from _cpp.phenotype import CPPN

logger = logging.getLogger(__name__)


class Config(CPPConfig):
    name = "ESHN-Config"
    file = f"{name}.config"
    dict = {
        k: v for k, v in CPPConfig.__dict__.items() if not k.startswith('_')
    }
    _sections = CPPConfig.__dict__['_sections']

    @staticmethod
    def write(folder: Optional[Path]):
        """
        Write the configuration to a file under folder

        The specific name for this class is ESHN-Config.config

        :param folder: where to write
        """

        path = None
        if folder is not None:
            path = f"{folder}/{Config.file}"
            if Path(path).exists():
                raise IOError(f"Will not overwrite existing file {path} with this one")

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
                logger.debug(f"Fetching config[{Config.name}][{section}][{item}]: "
                             f"{config[Config.name][section][item]}")
                str_value = config[Config.name][section][item]
                new_value = Config._convert(item, str_value)
                setattr(Config, item, new_value)

        # Then check relationship assertions
        if not Config.known_function(Config.activationFunc):
            raise ValueError(f"Activation function {Config.activationFunc} is unknown")

        for func in Config.functionSet:
            if not Config.known_function(func):
                raise ValueError(f"CPPN internal function {Config.activationFunc} is unknown")

        for func in Config.outputFunctions:
            if not Config.known_function(func):
                raise ValueError(f"CPPN output function {Config.activationFunc} is unknown")

        if len(Config.outputFunctions) != CPPN.OUTPUTS:
            raise ValueError(f"Invalid number of output functions in {Config.outputFunctions}. "
                             f"CPPN requires {CPPN.OUTPUTS}")

        if len(Config.mutationRates) == 0:
            raise ValueError("Mutation rates cannot be empty")

        if sum(r for r in Config.mutationRates.values()) == 0:
            raise ValueError("Mutation rates require at least one positive value")

        if any(r < 0 for r in Config.mutationRates.values()):
            raise ValueError("Mutation rates cannot be negative")

        if True:
            b = Config.cppnWeightBounds
            if not b.min <= b.rndMin <= b.rndMax <= b.max \
                    or b.stddev <= 0:
                raise ValueError(f"Invalid CPPN Weight bounds: {b}")

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
            if new_value == c_type():
                raise TypeError(f"Failed to parse config item {item} with expected type of "
                                f"{c_type} and value of {str_value}")
        else:
            try:
                new_value = type(c_value)(str_value)
            except Exception as e:
                raise TypeError(e)

        return new_value
