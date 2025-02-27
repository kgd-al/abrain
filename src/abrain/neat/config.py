import ast
import functools
import logging
from abc import ABC
from argparse import Action
from dataclasses import dataclass, fields
from functools import lru_cache
from pathlib import Path
from typing import get_args, get_origin, Union, Annotated, Optional, Tuple


@dataclass
class ConfigBase(ABC):

    @classmethod
    def __fields(cls):
        return [field for field in fields(cls) if get_origin(field.type) is Annotated]

    @staticmethod
    def _parse_tuple(_str, types):
        if _str.lower() == "none":
            return None
        else:
            return tuple(
                _t(_v) for _t, _v in zip(
                    types,
                    _str.replace("(", "").replace(")", "").split(",")
                )
            )

    @classmethod
    def populate_argparser(cls, parser):
        for field in cls.__fields():
            a_type = field.type.__args__[0]
            t_args = get_args(a_type)
            f_type, str_type = a_type, None
            default = field.default
            action = "store"

            if get_origin(a_type) is Union and type(None) in t_args:
                f_type = a_type = t_args[0]

            if a_type is bool:
                f_type = ast.literal_eval
                str_type = bool
            elif get_origin(a_type) is tuple:
                f_type = functools.partial(cls._parse_tuple, types=get_args(a_type))
                str_type = tuple

            if not str_type:
                str_type = f_type

            assert str_type, (
                f"Invalid user type {str_type} " f"(from {a_type=} {f_type=})"
            )

            help_kwargs = dict(default=default, type=str_type.__name__)
            arg_kwargs = dict()
            if str_type is bool:
                help_kwargs.update(const="True")
                arg_kwargs.update(const="True", nargs="?")
            help_msg = (
                f"{'.'.join(field.type.__metadata__)} ("
                + ", ".join(f"{k}: {v}" for k, v in help_kwargs.items())
                + ")"
            )
            parser.add_argument(
                f"--{field.name}".replace("_", "-"),
                action=action,
                dest=f"{field.name}",
                default=default,
                metavar="V",
                type=f_type,
                help=help_msg,
                **arg_kwargs,
            )

    @classmethod
    def from_argparse(cls, namespace):
        data = cls()
        for field in cls.__fields():
            f_name = f"{field.name}"
            attr = None
            if (
                    hasattr(namespace, f_name)
                    and (maybe_attr := getattr(namespace, f_name)) is not None
            ):
                attr = maybe_attr
            if attr is not None:
                setattr(data, field.name, attr)
        if post_init := getattr(data, "_post_init", None):  # pragma: no branch
            post_init(allow_unset=True)
        return data


@dataclass
class Config(ConfigBase):
    resume: Annotated[Path, "Resume evolution from provided checkpoint archive"] = None
    threads: Annotated[int, "Number of concurrent evaluations"] = None
    overwrite: Annotated[bool, "Do we allow running in an existing folder?"] = False
    symlink_last: Annotated[bool, "Should I add a symlink named *last* pointing to this run?"] = False

    data_root: Annotated[Optional[Union[Path, str]], "Where to store the generated data"] = None
    log_level: int = 1

    seed: Annotated[Optional[int], "RNG seed (set from time if none)"] = None

    population_size: Annotated[int, "Population size (duh)"] = 10
    generations: Annotated[int, "Number of generations (double duh)"] = 10
    tournament_size: Annotated[int, "How many individuals to compare for fitness"] = 4
    elitism: Annotated[int, "How many champions to keep unchanged each generation"] = 1

    species: Annotated[int, "Target number of species (for NEAT diversity)"] = 8
    species_size_inertia: Annotated[float, "How slow are species changing size based on relative fitness"] = .5
    initial_distance_threshold: Annotated[float, "Initial genetic distance to differentiate between species"] = .1
    distance_threshold_variation: Annotated[float, "How fast genetic distance cn vary, one way or another"] = 1.1

    external_mating: Annotated[float, "Probability of mating outside one's species"] = .01
    solitary_external_mating: Annotated[float, "Probability of mating outside a species of size 1, instead of muting"] = .5

    age_threshold: Annotated[int, "Number of generations before applying an age penalty to a species"] = 10
    protect_young: Annotated[bool, "Whether to protect new species by giving them a fitness boost"] = True

    logger: logging.Logger = None
