import copy
import functools
import inspect
import json
import math
import multiprocessing
import platform
import pprint
import shutil
import signal
from dataclasses import dataclass
from io import TextIOWrapper
from pathlib import Path
from random import Random
from typing import List, Callable, Optional, Any, Type, Dict, Tuple

import jsonpickle
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

from abrain.neat.config import Config
from . import _logging

try:
    from matplotlib import pyplot as plt
    import pandas as pd

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    pd = None

from abrain.core.genome import Genome


# Maybe there:
# - Elitism
# - Adaptive distance threshold (to get targeted number of species)
# - Stagnation
# > Use age as a fitness normalizer? young gets a boost, stagnant a huge debuff
# - Out of species crossover
# - genealogy
# - Built-in restart?

# Missing:
# - Nothing??

logger = _logging.logger


def _log(msg, level=_logging.MAYBE_DEBUG, *args, **kwargs):
    logger.log(level, msg, *args, **kwargs)


def _stats(population):
    if len(population) > 0:
        f_max, f_avg = -math.inf, 0
        for i in population:
            f = i.fitness
            f_max = max(f, f_max)
            f_avg += f
        f_avg /= len(population)
        f_dev = 0.
        for i in population:
            f_dev += (f_avg - i.fitness) ** 2
        f_dev = math.sqrt(f_dev / len(population))
        return dict(max=f_max, avg=f_avg, dev=f_dev)
    else:
        return dict(max=-math.inf, avg=-math.inf, dev=0)


class Species:
    def __init__(self, sid, representative):
        self.id = sid

        self.age = 0
        self.last_improved = 0

        self.representative = representative
        self.population = [representative]
        self.prev_size = 0

        self.f_stats = _stats([])

    def reset(self):
        self.prev_size = len(self)
        self.population = []

    def finalize(self):
        # Compute species fitness, update age and stagnation
        prev_fitness = self.f_stats["max"]
        self.f_stats = _stats(self.population)
        self.age += 1
        if prev_fitness < self.f_stats["max"]:
            self.last_improved = self.age

        # Sort in descending order (elites first)
        self.population.sort(key=lambda x: -x.fitness)

        # Representative is current champion
        self.representative = self.population[0]

    def __repr__(self):
        return (f"Species({self.id}, {len(self.population)}):"
                f" {self.representative}")

    def __len__(self): return len(self.population)


class _Distances:
    def __init__(self, threshold, function):
        self._data = {}
        self._threshold = threshold
        self._function = function

    def __call__(self, lhs, rhs):
        lhs_id, rhs_id = id(lhs), id(rhs)
        d = self._data.get((lhs_id, rhs_id))
        if d is None:
            d = self._function(lhs.genome, rhs.genome)
            # assert 0 <= d <= 1
            self._data[(lhs_id, rhs_id)] = d
        hit = (d < self._threshold)
        return d, hit


@dataclass
class EvaluationResult:
    fitness: float = float("-inf")
    stats: dict = None

    def __iter__(self): return iter((self.fitness, self.stats))


def valid_fitness(fitness):
    return not math.isnan(fitness) and not math.isinf(fitness)

#
# @jsonpickle.handlers.register(multiprocessing.Pool().__class__, base=True)
# class ProcessingPoolHandler(jsonpickle.handlers.BaseHandler):
#     magics = ('py/object', "multiprocessing.pool.Pool")
#
#     def flatten(self, obj, data):
#         assert data[self.magics[0]] == self.magics[1]
#         data["threads"] = len(obj._pool)
#         return data
#
#     def restore(self, data):
#         assert data[self.magics[0]] == self.magics[1]
#         return multiprocessing.Pool(data["threads"])
#


@jsonpickle.handlers.register(multiprocessing.Pool().__class__, base=True)
class ProcessingPoolHandler(jsonpickle.handlers.BaseHandler):
    def flatten(self, obj, data):
        return data

    def restore(self, data):
        return None


@jsonpickle.handlers.register(TextIOWrapper, base=True)
class OpenFilesHandler(jsonpickle.handlers.BaseHandler):
    magics = ('py/object', '_io.TextIOWrapper')

    def flatten(self, obj: TextIOWrapper, data):
        data["name"] = obj.name
        data["mode"] = obj.mode
        data["encoding"] = obj.encoding
        return data

    def restore(self, data):
        return open(file=data["name"], mode=data["mode"],
                    encoding=data["encoding"])


class Evolver:
    @dataclass
    class Interface:
        """
            :param g_class: the class of the genome to wrap into an individual
            :param data: data passed along to every genome operator.
            :param random: genome operator for creating a new, random genome.
            :param mutate: genome operator for transforming a genome.
            :param crossover: genome operator for creating an offspring from two parents.
            :param distance: genome operator for computing the genetic distance between two individuals.
        """
        g_class: Type = Genome
        data: Optional[Dict[Any, Any]] = None
        random: Optional[Callable] = None
        mutate: Optional[Callable] = None
        crossover: Optional[Callable] = None
        distance: Optional[Callable] = None

    def __init__(
            self,
            config: Config,
            evaluator: Callable[[Any], EvaluationResult],
            genotype_interface: Interface,
            global_config: Optional[Any] = None,
            process_initializer: Optional[Callable] = None,
    ):
        """
        Creates a NEAT evolver.
        :param config: The NEAT-specific configuration
        :param evaluator: Callable taking a genome and returning its fitness and an
                          optional dictionary of relevant statistics.
        :param genotype_interface: Collection of types and callables to access
                                   the appropriate genotype-level operators
                                   (mutation, crossover, ...)
        :param global_config: User-specific data that should be stored alongside the rest of the data.
        """

        nan = float("nan")

        self.config = config
        self.global_config = global_config

        if config.data_root is None:
            config.data_root = _logging.get_next_tmp_data_root()

        if not isinstance(config.data_root, Path):
            config.data_root = Path(config.data_root)
        if config.data_root.exists() and config.overwrite:
            shutil.rmtree(config.data_root)
        config.data_root.mkdir(exist_ok=config.overwrite,
                               parents=True)

        self.config.logger = _logging.setup_logging(config.data_root)

        if config.threads is None:
            config.threads = multiprocessing.cpu_count()

        logger.info(f"Created output folder {config.data_root}")
        logger.info(f"Running on {platform.node()}")
        logger.info(f"Using {config.threads} threads")

        if config.symlink_last:
            run_symlink = config.data_root.parent.joinpath("last")
            if run_symlink.is_symlink() or run_symlink.exists():
                run_symlink.unlink()

            run_symlink.symlink_to(config.data_root.absolute(), target_is_directory=True)

        self.species: List[Species] = []
        self.next_sid = 0

        self._generation = 0
        self.rng = Random(config.seed)

        self.evaluator = evaluator
        self.__evaluate = functools.partial(self._evaluate_one,
                                            evaluator=self.evaluator,
                                            config=self.config)
        self.__initializer = process_initializer

        self.fitnesses = dict(max=nan, avg=nan, std=nan)

        self.individual = _individual_class(genotype_interface)

        self._distance_threshold = self.config.initial_distance_threshold
        self._distances = dict(species=nan, inter=nan, intra=nan)

        self.stat_fields = {
            key: (f"{key:^{width}s}", f"{{:{width}{flags}}}", getattr(Evolver, prop).fget)
            for key, width, flags, prop in [
                ("Gen", 3, "d", "generation"),
                ("Sp", 3, "d", "species_count"),
                ("dis_t", 6, ".3g", "distance_threshold"),
                ("d_spc", 6, ".3g", "distance_between_species"),
                ("d_int", 6, ".3g", "distance_intra_species"),
                ("d_ext", 6, ".3g", "distance_inter_species"),
                ("F_max", 8, ".3g", "fitness_max"),
                ("F_avg", 8, ".3g", "fitness_avg"),
                ("F_dev", 8, ".2g", "fitness_stddev"),
            ]
        }
        self.files, self.file_names = {}, {}

        self.__started = False

    def run(self, n):
        self._begin()

        for _ in range(n):
            self.step()

        self._end()

    def __enter__(self):
        self._begin()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._end()

    def _begin(self):
        if self.config.log_level >= 0:
            logger.log(_logging.EVO,
                       " ".join(k[0] for k in self.stat_fields.values()))
        if self.config.data_root is not None:
            def make_file(name):
                key = name.split(".")[0]
                path = self.file_names[key] = (
                    self.config.data_root.joinpath(name))
                self.files[key] = f = open(path, "wt")
                return f

            file = make_file("stats.csv")
            print(",".join(k for k in self.stat_fields.keys()), file=file)

            make_file("species.csv")
            self._speciation_stats(init=True)

            make_file("genealogy.dat")

        if (t := self.config.threads) is None or t <= 1:
            self._processes_pool = None
        else:
            # self._processes_pool = multiprocessing.Pool(t)
            context = multiprocessing.get_context("forkserver")
            self._processes_pool = multiprocessing.pool.Pool(
                processes=t,
                initializer=self.__initializer,
                context=context)

        if not (self.generation > 0 and self.__started):
            population = [self.individual.random()
                          for _ in range(self.config.population_size)]
            population = self._evaluate(population)
            self._speciate(population)
            for s in self.species:
                s.prev_size = len(s)

        self._global_stats()

        def interrupt_handler(signum, _):
            logger.warning(f"Interrupted by signal {signum}.")
            self._processes_pool.terminate()
            self._processes_pool.join()
            raise RuntimeError(f"Interrupted by signal {signum}")

        signal.signal(signal.SIGINT, interrupt_handler)
        signal.signal(signal.SIGTERM, interrupt_handler)

        self.__started = True

    def _end(self):
        for f in self.files.values():
            f.close()
        if self._processes_pool is not None:
            self._processes_pool.close()

    def step(self):
        if not self.__started:
            raise RuntimeError("Stepping when evolver has not been started."
                               " Only use step() with the guard form of the evolver")

        new_population = self._reproduce()
        new_population = self._evaluate(new_population)
        self._speciate(new_population)

        self._generation += 1
        self._global_stats()
        self.dump()

    def dump(self, path=None):
        def _fail_safe(*args):
            print(*args)
            return None

        with open(path or self.config.data_root.joinpath("evolution.json"), "wt") as f:
            dct = copy.copy(self.__dict__)
            del dct["individual"]
            dct["interface"] = self.individual.interface()
            j = jsonpickle.encode(dct, f,
                                  fail_safe=_fail_safe,
                                  make_refs=False,
                                  keys=True,
                                  warn=True)
            f.write(j)

    @classmethod
    def _load(cls, path, keep_data=False):
        with open(path, "rt") as f:
            data = f.read()
            decoded = jsonpickle.decode(data, keys=True)
            if keep_data:
                return decoded, data
            else:
                return decoded

    @classmethod
    def load_config(cls, path) -> Tuple[Config, Any]:
        """Ugly but bypasses a lot of boilerplate code.

        Returns the configuration stored in the evolution.json file, as well
        as the static genetic data
        """
        data = json.load(open(path, "rt"))
        return (jsonpickle.decode(str(data["config"])),
                jsonpickle.decode(str(data["interface"]["data"]),
                                  keys=True))

    @classmethod
    def restore(cls, path, evaluator: Callable[[Any], EvaluationResult]):
        _self = cls.__new__(cls)
        _self.__dict__, data = cls._load(path, keep_data=True)

        interface: Evolver.Interface = _self.__dict__.pop("interface")
        individual = _individual_class(interface)

        # Second pass to get the individuals properly decoded.
        # TODO: Find a better solution
        _self.__dict__ = jsonpickle.decode(data,
                                           classes=[individual],
                                           keys=True)
        _self.individual = individual
        _self.evaluator = evaluator
        _self.__evaluate = functools.partial(cls._evaluate_one,
                                             evaluator=_self.evaluator,
                                             config=_self.config)
        _self.stat_fields = {
            k: (header, fmt, prop.fget)
            for k, (header, fmt, prop) in _self.stat_fields.items()
        }

        pprint.pprint(_self.config)

        _self.config.logger = _logging.setup_logging(
            _self.config.data_root,
            f"Resuming run from generation {_self.generation} up to {_self.config.generations}.")

        return _self

    @property
    def generation(self): return self._generation
    
    @property
    def species_count(self): return len(self.species)
    
    @property
    def distance_threshold(self): return self._distance_threshold

    @property
    def distance_between_species(self): return self._distances["species"]

    @property
    def distance_intra_species(self): return self._distances["intra"]

    @property
    def distance_inter_species(self): return self._distances["inter"]

    @property
    def fitness_max(self): return self.fitnesses["max"]

    @property
    def fitness_avg(self): return self.fitnesses["avg"]

    @property
    def fitness_stddev(self): return self.fitnesses["dev"]

    @property
    def best_fitness(self):
        return self.champion.fitness

    @property
    def champion(self):
        return self.species[0].population[0]

    @property
    def population(self):
        for s in self.species:
            yield from s.population

    @staticmethod
    def _evaluate_one(genome, evaluator, config):
        try:
            return evaluator(genome)

        except Exception as e:
            f = config.data_root.joinpath("failures")
            f.mkdir(parents=True, exist_ok=True)
            f = f.joinpath(f"{genome.id()}.json")
            genome.to_file(f, dict(exception=str(e),
                                   fitness=EvaluationResult.fitness))
            logger.error(
                f"Evaluation failed: {e.__class__.__name__}({e})."
                f" Guilty genotype written to {f}.")
            logger.exception("Stack trace:")

            return EvaluationResult()

    def _evaluate(self, population: List) -> List:
        if self._processes_pool is None:
            results = [self.__evaluate(i.genome) for i in population]
        else:
            results = self._processes_pool.map(self.__evaluate, [i.genome for i in population])
        for i, f in zip(population, results):
            i.fitness, i.stats = f

        # Filter out invalid fitnesses
        population = [_i for _i in population if valid_fitness(_i.fitness)]
        if (filtered := self.config.population_size - len(population)) > 0:
            if len(population) == 0:
                logger.error(f"Filtering out invalid individuals results"
                             " in an empty population.\n"
                             f"Fitness values were:\n"
                             f"{pprint.pformat(results)}")
                raise RuntimeError("Empty population.")
            else:
                logger.warning(f"Filtered out {filtered} invalid individuals.")

        self.fitnesses = _stats(population)

        return population

    def _speciate(self, population):
        distances = _Distances(self._distance_threshold,
                               self.individual.fn_distance)

        for s in self.species:
            s.reset()

        # if False:
        #     # First update current species
        #     extinct = []
        #     for s in self.species:
        #         s_distances = [(d[0], g) for g in population
        #                        if (d := distances(s.representative, g))[1]]
        #         # logger.debug("[kgd-debug]", s.id, s_distances)
        #
        #         if len(s_distances) == 0:
        #             if self.config.log_level >= 2:
        #                 _log(f"Species {s.id} died down (unrepresented)")
        #             extinct.append(s)
        #         else:
        #             _d, _g = min(s_distances, key=lambda x: x[0])
        #             s.representative = _g
        #             s.population = [_g]
        #             population.remove(s.representative)
        #
        #     for s in extinct:
        #         self.species.remove(s)

        # Assign population
        for g in population:
            g_distances = [(d[0], s) for s in self.species
                           if (d := distances(s.representative, g))[1]]
            if self.config.log_level > 10:
                _log(f">>> distances: {g_distances}")

            if len(g_distances) == 0:
                self.species.append(Species(self.next_sid, g))
                self.next_sid += 1
                if self.config.log_level > 2:
                    _log(f"New species: {self.species[-1]}")
            else:
                _, species = min(g_distances, key=lambda x: x[0])
                if self.config.log_level > 10:
                    _log(f">> best species: {species}")
                species.population.append(g)

        extinct = []
        for s in self.species:
            if len(s) > 0:
                s.finalize()
            else:
                extinct.append(s)

        # Clear out extinct ones
        for s in extinct:
            if self.config.log_level > 2:
                _log(f"Species {s.id} died down (unpopulated)")
            self.species.remove(s)

        self.species.sort(key=lambda x: -x.f_stats["max"])

        self._speciation_stats()

        self._distances["species"] = sum(
            distances(self.species[lhs].representative,
                      self.species[rhs].representative)[0]
            for lhs in range(len(self.species))
            for rhs in range(lhs + 1, len(self.species))
        ) / len(self.species)

        sample_size = 10
        self._distances["intra"] = sum(
            (0
             if len(sample) == 1 else
             distances(*self.rng.sample(sample, 2))[0])
            for _ in range(sample_size)
            if (sample := self.rng.choice(self.species).population)
        ) / sample_size

        self._distances["inter"] = sum(
            distances(
                self.rng.choice(sample[0].population),
                self.rng.choice(sample[1].population),
            )[0]
            for _ in range(sample_size)
            if (sample := self.rng.sample(self.species, 2))
        ) if len(self.species) > 1 else float("nan")

        tns = self.config.species
        if (ns := len(self.species)) != tns:
            base_factor = self.config.distance_threshold_variation
            factor = 1 / base_factor if ns < tns else base_factor

            # self._distance_threshold *= factor
            self._distance_threshold = max(
                self._distances["intra"],
                min(self._distance_threshold * factor,
                    self._distances["inter"]))

            # self._distance_threshold = factor * species_distance
            # self._distance_threshold = .5 * (self._distance_threshold + species_distance)
            #
            # sign = -1 if ns < tns else 1
            # delta = sign * .5 * sample_distance
            # logger.debug("d:", delta)
            # self._distance_threshold = max(.5 * self._distance_threshold,
            #                               min(self._distance_threshold + delta,
            #                                   1.5 * self._distance_threshold))

        # pprint.pprint(list(distances._data.values()))

    def _speciation_stats(self, init=False):
        if (f := self.files.get("species")) is None:
            return

        if init:
            print("Generation,Species,Size,F_max,F_avg", file=f)

        else:
            for s in self.species:
                print(self._generation, s.id, len(s),
                      s.f_stats["max"], s.f_stats["avg"],
                      sep=",", file=f)

    def _reproduce(self):
        fitnesses = [sum(ind.fitness for ind in s.population) / (len(s) ** 2)
                     for s in self.species]
        f_min, f_max = min(fitnesses), max(fitnesses)
        f_range = (f_max - f_min) or 1

        normalized_fitnesses = [
            (f - f_min) / f_range for f in fitnesses
        ]
        sum_fitnesses = sum(normalized_fitnesses)
        if sum_fitnesses == 0:
            normalized_fitnesses = [1 for _ in normalized_fitnesses]
            sum_fitnesses = sum(normalized_fitnesses)
        sum_fitnesses = sum_fitnesses

        ssi = self.config.species_size_inertia
        pop_size = self.config.population_size

        # Compute weighted spawns
        spawns = []
        for f, s in zip(normalized_fitnesses, self.species):
            spawn = pop_size * f / sum_fitnesses
            if s.prev_size > 0:
                spawn = ssi * s.prev_size + (1 - ssi) * spawn

            age_debt = s.age - s.last_improved
            age_bias = self._age_gain(age_debt)
            spawn = int(round(age_bias * spawn))

            # If elitism is on, keep that many from the best species
            if ((len(spawns) == 0 and self.config.elitism > 0)
                    # Also protect "young" species from premature extinction
                    or (self.config.protect_young
                        and age_debt - self.config.age_threshold <= 0)):
                spawn = min(spawn, self.config.elitism)

            spawns.append(spawn)

        if self.config.log_level >= 10:
            _log("Species:" + ''.join(f'\n> {s}' for s in self.species))
        if self.config.log_level >= 5:
            _log(f"Spawn rates (raw): {spawns}")

        # Normalize
        total_spawn = sum(spawns)
        spawns = [int(pop_size * spawn / total_spawn) for spawn in spawns]
        spawns[0] += (pop_size - sum(spawns))  # Give remainder to best species
        assert sum(spawns) == pop_size

        if self.config.log_level >= 5:
            _log(f"     (normalized): {spawns}")

        # Clear out extinct
        extinct = []
        for spawn, s in zip(spawns, self.species):
            if spawn < 1:
                if self.config.log_level >= 2:
                    _log(f"Species {s.id} died down (no offspring)")
                extinct.append(s)
        for s in extinct:
            self.species.remove(s)
        spawns = [spawn for spawn in spawns if spawn > 0]

        # print({s.id: spawn for s, spawn in zip(self.species, spawns)})

        # Spawn
        population = []
        for spawn, s in zip(spawns, self.species):
            i = 0
            elites = min(self.config.elitism, spawn, len(s))
            while i < elites:
                population.append(s.population[i])
                i += 1

            while i < spawn:
                if len(s) > 1:
                    parents = [self._tournament(s.population)]
                    if (len(self.species) > 1 and
                            self.rng.random() < self.config.external_mating):
                        out_species = self._tournament(
                            self.species,
                            exclude=s,
                            key=lambda x: x.f_stats["max"])
                        parents.append(out_species.population[0])

                    else:
                        parents.append(
                            self._tournament(s.population, *parents))
                    assert len(parents) == 2 and parents[0] != parents[1]

                else:
                    parents = [s.population[0]]
                    if (len(self.species) > 1 and
                            self.rng.random() < self.config.solitary_external_mating):
                        out_species = self._tournament(
                            self.species,
                            exclude=s,
                            key=lambda x: x.f_stats["max"])
                        parents.append(out_species.population[0])

                if len(parents) == 1:
                    child = parents[0].mutated()
                else:
                    child = self.individual.mating(*parents)

                population.append(child)
                i += 1

        return population

    def _global_stats(self):
        if self.config.log_level >= 0:
            logger.log(_logging.EVO,
                       " ".join(fmt.format(getter(self))
                                for _, fmt, getter in self.stat_fields.values()))

        if (log_file := self.files.get("stats")) is not None:
            print(",".join(str(getter(self))
                           for _, _, getter in self.stat_fields.values()),
                  file=log_file)

        if (log_file := self.files.get("genealogy")) is not None:
            for i in self.population:
                print(i.id(), i.fitness, *i.parents(), file=log_file)

    @staticmethod
    def _age_gain(age):
        # return 1
        a = 10  # Cutoff
        b = 4  # Maximal gain
        c = .5  # Minimal loss
        # return 1 if age <= a else (1-c) * a / age + c
        return b * (a - age) / (age + a) + 1 if age <= a else (1 - c) * a / age + c

    def _tournament(self, population: List,
                    exclude=None,
                    key=lambda x: x.fitness):
        if exclude is not None:
            population = [element for element in population
                          if element != exclude]
        return max(self.rng.sample(population,
                                   min(len(population),
                                       self.config.tournament_size)),
                   key=key)

    def generate_plots(self, ext="png", options: Optional[dict] = None):
        _Plotter.generate_plots(self, ext, options)


def _individual_class(interface: Evolver.Interface):
    _genome = interface.g_class
    _data = interface.data or {}
    _random = interface.random
    _mutate = interface.mutate
    _crossover = interface.crossover
    _distance = interface.distance

    def has_function(name, params):
        fn = getattr(_genome, name, None)
        if fn is None:
            raise ValueError(f"No function '{name}' for '{_genome.__qualname__}'")

        if not callable(fn):
            raise ValueError(f"'{name}' is not a callable function in"
                             f" '{_genome}'")

        sig = inspect.signature(fn)
        _params = sum(1 for p in sig.parameters.values()
                      if p.default is p.empty)
        if _params < params:
            raise ValueError(
                f"'{name}' has unexpected signature: '{sig}'.\n"
                f"Will provide {params} parameters, function expects"
                f" {_params}")

        return fn

    class Individual:
        __key = object()
        genome_class = _genome
        genome_data = _data
        __next_id = 0

        fn_random = _random or has_function("random", len(_data))
        fn_mutate = _mutate or has_function("mutate", 1 + len(_data))
        fn_crossover = _crossover or has_function("crossover", 2 + len(_data))
        fn_distance = _distance or has_function("distance", 2)

        @classmethod
        def interface(cls):
            return Evolver.Interface(
                g_class=Individual.genome_class,
                data=Individual.genome_data,
                random=Individual.fn_random,
                mutate=Individual.fn_mutate,
                crossover=Individual.fn_crossover,
                distance=Individual.fn_distance
            )

        def __init__(self, genome, gid=None, parents=None, key=None):
            assert key == Individual.__key
            self.genome = genome
            self.fitness, self.stats = None, None

            if (gid := getattr(genome, "id", None)) is not None:
                if callable(gid):
                    self.id = gid
                else:
                    self.id = lambda _: gid
            else:
                self._id = gid or Individual.__next_id
                self.id = lambda: self._id
                Individual.__next_id += 1

            if (pid := getattr(genome, "parents", None)) is not None:
                if callable(pid):
                    self.parents = pid
                else:
                    self.parents = lambda _: pid
            else:
                self._parents = [p if isinstance(p, int) else p.id()
                                 for p in parents]
                self.parents = lambda: self._parents

        def __getstate__(self):
            return dict(
                genome=self.genome,
                fitness=self.fitness, stats=self.stats,
                gid=getattr(self, "id")(),
                parents=getattr(self, "parents")()
            )

        def __setstate__(self, state):
            fitness, stats = state.pop("fitness"), state.pop("stats")
            self.__init__(**state, key=Individual.__key)
            self.fitness, self.stats = fitness, stats

        def __repr__(self):
            return f"Individual(fitness={self.fitness}, {self.genome})"

        @classmethod
        def random(cls):
            return Individual(
                cls.fn_random(**Individual.genome_data), parents=[],
                key=Individual.__key
            )

        @classmethod
        def mating(cls, lhs: 'Individual', rhs: 'Individual'):
            child = cls.fn_crossover(lhs.genome, rhs.genome,
                                     **cls.genome_data)
            cls.fn_mutate(child, **cls.genome_data)
            return Individual(child, parents=[lhs, rhs], key=Individual.__key)

        def mutated(self):
            child = copy.deepcopy(self.genome)
            self.__class__.fn_mutate(child, **self.genome_data)
            return Individual(child, parents=[self], key=Individual.__key)

        @classmethod
        def distance(cls, lhs: 'Individual', rhs: 'Individual'):
            return cls.fn_distance(lhs.genome, rhs.genome)

    return Individual


class _Plotter:
    @classmethod
    def generate_plots(cls, evolver: Evolver, ext: str,
                       options: Optional[dict]):
        options = options or {}

        if not HAS_MATPLOTLIB:
            logger.warning("Matplotlib/pandas is not installed."
                           " Cannot generate plots")
            return False

        o_dir = evolver.config.data_root
        if o_dir is None or not o_dir.exists():
            logger.warning(f"Cannot generate plots:"
                           f" output dir {o_dir} does not exist")
            return False

        logger.info(f"Generating plots under {o_dir}")

        df = pd.read_csv(evolver.file_names["stats"])

        if ext.lower() == "pdf":
            with PdfPages(o_dir.joinpath(f"stats.pdf")) as pdf:
                d_args = dict(target_species=evolver.config.species)
                for fn, args in [(cls.fitness, {}),
                                 (cls.distances, d_args)]:
                    fn(df, **args, options=options)
                    pdf.savefig()
                    plt.close()
        else:
            cls.fitness(df, options).savefig(o_dir.joinpath(f"fitness.{ext}"),
                                             bbox_inches="tight")

        # Species over generations (with fitness)
        cls.species_histogram(pd.read_csv(evolver.file_names["species"]),
                              options=options) \
            .savefig(o_dir.joinpath(f"species.{ext}"), bbox_inches="tight")

        return True

    # Simple fitness over generation
    @staticmethod
    def fitness(df: pd.DataFrame, options: dict):
        fig, ax = plt.subplots()
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
        ax.fill_between(df.Gen, df.F_avg - df.F_dev, df.F_avg + df.F_dev,
                        alpha=.1, label="Dev")
        ax.plot(df.Gen, df.F_avg, label="Avg")
        ax.plot(df.Gen, df.F_max, label="Max")
        if (of := options.get("optimal_fitness")) is not None:
            ax.axhline(of, linestyle="--", color='gray',
                       label="Optimal Fitness")
        ax.legend()
        return fig

    @staticmethod
    def distances(df: pd.DataFrame, target_species: int, options: dict):
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.grid()
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Distance")
        for key, label in [("d_spc", "Species"),
                           ("dis_t", "Threshold"),
                           ("d_int", "Internal"),
                           ("d_ext", "External")]:
            ax1.plot(df.Gen, df[key], label=label)
        ax1.legend(title="Distances")
        ax2.grid()
        ax2.set_ylabel("Species")
        ax2.plot(df.Gen, df.Sp, label="Species", linestyle="--")
        ax2.axhline(y=target_species, linestyle="--", color='black',
                    label="Target")
        ax2.legend()
        fig.tight_layout()
        # "Sp": ("{:2d}", lambda: len(self.species)),

    @staticmethod
    def species_histogram(df: pd.DataFrame, options: dict):
        length = df.Generation.max() + 1
        gb = df.groupby("Species")
        ys, fs = [], []
        for x in gb.groups:
            g = gb.get_group(x)
            ys.append(ys[-1][:] if len(ys) > 0 else [0 for _ in range(length + 1)])
            fs.append([float("nan") for _ in range(length)])
            for a in g[["Generation", "Size", "F_max"]].itertuples():
                gen = a.Generation
                ys[-1][gen] += a.Size
                fs[-1][gen] = a.F_max
            ys[-1][length] = ys[-1][length - 1]

        fig, ax = plt.subplots()

        patches, colors = [], []
        for s in range(len(ys)):
            curr = ys[s]
            for gen in range(0, length):
                def prev(_g): return ys[s - 1][_g] if s > 0 else 0

                coords = [
                    (gen, curr[gen]),
                    (gen, prev(gen)),
                    (gen + 1, prev(gen + 1)),
                    (gen + 1, curr[gen + 1])
                ]
                polygon = Polygon(coords)
                patches.append(polygon)
                colors.append(fs[s][gen])

        p = PatchCollection(patches, alpha=1)
        p.set_cmap("magma")
        p.set_clim(vmin=None, vmax=options.get("optimal_fitness"))
        p.set_array(colors)
        ax.add_collection(p)

        for i in range(len(ys)):
            ax.plot(range(length + 1), ys[i], label=None, color="black")

        ax.set_xlabel("Generation")
        ax.set_ylabel("Population")
        ax.autoscale(enable=True, tight=True)

        cbar = fig.colorbar(p, ax=ax)
        cbar.ax.set_ylabel("Fitness (max)")

        return fig
