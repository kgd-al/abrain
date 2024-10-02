import copy
import inspect
import logging
import math
import multiprocessing
import shutil
from dataclasses import dataclass
from fileinput import filename
from pathlib import Path
from random import Random
from typing import List, Callable, Optional

import graphviz
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

try:
    from matplotlib import pyplot as plt
    import pandas as pd
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from abrain.core.genome import Genome


logger = logging.getLogger(__name__)
logging.MAYBE_DEBUG = logging.INFO + 5
logging.addLevelName(logging.MAYBE_DEBUG, 'MAYBE_DEBUG')


def _log(msg, level=logging.MAYBE_DEBUG, *args, **kwargs):
    logger.log(level, msg, *args, **kwargs)


# Maybe there:
# - Elitism
# - Adaptive distance threshold (to get targeted number of species)
# - Stagnation
# > Use age as a fitness normalizer? young gets a boost, stagnant a huge debuff
# - Out of species crossover
# - genealogy

# Missing:
# - Built-in restart?


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
            f_dev += (f_avg - i.fitness)**2
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
class NEATConfig:
    population_size: int = 10
    tournament_size: int = 4
    elitism: int = 1

    seed: Optional[int] = None
    threads: Optional[int] = None

    # minimal_species_size = 2
    species_count: int = 4
    species_size_inertia = .5

    external_mating = .01
    solitary_external_mating = .5

    initial_distance_threshold: float = 1
    distance_threshold_variation: float = 1.1

    age_threshold = 10
    protect_young = True

    log_dir: Optional[Path | str] = None
    log_level: int = 1
    overwrite: bool = False


class NEATEvolver:
    def __init__(self, config: NEATConfig, evaluator: Callable,
                 genome_class=Genome, genome_data=None):
        nan = float("nan")

        self.config = config

        if config.log_dir is not None:
            if not isinstance(config.log_dir, Path):
                config.log_dir = Path(config.log_dir)
            if config.log_dir.exists() and config.overwrite:
                shutil.rmtree(config.log_dir)
            config.log_dir.mkdir(exist_ok=config.overwrite,
                                 parents=True)
            logger.info(f"Created output folder {config.log_dir}")

        self.species: List[Species] = []
        self.next_sid = 0

        self.generation = 0
        self.rng = Random(config.seed)

        self.evaluator = evaluator
        self.fitnesses = dict(max=nan, avg=nan, std=nan)

        self.individual = self._individual_class(genome_class, genome_data)

        self.distance_threshold = self.config.initial_distance_threshold
        self._distances = dict(species=nan, inter=nan, intra=nan)

        self.stat_fields = {
            "Gen": ("{:3d}", lambda: self.generation),
            "Sp": ("{:2d}", lambda: len(self.species)),
            "dis_t": ("{:5.3g}", lambda: self.distance_threshold),
            "d_spc": ("{:5.3g}", lambda: self._distances["species"]),
            "d_int": ("{:5.3g}", lambda: self._distances["intra"]),
            "d_ext": ("{:5.3g}", lambda: self._distances["inter"]),
            "F_max": ("{: 5.3g}", lambda: self.fitnesses["max"]),
            "F_avg": ("{: 5.3g}", lambda: self.fitnesses["avg"]),
            "F_dev": ("{: 5.2g}", lambda: self.fitnesses["dev"]),
        }
        self.files, self.file_names = {}, {}

        population = [self.individual.random()
                      for _ in range(self.config.population_size)]
        self._evaluate(population)
        self._speciate(population)
        for s in self.species:
            s.prev_size = len(s)

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
            print(" ".join(k for k in self.stat_fields.keys()))
        if self.config.log_dir is not None:
            def make_file(name):
                key = name.split(".")[0]
                path = self.file_names[key] = (
                    self.config.log_dir.joinpath(name))
                self.files[key] = f = open(path, "wt")
                return f

            file = make_file("stats.csv")
            print(",".join(k for k in self.stat_fields.keys()), file=file)

            make_file("species.csv")
            self._speciation_stats(init=True)

            make_file("genealogy.dat")

        self._global_stats()

    def _end(self):
        for f in self.files.values():
            f.close()

    def step(self):
        new_population = self._reproduce()
        self._evaluate(new_population)
        self._speciate(new_population)

        self.generation += 1
        self._global_stats()

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

    def _evaluate(self, population: List):
        if (t := self.config.threads) is None or t <= 1:
            for i in population:
                i.fitness = self.evaluator(i.genome)
        else:
            with multiprocessing.Pool(processes=t) as pool:
                results = pool.map(self.evaluator,
                                   [i.genome for i in population])
                for i, f in zip(population, results):
                    i.fitness = f
        self.fitnesses = _stats(population)

    def _speciate(self, population):
        distances = _Distances(self.distance_threshold,
                               self.individual.fn_distance)

        for s in self.species:
            s.reset()

        # if False:
        #     # First update current species
        #     extinct = []
        #     for s in self.species:
        #         s_distances = [(d[0], g) for g in population
        #                        if (d := distances(s.representative, g))[1]]
        #         # print("[kgd-debug]", s.id, s_distances)
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
            for rhs in range(lhs+1, len(self.species))
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

        tns = self.config.species_count
        if (ns := len(self.species)) != tns:
            base_factor = self.config.distance_threshold_variation
            factor = 1 / base_factor if ns < tns else base_factor

            # self.distance_threshold *= factor
            self.distance_threshold = max(
                self._distances["intra"],
                min(self.distance_threshold * factor,
                    self._distances["inter"]))

            # self.distance_threshold = factor * species_distance
            # self.distance_threshold = .5 * (self.distance_threshold + species_distance)
            #
            # sign = -1 if ns < tns else 1
            # delta = sign * .5 * sample_distance
            # print("d:", delta)
            # self.distance_threshold = max(.5 * self.distance_threshold,
            #                               min(self.distance_threshold + delta,
            #                                   1.5 * self.distance_threshold))

        # pprint.pprint(list(distances._data.values()))

    def _speciation_stats(self, init=False):
        if (f := self.files.get("species")) is None:
            return

        if init:
            print("Generation,Species,Size,F_max,F_avg", file=f)

        else:
            for s in self.species:
                print(self.generation, s.id, len(s),
                      s.f_stats["max"], s.f_stats["avg"],
                      sep=",", file=f)

    def _reproduce(self):
        fitnesses = [sum(ind.fitness for ind in s.population) / (len(s)**2)
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
            print(" ".join(fmt.format(getter())
                           for fmt, getter in self.stat_fields.values()))

        if (log_file := self.files.get("stats")) is not None:
            print(",".join(str(getter())
                           for _, getter in self.stat_fields.values()),
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
        return b * (a-age) / (age + a) + 1 if age <= a else (1-c) * a / age + c

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

    @staticmethod
    def _individual_class(_genome, _genome_data):
        def has_function(name, params):
            fn = getattr(_genome, name)
            if fn is None:
                raise ValueError(f"No function '{name}' for '{_genome}'")

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
            genome_data = _genome_data
            __next_id = 0

            fn_random = has_function("random", len(_genome_data))
            fn_mutate = has_function("mutate", 1+len(_genome_data))
            fn_crossover = has_function("crossover", 2+len(_genome_data))
            fn_distance = has_function("distance", 2)

            def __init__(self, genome, parents=None, key=None):
                assert key == Individual.__key
                self.genome = genome
                self.fitness = None

                if (gid := getattr(genome, "id")) is not None:
                    if callable(gid):
                        self.id = gid
                    else:
                        self.id = lambda _: gid
                else:
                    self._id = Individual.__next_id
                    self.id = lambda _: self._id
                    Individual.__next_id += 1

                if (pid := getattr(genome, "parents")) is not None:
                    if callable(pid):
                        self.parents = pid
                    else:
                        self.parents = lambda _: pid
                else:
                    self._parents = [p.id for p in parents]
                    self.parents = lambda _: self._parents
                assert type(self.parents) is type(self.__repr__)

            def __repr__(self):
                return f"Individual(fitness={self.fitness}, {self.genome})"

            @classmethod
            def random(cls):
                return Individual(
                    cls.fn_random(**Individual.genome_data), [],
                    Individual.__key
                )

            @classmethod
            def mating(cls, lhs: 'Individual', rhs: 'Individual'):
                child = cls.fn_crossover(lhs.genome, rhs.genome,
                                         **cls.genome_data)
                cls.fn_mutate(child, **cls.genome_data)
                return Individual(child, [lhs, rhs], Individual.__key)

            def mutated(self):
                child = copy.deepcopy(self.genome)
                self.__class__.fn_mutate(child, **self.genome_data)
                return Individual(child, [self], Individual.__key)

            @classmethod
            def distance(cls, lhs: 'Individual', rhs: 'Individual'):
                return cls.fn_distance(lhs.genome, rhs.genome)

        return Individual

    def generate_plots(self, ext="png", options: Optional[dict] = None):
        _Plotter.generate_plots(self, ext, options)


class _Plotter:
    @classmethod
    def generate_plots(cls, evolver: NEATEvolver, ext: str,
                       options: Optional[dict]):
        options = options or {}

        if not HAS_MATPLOTLIB:
            logger.warning("Matplotlib/pandas is not installed."
                           " Cannot generate plots")
            return False

        o_dir = evolver.config.log_dir
        if o_dir is None or not o_dir.exists():
            logger.warning(f"Cannot generate plots:"
                           f" output dir {o_dir} does not exist")
            return False

        logger.info(f"Generating plots under {o_dir}")

        df = pd.read_csv(evolver.file_names["stats"])

        if ext.lower() == "pdf":
            with PdfPages(o_dir.joinpath(f"stats.pdf")) as pdf:
                d_args = dict(target_species=evolver.config.species_count)
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
                              options=options)\
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
