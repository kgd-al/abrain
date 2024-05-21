"""
Reasonable attempt at making abrain play nice with qdpy.
Provided as inspiration without any guarantee of functionality
"""


import itertools
import json
import logging
import math
import multiprocessing
import pickle
import random
import shutil
import warnings
from pathlib import Path

import PIL
import matplotlib
import numpy as np
from qdpy import algorithms, tools, containers, plots
from qdpy.algorithms import Evolution, partial
from qdpy.base import ParallelismManager
from qdpy.containers import Grid, Container
from qdpy.phenotype import IndividualLike, Fitness, Features
from rich.progress import track

from abrain import Point3D as Point, ANN3D as ANN
from abrain.core.ann import plotly_render
from abrain.core.config import Config
from abrain.core.genome import Genome, GIDManager
from examples.common import example_path

matplotlib.pyplot.set_loglevel(level='warning')

# BUDGET = 50
BUDGET = 50_000
EVOLVE = True
PROCESS = True
RUN_FOLDER = example_path("tmp/qdpy-evolution")


class Individual(IndividualLike):
    INPUTS = [Point(x, -1, z) for x, z
              in itertools.product([-1, -.5, 0, .5, 1],
                                   [-1, -.6, -.2, .2, .6, 1])]
    OUTPUTS = [Point(r * math.cos(a),
                     1,
                     r * math.sin(a)) for r, a in
               [(0, 0)] + [(1, 2*i*math.pi/5) for i in range(5)]]

    def __init__(self, genome: Genome, gen=0):
        self.genome = genome
        self._fitness = None
        self._features = None
        self.stats = None
        self.gen = gen
        self.name = str(genome.id())
        self.evaluate()

    def __repr__(self):
        return (f"Ind({self.gen}:{self.genome.id()},"
                f" {self._fitness} {self._features}")

    def evaluate(self):
        if self._fitness is None:
            ann = ANN.build(self.INPUTS, self.OUTPUTS,
                            self.genome)
            self._fitness = ann.stats().depth
            self._features = (
                ann.stats().density,
                ann.stats().utility
            )
            self.stats = ann.stats().dict()

    @staticmethod
    def ranges():
        return dict(
            fitness_domain=[(0, Config.iterations)],
            features_domain=[(0, 1),
                             (0, 1)]
        )

    def signed_features(self):
        return -self._features[0], self._features[1]

    @staticmethod
    def grid_shape():
        return 32, 32

    @property
    def fitness(self):
        return Fitness([self._fitness], [1])

    @property
    def features(self):
        return Features(self._features)


class Algorithm(Evolution):
    def __init__(self, container: Container,
                 seed, tournament, initial_mutations,
                 run_folder, budget, batch_size,
                 **kwargs):
        self.genome_data = Genome.Data.create_for_eshn_cppn(
            dimension=3, seed=seed
        )
        random.seed(seed)
        np.random.seed(seed % (2**32-1))

        self.id_manager = GIDManager()
        if PROCESS:
            self.genealogy = dict()
            self.population = dict()

        def select(grid):
            # return self.rng.choice(grid)
            k = min(len(grid), tournament)
            candidates = self.genome_data.rng.sample(grid.items, k)
            candidate_cells = [grid.index_grid(c.features) for c in candidates]
            cell = self.genome_data.rng.choice(candidate_cells)
            selection = candidates[candidate_cells.index(cell)]
            return selection

        def init(_):
            genome = Genome.random(self.genome_data)
            for _ in range(initial_mutations):
                genome.mutate(self.genome_data)
            ind = Individual(genome)
            if PROCESS:
                self.population[genome.id()] = ind
            return ind

        def vary(parent: Individual):
            child = Individual(
                parent.genome.mutated(self.genome_data),
                parent.gen + 1)

            if PROCESS:
                self.genealogy[child.genome.id()] = parent.genome.id()
                self.population[child.genome.id()] = child

            return child

        sel_or_init = partial(tools.sel_or_init,
                              init_fn=init, sel_fn=select, sel_pb=1)

        run_folder = Path(run_folder)
        if run_folder.exists():
            shutil.rmtree(run_folder, ignore_errors=True)
            logging.warning(f"Purging contents of {run_folder}, as requested")

        run_folder.mkdir(parents=True, exist_ok=False)

        self._latest_champion = None
        self._snapshots = run_folder.joinpath("snapshots")
        self._snapshots.mkdir(exist_ok=False)

        _bd = math.ceil(math.log10(budget))
        self._champion_fmt = f"better-{{:0{_bd}d}}-{{}}-{{:g}}.json"

        Evolution.__init__(self, container=container, name="Deeper",
                           budget=budget, batch_size=batch_size,
                           select_or_initialise=sel_or_init, vary=vary,
                           optimisation_task="maximisation",
                           **kwargs)

    def tell(self, individual: Individual, *args, **kwargs) -> bool:
        added = super().tell(individual, *args, **kwargs)

        if added:
            new_champion = False
            if self._latest_champion is None:
                self._latest_champion = (0, self.nb_evaluations,
                                         individual.fitness)
                new_champion = True

            else:
                timestamp, n, fitness = self._latest_champion
                if individual.fitness.dominates(fitness):
                    self._latest_champion = (self.nb_evaluations, n + 1,
                                             individual.fitness)
                    new_champion = True

            if new_champion:
                _t, _n, _f = self._latest_champion
                file = self._snapshots.joinpath(
                    self._champion_fmt.format(_t, _n, _f.values[0]))
                with open(file, "w") as f:
                    json.dump(self.to_json(individual), f)

        return added

    @classmethod
    def to_json(cls, i: Individual):
        return {
            "id": i.genome.id(), "parents": i.genome.parents(),
            "fitness": i._fitness,
            "descriptors": i._features,
            "stats": i.stats,
            "genome": i.genome.to_json()
        }


def eval_fn(ind):
    ind.evaluate()
    return ind


def evolve(is_test):
    Config.iterations = 32
    Config.maxDepth = 2

    if is_test:
        global BUDGET
        BUDGET = 250

    grid = containers.Grid(
        shape=Individual.grid_shape(),
        max_items_per_bin=1,
        **Individual.ranges())

    algo = Algorithm(
        grid, seed=0, tournament=4, initial_mutations=10,
        run_folder=RUN_FOLDER,
        budget=BUDGET, batch_size=max(10, BUDGET//5000))

    logger = algorithms.TQDMAlgorithmLogger(
        algo, save_period=0, log_base_path=RUN_FOLDER)

    with ParallelismManager(max_workers=7) as mgr:
        mgr.executor._mp_context = (
            multiprocessing.get_context("fork"))  # TODO: Very brittle
        logging.info("Starting illumination!")
        algo.optimise(evaluate=eval_fn, executor=mgr.executor, batch_mode=True)

    with warnings.catch_warnings():      # Emits warning. Should be corrected
        warnings.simplefilter("ignore")  # upstream
        plots.default_plots_grid(logger, output_dir=RUN_FOLDER)

    return algo.__getstate__()


def process(data, is_test):
    grid: Grid = data["container"]
    population = data["population"]

    champions = []
    max_depth = grid.best_fitness
    for cell in grid.solutions.values():
        if not cell:
            continue
        ind = cell[0]
        if ind.fitness < max_depth:
            continue
        champions.append(ind)

    if is_test:
        champions = [champions[0]]

    for champion in champions:
        lineage = [champion]
        while parent_ids := lineage[-1].genome.parents():
            lineage.append(population.get(parent_ids[0]))

        name = f"lineage_{champion.genome.id()}"
        _fmt = f"{{:0{math.ceil(math.log10(len(lineage)))}d}}.png"
        folder = RUN_FOLDER.joinpath(name)
        folder.mkdir(parents=False, exist_ok=True)
        files = []

        lineage_sorted = sorted(lineage,
                                key=lambda _c: (_c.stats["hidden"],
                                                _c.stats["edges"]))

        for i, ind in track(enumerate(lineage_sorted),
                            description=name, total=len(lineage)):
            file = folder.joinpath(_fmt.format(i))
            files.append(file)
            if True or not file.exists():
                ann = ANN.build(Individual.INPUTS,
                                Individual.OUTPUTS,
                                ind.genome)
                pr = plotly_render(ann, edges_alpha=0.1)
                pr.write_html(RUN_FOLDER.joinpath(f"{name}.html"))
                xd = dict(title=None, visible=False, showticklabels=False)
                cr = 2
                pr.update_layout(
                    title=dict(automargin=False),
                    scene=dict(xaxis=xd, yaxis=xd, zaxis=xd),
                    margin=dict(autoexpand=False, b=0, l=0, r=0, t=0),
                    width=500, height=500,
                    scene_camera=dict(
                        eye=dict(
                            x=cr*math.cos(math.pi/4),
                            y=0,
                            z=cr*math.sin(math.pi/4))))
                with warnings.catch_warnings():      # Emits warning. Should be
                    warnings.simplefilter("ignore")  # corrected upstream
                    pr.write_image(str(file))

        frames = [PIL.Image.open(f) for f in files]
        duration = 8000 // len(frames)
        frames[0].save(RUN_FOLDER.joinpath(f"{name}.gif"),
                       format="GIF",
                       append_images=frames,
                       save_all=True, duration=duration,
                       loop=0)


def main(is_test=False):
    if EVOLVE:
        data = evolve(is_test)
    else:
        with open(RUN_FOLDER.joinpath("final.p"), "rb") as bf:
            data = pickle.load(bf)
    if PROCESS:
        process(data, is_test)
    return data


if __name__ == "__main__":
    main()
