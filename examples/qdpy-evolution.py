import itertools
import json
import logging
import math
import multiprocessing
import random
import shutil
from pathlib import Path
from random import Random

import numpy as np
from qdpy import algorithms, tools, containers, plots
from qdpy.algorithms import Evolution, partial
from qdpy.base import ParallelismManager
from qdpy.containers import Grid, Container
from qdpy.phenotype import IndividualLike, Fitness, Fitness, Features

from abrain import Point, ANN
from abrain.core.config import Config
from abrain.core.genome import Genome, GIDManager


class Individual(IndividualLike):
    _inputs = [Point(x, -1, z) for x, z
               in itertools.product([0, .25, .5, .75, 1],
                                    [0, .2, .4, .6, .8, 1])]
    _outputs = [Point(r * math.cos(a),
                    1,
                      r * math.sin(a)) for r, a in
                [(0, 0)] + [(1, i*math.pi/10) for i in range(5)]]

    def __init__(self, genome: Genome):
        self.genome = genome
        self._fitness = None
        self._features = None
        self.stats = None
        self.name = str(genome.id())

    def evaluate(self):
        if self._fitness is None:
            ann = ANN.build(self._inputs, self._outputs,
                            self.genome)
            self._fitness = ann.stats().depth
            self._features = (
                ann.stats().density,
                ann.stats().iterations
            )
            self.stats = ann.stats().dict()

    @staticmethod
    def ranges():
        return dict(
            fitness_domain=[(0, Config.iterations)],
            features_domain=[(0, 1),
                             (0, Config.iterations)]
        )

    @staticmethod
    def grid_shape():
        return 32, Config.iterations

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
        self.rng = Random(seed)
        random.seed(seed)
        np.random.seed(seed % (2**32-1))

        self.id_manager = GIDManager()

        def select(grid):
            # return self.rng.choice(grid)
            k = min(len(grid), tournament)
            candidates = self.rng.sample(grid.items, k)
            candidate_cells = [grid.index_grid(c.features) for c in candidates]
            cell = self.rng.choice(candidate_cells)
            selection = candidates[candidate_cells.index(cell)]
            return selection

        def init(_):
            genome = Genome.random(self.rng, self.id_manager)
            for _ in range(initial_mutations):
                genome.mutate(self.rng)
            return Individual(genome)

        def vary(parent: Individual):
            return Individual(parent.genome.mutated(self.rng, self.id_manager))

        sel_or_init = partial(tools.sel_or_init,
                              init_fn=init, sel_fn=select, sel_pb=1)

        run_folder = Path(run_folder)
        if run_folder.exists():
            shutil.rmtree(run_folder, ignore_errors=True)
            logging.warning(f"Purging contents of {run_folder}, as requested")

        run_folder.mkdir(parents=True, exist_ok=False)
        print(f"[kgd-debug] Using run folder", run_folder.resolve())

        self._latest_champion = None
        self._snapshots = run_folder.joinpath("snapshots")
        self._snapshots.mkdir(exist_ok=False)
        print("[kgd-debug] Created folder", self._snapshots, self._snapshots.exists())

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
                print("[kgd-debug] First champion:", self._latest_champion)
            else:
                n, timestamp, fitness = self._latest_champion
                if individual.fitness.dominates(fitness):
                    self._latest_champion = (n + 1, self.nb_evaluations,
                                             individual.fitness)
                    new_champion = True
                    print("[kgd-debug] New champion:", self._latest_champion)

            if new_champion:
                n, t, _ = self._latest_champion
                file = self._snapshots.joinpath(f"better-{n}-{t}.json")
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


def main():
    run_folder = "tmp/qdpy-evolution"
    Config.iterations = 32

    grid = containers.Grid(
        shape=Individual.grid_shape(),
        max_items_per_bin=1,
        **Individual.ranges())

    algo = Algorithm(
        grid, seed=0, tournament=4, initial_mutations=10,
        run_folder=run_folder,
        budget=50_000, batch_size=128)

    ind, _ = algo._select_or_initialise_fn([], object())
    print(ind.evaluate())

    logger = algorithms.TQDMAlgorithmLogger(
        algo, save_period=10, log_base_path=run_folder)

    with ParallelismManager(max_workers=7) as mgr:
        mgr.executor._mp_context = multiprocessing.get_context("fork")  # TODO: Very brittle
        logging.info("Starting illumination!")
        best = algo.optimise(evaluate=eval_fn, executor=mgr.executor, batch_mode=True)

    plots.default_plots_grid(logger, output_dir=run_folder)


if __name__ == "__main__":
    main()
