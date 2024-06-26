"""
Reasonable attempt at making abrain play nice with leap_ec.
Provided as inspiration without any guarantee of functionality
"""

import itertools
import math
import shutil

import leap_ec
import pandas as pd
from leap_ec import Decoder, context
from leap_ec.multiobjective.nsga2 import generalized_nsga_2
from leap_ec.multiobjective.ops import rank_ordinal_sort
from leap_ec.multiobjective.problems import MultiObjectiveProblem
from leap_ec.ops import tournament_selection, clone, evaluate, pool
from matplotlib import pyplot as plt
from rich.progress import Progress

from abrain import Config, Genome, Point3D as Point, ANN3D as ANN
from examples.common import example_path

_inputs = [Point(x, -1, z) for x, z
           in itertools.product([0, .25, .5, .75, 1],
                                [0, .2, .4, .6, .8, 1])]
_outputs = [Point(r * math.cos(a),
                  1,
                  r * math.sin(a)) for r, a in
            [(0, 0)] + [(1, i * math.pi / 10) for i in range(5)]]

_data = Genome.Data.create_for_eshn_cppn(dimension=3, seed=0)


class ANNBuilder(Decoder):
    def decode(self, genome, *args, **kwargs):
        return ANN.build(_inputs, _outputs, genome)


class Problem(MultiObjectiveProblem):
    fitness_names = ["Depth", "Density", "Utility"]

    def __init__(self):
        super().__init__([True, False, True])

    def evaluate(self, ann, *args, **kwargs):
        return ann.stats().depth, ann.stats().density, ann.stats().utility


class Representation(leap_ec.Representation):
    def __init__(self):
        super().__init__(
            initialize=lambda: Genome.random(_data),
            decoder=ANNBuilder(),

        )


def mutate(next_individual):
    while True:
        individual = next(next_individual)
        individual.genome.mutate(_data)
        individual.genome.update_lineage(_data, [individual.genome])
        individual.fitness = None
        yield individual


def main(is_test=False):
    Config.iterations = 32
    run_folder = example_path("leap-evolution")
    pop_size = 10 if is_test else 100
    generations = 10 if is_test else 100

    _g_fmt = f"{{:0{max(3, math.ceil(math.log10(generations)))}d}}"

    progress = Progress()
    progress.start()
    _p_bar = progress.add_task("Evolving", total=generations)

    def dataframe(population):
        with_rank = hasattr(population[0], "rank")
        if with_rank:
            def _get(x): return x.genome.id(), *x.fitness, x.rank, x.distance
        else:
            def _get(x): return x.genome.id(), *x.fitness

        _df = pd.DataFrame([_get(x) for x in population])
        _df.columns = ["ID", *Problem.fitness_names] \
            + (["Rank", "Distance"] if with_rank else [])

        return _df

    def on_generation(population):
        progress.advance(_p_bar)

        _g = context['leap']['generation']

        _df = dataframe(population)
        if _g == 0:
            progress.console.print(
                "Gen", " ".join(f"{c:7s}" for c in Problem.fitness_names))
        msg = _g_fmt.format(_g)
        for _i, c in enumerate(Problem.fitness_names):
            _ix = _df[c].argmax()
            msg += f" {population[_ix].fitness[_i]:7.2g}"
        progress.console.print(msg)

        return population

    if run_folder.exists():
        shutil.rmtree(run_folder)
    run_folder.mkdir(parents=True, exist_ok=False)

    pipeline = [
        tournament_selection,
        clone,
        mutate,
        evaluate,
        pool(size=pop_size),
        on_generation
    ]

    solutions = generalized_nsga_2(
        max_generations=generations,
        pop_size=pop_size,
        problem=Problem(),
        representation=Representation(),
        pipeline=pipeline,
        rank_func=rank_ordinal_sort
    )

    # pprint.pprint(solutions.__dict__)

    progress.stop()

    df = dataframe(solutions)
    print(df)
    df.to_csv(run_folder.joinpath("solutions.csv"))

    fig, axes = plt.subplots(ncols=3, subplot_kw=dict(box_aspect=1),
                             figsize=(15, 5))
    for ax, (i, j) in zip(axes, [(0, 1), (0, 2), (1, 2)]):
        _n = Problem.fitness_names
        df.plot.scatter(ax=ax, x=_n[i], y=_n[j])
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(run_folder.joinpath("pareto." + ext))


if __name__ == '__main__':
    main()
