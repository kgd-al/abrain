import math
import os
import pprint
import random
import time
from random import Random

import numpy as np
import pytest
from matplotlib import pyplot as plt, lines
from matplotlib.collections import LineCollection

from abrain.neat.evolver import NEATEvolver, NEATConfig


class RGBXYGenome:
    __key = object()

    ranges = dict(
        x=[-1, 1], y=[-1, 1],
        r=[0, 1], g=[0, 1], b=[0, 1]
    )

    def __init__(self, key, *args):
        assert key == self.__key
        self.x, self.y, self.r, self.g, self.b = args

    @classmethod
    def random(cls, rng: Random):
        # c, s, pi = math.cos, math.sin, math.pi
        # return cls(cls.__key, c(pi/4), c(pi/4), 1, 1, 0)
        # return cls(cls.__key, 1, 0, 1, 0, 0)
        # return cls(cls.__key, c(pi*2/3), s(pi*2/3), 0, 1, 0)
        # return cls(cls.__key, c(pi*4/3), s(pi*4/3), 0, 0, 1)
        return cls(cls.__key, *[rng.uniform(rl/100, ru/100)
                                for rl, ru in cls.ranges.values()])

    def mutate(self, rng: Random):
        fields = [f for f in self.__dict__ if len(f) == 1]

        def update(field, amplitude):
            field_range = self.ranges[field]
            dev = (field_range[1] - field_range[0]) * amplitude
            value = (getattr(self, field) + rng.normalvariate(0, dev))
            value = max(field_range[0], min(field_range[1], value))
            setattr(self, field, value)

        scale = .02
        if rng.random() < .25:
            for f in fields:
                update(f, scale / 2)
        else:
            update(rng.choice(fields), scale)

    def crossover(self, other: 'RGBXYGenome', rng: Random):
        return RGBXYGenome(self.__key, *[
            rng.choice([a, b]) for a, b in zip(iter(self), iter(other))
        ])

    def __repr__(self):
        return (f"[{id(self)}|{self.x}, {self.y}|#"
                + "".join(f"{round(255*x):x}" for x in [self.r, self.g, self.b])
                + "]")

    def __iter__(self):
        return iter([self.x, self.y, self.r, self.g, self.b])

    def distance(self, other):
        return math.sqrt(sum((lhs_v - rhs_v)**2 for lhs_v, rhs_v in zip(
            iter(self), iter(other))))

    @staticmethod
    def evaluate(genome):
        r, g, b = genome.r, genome.g, genome.b
        a = math.atan2(math.sqrt(3) * (g - b) / 2, .5 * (2*r - g - b))
        l = max(r, g, b)

        x, y = math.cos(a), math.sin(a)
        res = -math.sqrt((genome.x - x) ** 2 + (genome.y - y) ** 2) + l - 1

        # time.sleep(random.random()*2)
        # print(f"[{os.getpid()}]", genome)

        return res

    @staticmethod
    def fitness_range(): return -math.sqrt(2) - 1, 0


@pytest.mark.parametrize("neat_seed", list(range(10)))
def test_evolver(neat_seed, tmp_path):
    # neat_seed = 1
    rng = Random(neat_seed)
    pop, gen = 100, 100
    species = 8
    config = NEATConfig(
        seed=neat_seed,
        threads=4,
        log_dir=tmp_path,
        log_level=10,
        population_size=pop,
        species_count=species,
        initial_distance_threshold=.1,
        overwrite=True,
    )
    pprint.pprint(config)
    evolver = NEATEvolver(config,
                          evaluator=RGBXYGenome.evaluate,
                          genome_class=RGBXYGenome, genome_data=dict(rng=rng))

    gen_digits = math.ceil(math.log10(gen+1))

    def plot(g, population=True, suffix=""):
        fig, ax = plt.subplots()
        ax.set_xlim([-1.05, 1.05])
        ax.set_ylim([-1.05, 1.05])
        ax.set_box_aspect(1)

        a = np.linspace(0, 2*math.pi, 500)
        x, y = np.cos(a), np.sin(a)

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap="hsv", zorder=-10)
        lc.set_array(a)
        ax.add_collection(lc)

        ls_levels = 10
        def ls_dashes(_x): return 0, (1, int((1-_x)*ls_levels))

        norm = plt.Normalize(*RGBXYGenome.fitness_range())

        def do_plot(_p, **kwargs):
            xs, ys, fcs, ecs = (
                zip(*[(_g.x, _g.y, (_g.r, _g.g, _g.b), _i.fitness)
                      for _i in sorted(_p, key=lambda __i: __i.fitness)
                      if (_g := _i.genome)]))
            ls = [ls_dashes(_x) for _x in norm(ecs)]
            args = dict(facecolors=fcs, edgecolors="black", linestyles=ls)
            args.update(**kwargs)
            ax.scatter(xs, ys, **args)

        if population:
            do_plot(evolver.population)
        do_plot([s.representative for s in evolver.species], marker='D')

        fig.legend(handles=[
            lines.Line2D([], [],
                         color='black', linestyle=ls_dashes(norm(x)),
                         label=f'{x:.2g}')
            for x in reversed(np.linspace(*RGBXYGenome.fitness_range(),
                                          ls_levels))
        ], loc='outside right', title='Fitness')

        fig.savefig(tmp_path.joinpath(
            f"population{g:0{gen_digits}d}{suffix}.png"),
                    bbox_inches="tight")
        plt.close()

    plot(0)
    with evolver:
        for i in range(gen):
            evolver.step()
            plot(i+1)
    plot(gen, population=False, suffix="_champions")
    # evolver.generate_plots()
    evolver.generate_plots(ext="pdf", options=dict())
