#!/usr/bin/env python3
import argparse
import importlib
import json
import logging
import os
import pprint
import random
import sys
import time
from dataclasses import field
from datetime import timedelta
from pathlib import Path

import humanize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import abrain._cpp.phenotype
import abrain

import numpy.polynomial.polynomial as npoly

EVOLVER = Path("examples/qdpy-evolution.py")


def __import_evolver():
    assert EVOLVER.exists()
    spec = importlib.util.spec_from_file_location(EVOLVER.stem, EVOLVER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[EVOLVER.stem] = module
    spec.loader.exec_module(module)
    return module


EVOLVER_MODULE = __import_evolver()


def __load_genome(file):
    with open(file, "r") as f:
        return abrain.Genome.from_json(json.load(f))


def __build(genome):
    return abrain.ANN.build(
        [abrain.Point(*p.tuple()) for p in EVOLVER_MODULE.Individual.INPUTS],
        [abrain.Point(*p.tuple()) for p in EVOLVER_MODULE.Individual.OUTPUTS],
        genome)


class Options:
    def __init__(self):
        self.evaluate: bool = True
        self.plot: bool = True
        self.iterations: int = 1
        self.sample: int = 10
        self.debug: bool = False
        self.i_folder = Path("tmp/ann_time_tests/data")

        self.o_folder = Path("tmp/ann_time_tests/results")

        self.tests = [True, False]

        self.stats_keys = ["hidden", "depth", "axons", "edges", "density", "iterations"]
        self.cppn_keys = ["nodes", "links"]
        self.time_keys = [f"{t}_{n}_t" for t in ["build", "eval"] for n in ["ann", "cppn"]]

    def dataframe(self): return self.o_folder.joinpath("stats.csv")

    @staticmethod
    def populate(parser: argparse.ArgumentParser):
        parser.add_argument('-N', '--iterations', type=int,
                            help="Specify number of iterations used to"
                                 " smoothen out stochasticity")
        parser.add_argument('--no-evaluate', action='store_false',
                            dest='evaluate', help="Use existing data")
        parser.add_argument('--no-plot', action='store_false',
                            dest='plot', help="Do not generate plots")
        parser.add_argument('--debug', action='store_true',
                            help="Write debug data to logs (huge slow down)")

        # abrain.Config.argparse_setup(parser)


def main():
    args = Options()
    parser = argparse.ArgumentParser(description="Rerun evolved champions")
    Options.populate(parser)
    parser.parse_args(namespace=args)

    genome_folder = args.i_folder.joinpath("genomes")
    if not args.i_folder.exists():
        EVOLVER_MODULE.BUDGET = 1_000
        EVOLVER_MODULE.EVOLVE = True
        EVOLVER_MODULE.PROCESS = False
        EVOLVER_MODULE.RUN_FOLDER = args.i_folder
        evolution_data = EVOLVER_MODULE.main()

        genome_folder.mkdir(parents=True, exist_ok=True)
        for i, genome in enumerate(
                [cell[0].genome for cell
                 in evolution_data["container"].solutions.values()
                 if len(cell) > 0]):
            with open(genome_folder.joinpath(f"{i}.json"), "w") as f:
                json.dump(genome.to_json(), f)

    files = genome_folder.glob("*.json")

    # os.environ["QT_QPA_PLATFORM"] = "offscreen"

    args.o_folder.mkdir(parents=True, exist_ok=True)

    if args.evaluate:
        df = evaluate(args, files)
    else:
        df = pd.read_csv(args.dataframe())

    print(df.sort_values(by=["file", "impl"]).to_string(max_rows=1000))

    if args.plot:
        plot(args, df)


def plot(args, df):
    o_folder = args.o_folder.joinpath("plots")
    o_folder.mkdir(parents=True, exist_ok=True)
    
    if len(args.tests) == 2:
        groups = df.groupby(by="impl")
        for lhs_k in args.cppn_keys + args.stats_keys:
            for rhs_k in args.time_keys:
                fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=True)
                coefficients = {}
                for key, ax in zip(groups.groups.keys(), axes.flatten()):
                    g = groups.get_group(key)
                    x, y = g[lhs_k], g[rhs_k]
                    x_min, x_max = np.quantile(x, [0, 1])

                    g.plot.scatter(x=lhs_k, y=rhs_k, ax=ax)

                    r = npoly.Polynomial.fit(x, y, deg=1, full=True)
                    b, a = r[0]
                    ax.plot(*r[0].linspace(100, ax.get_xlim()),
                            color='C1', label='ax+b')

                    coefficients[key] = [a, b]

                    ax.set_title(f"{key}: {a:.3g}x + {b:.3g}")
                    ax.set_yscale('log')

                def summarize():
                    return ", ".join(
                        f"{k}={100 * coefficients['py'][i] / coefficients['cpp'][i]:.0f}%"
                        for i, k in enumerate(["a", "b"])
                    )

                fig.suptitle(f"{lhs_k} / {rhs_k}: ({summarize()})")
                fig.tight_layout()
                o_file = o_folder.joinpath(f"{lhs_k}_vs_{rhs_k}.png")
                fig.savefig(o_file, bbox_inches='tight')
                print("Generated", o_file)
                plt.close(fig)


def evaluate(args, files):
    start = time.perf_counter()

    df = pd.DataFrame(
        columns=args.cppn_keys + args.stats_keys + args.time_keys + ["impl", "file"])

    files = list(files)
    if args.sample > 0:
        files = random.sample(files, args.sample)

    tests = [True, False]
    for pure_python in tests:
        ann_files = dict()
        abrain.use_pure_python(pure_python)
        print(
            f"Monkey-patching abrain for pure python: {pure_python}\n"
            + "\n".join(f"{c}" for c in [abrain.ANN, abrain.Point]))
        flag = "py" if pure_python else "cpp"
        logger = logging.getLogger(flag)
        file_handler = logging.FileHandler(args.o_folder.joinpath(
            flag + ".log"), mode="w")
        logger.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        if args.debug:
            logger.debug(flag)

        o_dir = args.o_folder.joinpath(flag)
        o_dir.mkdir(parents=True, exist_ok=True)

        if pure_python:
            def pretty_point(p_: abrain.Point):
                return "{" + ",".join(f"{v:+.3g}" for v in p_.__iter__()) + "}"

            def pretty_outputs(outs):
                return [o for o in outs]
        else:
            def pretty_point(p_: abrain.Point):
                return "{" + ",".join(f"{v:+.3g}" for v in p_.tuple()) + "}"

            def pretty_outputs(outs):
                return [outs[i_] for i_ in range(len(outs))]

        for i, file in enumerate(files):
            print(i, file, end=' ')

            genome: abrain.Genome = __load_genome(file)
            stats = dict()
            for k in args.stats_keys:
                stats[k] = []
            times = {k: [] for k in args.time_keys}

            for i in range(args.iterations):
                start_time = abrain._python.ann._time()
                cppn = abrain.CPPN(genome)
                times["build_cppn_t"].append(abrain._python.ann._time_diff(start_time))

                rng = random.Random(0)

                times["eval_cppn_t"].append(0)
                cppn_outputs = cppn.outputs()
                for _ in range(1000):
                    start_time = abrain._python.ann._time()

                    def p():
                        return abrain.Point(*[rng.uniform(-1, 1) for _ in range(3)])

                    p0, p1 = p(), p()
                    cppn(p0, p1, cppn_outputs)

                    if args.debug:
                        logger.debug(f"p0={pretty_point(p0)}"
                                     f" p1={pretty_point(p1)}"
                                     f" outputs={pretty_outputs(cppn_outputs)}")

                    times["eval_cppn_t"][-1] += abrain._python.ann._time_diff(start_time)

                ann = __build(genome)

                inputs, outputs = ann.buffers()
                for _ in range(1000):
                    inputs[:] = np.random.uniform(-1, 1, len(inputs))
                    ann(inputs, outputs)

                ann_stats = ann.stats().dict()
                for k in args.stats_keys:
                    stats[k].append(ann_stats[k])
                times["build_ann_t"].append(ann_stats["time"]["build"])
                times["eval_ann_t"].append(ann_stats["time"]["eval"])

            for k in args.stats_keys:
                assert len(set(stats[k])) == 1
                stats[k] = stats[k][0]
            for k in times:
                times[k] = np.average(times[k])

            # for k in stats_keys:
            #     if ind.stats[k] != stats[k]:
            #         logging.warning(f"[{file}:{k}] {ind.stats[k]} != {stats[k]}")

            df.loc[len(df)] = [
                                  len(genome.nodes), len(genome.links)
                              ] + list(stats.values()) + list(times.values()) + [
                                  flag, file
                              ]

            stats["time"] = times
            ind = dict(
                genome=genome.to_json(),
                stats=stats
            )

            hidden = stats["hidden"]
            o_file_id = ann_files.get(hidden, 0)
            ann_files[hidden] = o_file_id + 1

            o_file = o_dir.joinpath(f"ann_{hidden}_{o_file_id}.json")
            with open(o_file, "w") as f:
                json.dump(ind, f)
            print("->", o_file)
            # break

    df.to_csv(args.dataframe())

    duration = humanize.precisedelta(timedelta(seconds=time.perf_counter() - start))
    print(f"Generated ANN stats for {len(files) - 1} files in {duration}")

    return df


if __name__ == '__main__':
    main()
