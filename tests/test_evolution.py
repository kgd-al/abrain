import json
import math
import random
from pathlib import Path
from random import Random

from _utils import assert_equal
from abrain.core.ann import ANN, Point
from abrain.core.genome import Genome, GIDManager


class Robot:
    """A bare-bones individual

    :param genome: the collection of inheritable characteristics
    """

    ann_inputs = None
    ann_outputs = None

    def __init__(self, genome: Genome):
        self.genome = genome
        self.brain = ANN.build(Robot.ann_inputs, Robot.ann_outputs, genome)
        self.fitness = self.brain.stats().depth

    def __repr__(self):
        return f"Robot(id={self.genome.id()}," \
               f" depth={self.fitness}, neurons={len(self.brain.neurons())}," \
               f" axons={self.brain.stats().edges})"

    def to_json(self, file):
        with open(file, 'w') as f:
            json.dump(dict(
                genome=self.genome.to_json(),
                fitness=self.fitness
            ), f)

    @staticmethod
    def from_json_file(file):
        with open(file, 'r') as f:
            data = json.load(f)
            r = Robot(Genome.from_json(data["genome"]))
            r.fitness = data["fitness"]
            return r

    @staticmethod
    def assert_equal(lhs, rhs):
        assert_equal(lhs.genome, rhs.genome)
        assert lhs.fitness == rhs.fitness
        assert len(lhs.brain.neurons()) == len(rhs.brain.neurons())

        lhs_stats, rhs_stats = lhs.brain.stats(), rhs.brain.stats()
        assert lhs_stats.depth == rhs_stats.depth
        assert lhs_stats.edges == rhs_stats.edges
        assert lhs_stats.axons == rhs_stats.axons


def test_evolution(evo_config, capsys, tmp_path):
    pop_size = evo_config["population"]
    gens = evo_config["generations"]
    seed = evo_config["seed"]
    fitness = evo_config["fitness"]

    tour_size = 4
    rng = Random(seed)

    def random_coordinates(n_min, n_max, y):
        def rc(): return random.uniform(-1, 1)
        return [Point(rc(), y, rc()) for _ in
                range(random.randint(n_min, n_max))]

    # By personal convention inputs and outputs are on the y=-1 and y=1 planes,
    # respectively
    Robot.ann_inputs = random_coordinates(10, 20, -1)
    Robot.ann_outputs = random_coordinates(5, 10, 1)

    id_manager = GIDManager()

    with capsys.disabled():
        print()
        print("  ========================")
        print("==== Starting evolution ====")
        for k, v in evo_config.items():
            print(f"\t{k}: {v}")
        print()

        population = [Robot(Genome.random(rng, id_manager))
                      for _ in range(pop_size)]
        print("== Initialized population")

        def champion(pop=None) -> Robot:
            if pop is None:
                pop = population
            return pop[max(range(len(pop)), key=lambda r: pop[r].fitness)]

        if True:
            first_champ = champion()
            file = f"{tmp_path}/first_champion.json"
            first_champ.to_json(file)

            assert Path(file).exists()

            roundtrip_champ = Robot.from_json_file(file)
            Robot.assert_equal(first_champ, roundtrip_champ)

        for g in range(gens):
            print(f"> Best {fitness}: {champion().fitness}")

            new_population = []
            for _ in range(pop_size):
                winner = champion(rng.sample(population, tour_size))
                child = Robot(winner.genome.mutated(rng, id_manager))
                child.genome.update_lineage(id_manager, [winner.genome])
                new_population.append(child)

            population = new_population
            print("== Generated new population [{:0{width}d}] =="
                  .format(g+1, width=math.ceil(math.log10(gens))))

        print("== Final champion ==")
        champ = champion()
        print(f">> {champ}")

        base_path = f"{tmp_path}/champion_gen{gens}"

        json_path = f"{base_path}.json"
        champ.to_json(json_path)
        print(">>> Logged to", json_path)

        roundtrip_champ = Robot.from_json_file(json_path)
        Robot.assert_equal(champ, roundtrip_champ)

        dot_path = champ.genome.to_dot(base_path, "pdf")
        print(">>> Drawn to", dot_path)
