import json
import shutil
from random import Random

# /- abrain imports -/
from abrain import Config, Genome, ANN, Point
from abrain.core.config import Strings
from abrain.core.genome import GIDManager
# /- abrain imports -/

from common import example_path


class MyGenome:
    def __init__(self, abrain_genome: Genome, nested_field: float):
        self.abrain_genome = abrain_genome
        self.nested_field = nested_field

    @staticmethod
    def random(rng: Random, id_m: GIDManager):
        return MyGenome(Genome.random(rng, id_manager=id_m),
                        rng.uniform(-1, 1))

    def mutate(self, rng: Random):
        if rng.random() < .9:
            self.abrain_genome.mutate(rng)
        else:
            self.nested_field += rng.normalvariate(0, 1)

    def mutated(self, rng: Random, id_m: GIDManager):
        copy = self.copy()
        copy.mutate(rng)
        copy.abrain_genome.update_lineage(id_m, [self.abrain_genome])
        return copy

    def copy(self):
        return MyGenome(
            self.abrain_genome.copy(),
            self.nested_field
        )


class Individual:
    _inputs = [Point(x, -1, z) for x, z in [(0, 0), (-1, -1), (1, 1)]]
    _outputs = [Point(x, 1, z) for x, z in [(0, 0), (1, -1), (-1, 1)]]

    def __init__(self, genome: MyGenome):
        self.genome = genome
        self.fitness = None

    def evaluate(self):
        if self.fitness is None:
            ann = ANN.build(self._inputs, self._outputs,
                            self.genome.abrain_genome)
            self.fitness = self.genome.nested_field * ann.stats().depth

    def write(self, file):
        json.dump(dict(
            abrain_genome=self.genome.abrain_genome.to_json(),
            float_field=self.genome.nested_field,
            fitness=self.fitness
        ), file)

def main():
    output_folder = example_path("evolution")
    shutil.rmtree(output_folder, ignore_errors=True)
    output_folder.mkdir(exist_ok=False)

    # /- configuration -/
    Config.functionSet = Strings(['sin', 'abs', 'id'])
    Config.allowPerceptrons = False
    Config.iterations = 4
    Config.write(output_folder.joinpath("config.json"))
    Config.show()
    # /- configuration -/

    # /- init -/
    seed = 0
    rng = Random(seed)
    id_manager = GIDManager()
    population = [Individual(MyGenome.random(rng, id_manager))
                  for _ in range(100)]
    # /- init -/

    def get_champion(pop=None) -> Individual:
        if pop is None:
            pop = population
        return pop[max(range(len(pop)), key=lambda r: pop[r].fitness)]

    for g in range(100):
        for p in population:
            p.evaluate()

        gen_champion = get_champion()
        print(f"{g}: {gen_champion.fitness}")

        new_population = []
        for _ in range(len(population)):
            competitors = rng.sample(population, 5)
            champion = get_champion(competitors)
            new_population.append(
                Individual(
                    champion.genome.mutated(rng, id_manager)))

        population = new_population

    print("-"*80)
    print("Final champion:", gen_champion.fitness)
    print(gen_champion.genome.nested_field)
    print(gen_champion.genome.abrain_genome.to_json())

    with open(output_folder.joinpath("champion.json"), 'w') as f:
        gen_champion.write(f)


if __name__ == '__main__':
    main()
