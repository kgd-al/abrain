import json
import shutil

# /- abrain imports -/
from abrain import Config, Genome, ANN3D as ANN, Point3D as Point
# /- abrain imports -/

from examples.common import example_path


class MyGenome:
    def __init__(self, abrain_genome: Genome, nested_field: float):
        self.abrain_genome = abrain_genome
        self.nested_field = nested_field

    @staticmethod
    def random(data: Genome.Data):
        return MyGenome(Genome.random(data), data.rng.uniform(-1, 1))

    def mutate(self, data: Genome.Data):
        if data.rng.random() < .9:
            self.abrain_genome.mutate(data)
        else:
            self.nested_field += data.rng.normalvariate(0, 1)

    def mutated(self, data: Genome.Data):
        copy = self.copy()
        copy.mutate(data)
        copy.abrain_genome.update_lineage(data, [self.abrain_genome])
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


def main(is_test=False):
    output_folder = example_path("evolution")
    shutil.rmtree(output_folder, ignore_errors=True)
    output_folder.mkdir(exist_ok=False)

    # /- configuration -/
    Config.functionSet = Config.Strings(['sin', 'abs', 'id'])
    Config.allowPerceptrons = False
    Config.iterations = 4
    Config.write(output_folder.joinpath("config.json"))
    Config.show()
    # /- configuration -/

    # /- init -/
    pop_size = 10 if is_test else 100
    genome_shared_data = Genome.Data.create_for_eshn_cppn(
        dimension=3, seed=0,
        with_lineage=True
    )
    population = [Individual(MyGenome.random(genome_shared_data))
                  for _ in range(pop_size)]
    # /- init -/

    def get_champion(pop=None) -> Individual:
        if pop is None:
            pop = population
        return pop[max(range(len(pop)), key=lambda r: pop[r].fitness)]

    gen_champion = None
    generations = 10 if is_test else 100
    for g in range(generations):
        for p in population:
            p.evaluate()

        gen_champion = get_champion()
        print(f"{g}: {gen_champion.fitness}")

        new_population = []
        for _ in range(len(population)):
            competitors = genome_shared_data.rng.sample(population, 5)
            champion = get_champion(competitors)
            new_population.append(
                Individual(
                    champion.genome.mutated(genome_shared_data)))

        population = new_population

    print("-"*80)
    print("Final champion:", gen_champion.fitness)
    print(gen_champion.genome.nested_field)
    print(gen_champion.genome.abrain_genome.to_json())

    with open(output_folder.joinpath("champion.json"), 'w') as f:
        gen_champion.write(f)


if __name__ == '__main__':
    main()
