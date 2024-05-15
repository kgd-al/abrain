import sys
from random import Random

from abrain import Genome, Point3D as Pt, ANN3D
from common import example_path

seed = 0 if len(sys.argv) < 2 else int(sys.argv[1])
mutations = 10 if len(sys.argv) < 3 else int(sys.argv[2])
print(f"{seed=}, {mutations=}")

# /1/ Create/Evolve a genome
rng = Random(seed)
g = Genome.eshn_random(rng, dimension=2)
for _ in range(mutations):
    g.mutate(rng)

# /2/ Specify inputs/outputs based on your robots sensors/effectors positions
inputs = [Pt(-1, -1, -1), Pt(-1, -1, 1), Pt(1, -1, -1), Pt(1, -1, 1)]
outputs = [Pt(0, 1, -1), Pt(0, 1, 0), Pt(0, 1, 1)]

# /3/ Generate the Artificial Neural Network
ann = ANN3D.build(inputs, outputs, g)
print(f"empty ANN: {ann.empty()}")
print(f"maximal depth: {ann.stats().depth}")

# /3.1/ Generate visualization
ann.render3D().write_html(example_path("./sample_ann.html"))

# /4/ Assign neural inputs
inputs, outputs = ann.buffers()
inputs[0] = 1
inputs[1:3] = [rng.uniform(-1, 1) for _ in range(2)]
inputs[3] = -1

# /5/ Activate ANN n times
n = 5
ann(inputs, outputs, substeps=n)

# /6/ Retrieve responses
print("Outputs:", outputs[0], outputs[1:3])

# /7/ An empty ANN is generally useless
exit(ann.empty())
