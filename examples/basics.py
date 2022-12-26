from abrain import Genome, Point as Pt, ANN, plotly_render
from random import Random
import sys

seed = 0 if len(sys.argv) < 2 else int(sys.argv[1])
mutations = 10 if len(sys.argv) < 3 else int(sys.argv[2])
print(f"{seed=}, {mutations=}")

# /1/ Create/Evolve a genome
rng = Random(seed)
g = Genome.random(rng)
for _ in range(mutations):
    g.mutate(rng)

# /2/ Specify inputs/outputs based on your robots sensors/effectors positions
inputs = [Pt(-1, -1, -1), Pt(-1, -1, 1), Pt(1, -1, -1), Pt(1, -1, 1)]
outputs = [Pt(0, 1, -1), Pt(0, 1, 0), Pt(0, 1, 1)]

# /3/ Generate the Artificial Neural Network
ann = ANN.build(inputs, outputs, g)
print(f"empty ANN: {ann.empty()}")
print(f"maximal depth: {ann.stats().depth}")

# /3.1/ Generate visualization
plotly_render(ann).write_html("./sample_ann.html")

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
