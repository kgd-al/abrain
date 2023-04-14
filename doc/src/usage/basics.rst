.. _usage-basics-ann:

Basic usage
-----------

This section showcases the main components of the library by detailing the
contents of `examples/basics.py`.

.. literalinclude:: ../../../examples/basics.py
    :language: python
    :linenos:
    :lineno-match:
    :lines: -2

We start by importing the essential components, aliased directly under the main
package:

- :class:`~abrain.Genome` abstracts the evolvable part of the library
- :class:`~abrain.ANN` is the callable object representing an Artificial
  Neural Network of emergent topology
- :class:`~abrain.Point` describes a coordinate in the substrate ("the brain")
- :func:`~abrain.plotly_render` is a helper function for rendering
  ANN to a, potentially interactive, figure
- :class:`random.Random` is used as the source of random numbers

.. literalinclude:: ../../../examples/basics.py
    :language: python
    :linenos:
    :lineno-match:
    :lines: 10-13

The first object we need is the Genome which can be created by providing the
random number generator to its :func:`~abrain.Genome.random` function.
To simulate an evolutionary process, we subject this Genome `g` to a number of
undirected mutations (see :ref:`usage-advanced-mutations`)

.. literalinclude:: ../../../examples/basics.py
    :language: python
    :linenos:
    :lineno-match:
    :lines: 16-17

Before instantiating the ANN, we define the coordinates of the neural inputs
(sensors) and outputs (effectors).
Thanks to the `ES-HyperNEAT <https://doi.org/10.1162/ARTL_a_00071>`_ algorithms
the topology (hidden neurons & connections) will be automatically determined.
As much as possible, the provided coordinates should respect the geometrical
relationships (i.e. bilateral symmetry, front-back ...).

.. warning::

    It is essential that neurons are placed at *unique* coordinates
    including hidden ones.
    Safe coordinates for inputs/outputs are of the form

    .. math::

        \{(x,y,z) /\ \exists c \in \{x,y,z\} /\ \not\exists \{i_1,...,i_n\}
        /\ c = \sum_{j=1}^n 2^{-i_j}\}

    with :math:`1 \leq i_j \leq` :attr:`~abrain.Config.maxDepth`

    In particular, this means that all outside planes (e.g. :math:`y=\pm 1`)
    can never contain hidden neurons and are thus safe for user-defined
    inputs/outputs

.. literalinclude:: ../../../examples/basics.py
    :language: python
    :linenos:
    :lineno-match:
    :emphasize-lines: 1
    :lines: 20-22

Creating the ANN is then as trivial as calling the static
:func:`~abrain.ANN.build` function with the set of inputs/outputs and the
evolved genome.
Various statistics can be queried on the resulting object including whether the
build procedure resulted in a functional network.

.. literalinclude:: ../../../examples/basics.py
    :language: python
    :linenos:
    :lineno-match:
    :lines: 25

Optionally, one can produce a 3D rendering of the network through the utility
function :func:`~abrain.plotly_render`.

.. literalinclude:: ../../../examples/basics.py
    :language: python
    :linenos:
    :lineno-match:
    :lines: 28-31

Actually using the ANN requires defining the neural inputs at a given time step,
which can be done by direct assignment (line 29) or through slices (line 30).
At the same time we also retrieve the output buffer which will store the neural
responses computed in the next step.

.. warning::

    The default :ref:`activation function <api/functions.html#activation-function>`


.. literalinclude:: ../../../examples/basics.py
    :language: python
    :linenos:
    :lineno-match:
    :lines: 34-35

Following that, we can query the outbound activity by invoking the ANN with both
buffers.
An optional parameter `substeps` can be provided if more than a single
activation step is desired, e.g. deep networks with a low update rate. .

.. literalinclude:: ../../../examples/basics.py
    :language: python
    :linenos:
    :lineno-match:
    :lines: 38

As with the input buffer, the results can be queried individually or in bulk to
set the robot's outputs (motors...).
