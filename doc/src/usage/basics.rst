.. _usage-basics-ann:

Basic usage
-----------

This section showcases the main components of the library by detailing the
contents of `examples/basics.py`.

.. literalinclude:: ../../../examples/basics.py
    :lineno-match:
    :emphasize-lines: 4
    :lines: -5

We start by importing the essential components, aliased directly under the main
package:

- :class:`~abrain.Genome` abstracts the evolvable part of the library
- :class:`~abrain.ANN2D` is the callable object representing an Artificial
  Neural Network of emergent topology in 2 dimensions
- :class:`~abrain.Point3D` describes a coordinate in the substrate ("the brain")
- :func:`~abrain.ANN3D.render3D` is a helper function for rendering
  ANN to a, potentially interactive, figure
- :class:`random.Random` is used as the source of random numbers

.. note::

    While we illustrate here the use of a 3D ANN, everything (but the 3D rendering)
    works identically for :class:`abrain.ANN2D`

.. literalinclude:: ../../../examples/basics.py
    :lineno-match:
    :lines: 12-15

The first object we need is the Genome which can be created by providing the
random number generator to its :func:`~abrain.Genome.eshn_random` function.
This function is dedicated to generating a genome for use with Es-HyperNEAT while
:func:`~abrain.Genome.random` is meant for generic-purpose CPPNs.

To simulate an evolutionary process, we subject this Genome `g` to a number of
undirected mutations (see :ref:`usage-advanced-mutations`)

.. literalinclude:: ../../../examples/basics.py
    :lineno-match:
    :lines: 18-19

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
    :lineno-match:
    :emphasize-lines: 1
    :lines: 22-24

Creating the ANN is then as trivial as calling the static
:func:`~abrain.ANN3D.build` function with the set of inputs/outputs and the
evolved genome.
Various statistics can be queried on the resulting object including whether the
build procedure resulted in a functional network.

.. literalinclude:: ../../../examples/basics.py
    :lineno-match:
    :lines: 27

Optionally, one can produce a 3D rendering of the network through the utility
function :func:`~abrain.ANN3D.render3D`.

.. literalinclude:: ../../../examples/basics.py
    :lineno-match:
    :lines: 30-33

Actually using the ANN requires defining the neural inputs at a given time step,
which can be done by direct assignment (line 29) or through slices (line 30).
At the same time we also retrieve the output buffer which will store the neural
responses computed in the next step.

.. note::

    The default
    :ref:`activation function <_autogen/functions:activation function>`
    for every *hidden* and *output* neurons maps 0 to 0.
    By contrast input neurons expose the exact same value as that provided.
    This means that providing constant, small values might result in the whole
    network staying in a quiescent state.


.. literalinclude:: ../../../examples/basics.py
    :lineno-match:
    :lines: 36-37

Following that, we can query the outbound activity by invoking the ANN with both
buffers.
An optional parameter `substeps` can be provided if more than a single
activation step is desired, e.g. deep networks with a low update rate.

.. literalinclude:: ../../../examples/basics.py
    :lineno-match:
    :lines: 40

As with the input buffer, the results can be queried individually or in bulk to
set the robot's outputs (motors...).
