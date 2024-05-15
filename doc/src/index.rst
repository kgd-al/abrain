ABrain's documentation
======================

This package implements the ES-HyperNEAT algorithms for the production of
large-scale, regular Artificial Neural Networks.
For a adequate overview of the related literature see the official homepage
(http://eplex.cs.ucf.edu/ESHyperNEAT/).

Currently, the package can evolve both 2D and 3D ANNs and also exposes a
generic CPPN e.g. for body/brain co-evolution.
Illustrative examples are available under :ref:`usage/index:Usage` and the full
:ref:`api/index:API` is documented under its own section.

Notable missing features:

- Crossover and historical markings (WIP)
- Built-in NEAT evolution algorithms (WIP)
- Hardware-accelerated ANN activation as a sparse-matrix, when relevant (WIP)

Contents
--------

.. toctree::
    :maxdepth: 2

    usage/index
    api/index

.. toctree::
    :maxdepth: 1

    api/functions

    misc


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



