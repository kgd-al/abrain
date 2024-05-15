API
===

Generic components
------------------

Either always useful or not dependent on a specific substrate dimension

.. autosummary::
    :nosignatures:

    abrain.Config
    abrain.Genome
    abrain.CPPN

2D
--

Generating ANNs on a 2D substrates: great for articles, bad for performance

.. autosummary::

    abrain.Point2D
    abrain.CPPN2D
    abrain.ANN2D

3D
--

Generating ANNs on a 3D substrate: great I/O separation, trickier to visualize

.. autosummary::

    abrain.Point3D
    abrain.CPPN3D
    abrain.ANN3D

Table of contents
-----------------

.. toctree::
    :maxdepth: 1

    config
    genome
    cppn
    ann
