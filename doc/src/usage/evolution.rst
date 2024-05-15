.. _usage-basics-evolution:

Basic evolution
---------------

This section focuses on using the library in an evolutionary context.
It showcases:

    * how to include `abrain.Genome` into another class
    * how to use the configuration
    * how to actually produce novel solutions

First we import the relevant modules from the library (among others)

.. literalinclude:: ../../../examples/evolution.py
    :lineno-match:
    :start-after: /- abrain imports -/
    :end-before: /- abrain imports -/

As with the previous example, we need :class:`~abrain.Genome`,
:class:`~abrain.ANN3D` and :class:`~abrain.Point3D` to encode/decode a 3D Artificial
Neural Network.
:class:`~abrain.Config` is responsible for statically stored settings and
persistent configuration files. In this specific case, we also need
:class:`~abrain.core.config.Config.Strings` for one particular value.

Finally, the :class:`~abrain.GIDManager` is a tiny helper class
to provide consecutive int-based id to the genome. It can be used as-is,
replaced with a more powerful alternative or just ignored if you do not need
to identify individual genomes.

Helper classes
==============

To showcase real use of the genome, we define a trivial wrapper containing two
fields:

.. literalinclude:: ../../../examples/evolution.py
    :lineno-match:
    :emphasize-lines: 7, 11, 17, 23
    :pyobject: MyGenome

The presented pattern consists of the two essential functions `random` (to
generate the initial population) and `mutated` (to create a mutated copy of a
genome). The `mutate` function performs the bulk of the work by delegating
to field-wise mutators (including :func:`abrain.Genome.mutate`).
Note that, if using an id generator (such as
:class:`abrain.GIDManager`) you can use the
:func:`~abrain.Genome.update_lineage` function to update the `id`/`parents`
fields based on the values of the parents (self in case of a mutation).

We then define an individual, in the sense of an evolutionary algorithm, as the
composition of a genome and a fitness (trivially based on the ANN's depth).
For completeness, we provide a serialization method which relies on
:func:`abrain.Genome.to_json`.

.. literalinclude:: ../../../examples/evolution.py
    :lineno-match:
    :emphasize-lines: 17
    :pyobject: Individual

The main
==============

The following sections describe the components of a trivial EA and how to use
the various parts of `abrain` to smoothly implement them.

Configuration
*************

The following lines showcase how the end-user can tweak the various fields in
:class:`abrain.Config`:

.. literalinclude:: ../../../examples/evolution.py
    :lineno-match:
    :start-after: /- configuration -/
    :end-before: /- configuration -/

Most such fields use elementary python types (`int`, `float`, `str`, `bool`)
and can thus be trivially manipulated. A few other use composite types
encapsulated, for type-safety, in a C++ object. Those are exposed in the
:mod:`abrain.core.config` module and can be used to generate new values.
Additionally, the configuration can be written to a file (and read back with
:func:`~abrain.Config.read`) and displayed on the screen (for the log).

Variables
*********

The initial state of this trivial EA is just as straightforward. The only thing
of note is the highlighted statement where we create the genome id manager.
This is purely optional and only provided for convenience.

The actual generation of the initial population simply consists of delegating
the work to the dedicated function in our wrapper genome.

.. literalinclude:: ../../../examples/evolution.py
    :lineno-match:
    :emphasize-lines: 3
    :start-after: /- init -/
    :end-before: /- init -/
