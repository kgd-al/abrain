=============
Configuration
=============

.. module:: abrain.core.config

    Contains the classes and functions related to abrain's configuration

.. autoclass:: abrain.Config
    :inherited-members:
    :exclude-members: OutputFunctions, Strings, MutationRates, ESHNOutputs, FBounds
    :member-order: groupwise

.. autoclass:: abrain.Config.ESHNOutputs
.. class:: Config.Strings(values: list[str])
.. class:: Config.MutationRates(values: dict[str,float])

.. class:: FBounds(min: float, rndMin: float, rndMax: float, max: float, stddev: float)

    A wrapper for mutation bounds.
    Absolute range is `[min, max]`. Values produced through random initialization
    are further restricted to `[rndMin, rndMax]` with

    .. math::
        min \leq rndMin \leq rndMax \leq max

    `stddev` is the standard deviation for every point-mutation applied to the
    corresponding field.

Internals
---------
