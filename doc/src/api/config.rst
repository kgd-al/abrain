=============
Configuration
=============

.. autoclass:: abrain.Config
    :inherited-members:
    :member-order: groupwise

.. class:: abrain.core.config.Strings(values: list[str])
.. class:: abrain.core.config.MutationRates(values: dict[str,float])

.. class:: abrain.core.config.FBounds(min: float, rndMin: float, rndMax: float, max: float, stddev: float)

    A wrapper for mutation bounds.
    Absolute range is `[min, max]`. Values produced through random initialization
    are further restricted to `[rndMin, rndMax]` with

    .. math::
        min \leq rndMin \leq rndMax \leq max

    `stddev` is the standard deviation for every point-mutation applied to the
    corresponding field.

.. module:: abrain.core.config

    Contains the classes and functions related to abrain's configuration
