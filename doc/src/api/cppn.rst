===================================
Composite Pattern-Producing Network
===================================

Generic
-------

.. autoclass:: abrain.CPPN
    :exclude-members: Output
    :special-members: __call__

ES-HyperNEAT
------------

.. tabs::

    .. group-tab:: 2D

        .. autoclass:: abrain.CPPN2D
            :exclude-members: Output
            :special-members: __call__

            .. autoclass:: abrain::CPPN2D.Output
                :no-undoc-members:
                :no-inherited-members:
                :exclude-members: name

    .. group-tab:: 3D

        .. autoclass:: abrain.CPPN3D
            :exclude-members: Output
            :special-members: __call__

            .. autoclass:: abrain::CPPN3D.Output
                :no-undoc-members:
                :no-inherited-members:
                :exclude-members: name
