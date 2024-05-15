=========================
Artificial Neural Network
=========================

.. todo:: Missing reference to plotly (hacked something in but fails with 404)
.. todo:: Implement read-/write-through I/O Buffers

.. tabs::

    .. group-tab:: 2D

        .. autoclass:: abrain.ANN2D
            :exclude-members: IBuffer, OBuffer, Neuron, Neurons, Stats
            :special-members: __call__

    .. group-tab:: 3D

        .. autoclass:: abrain.ANN3D
            :exclude-members: IBuffer, OBuffer, Neuron, Neurons, Stats
            :special-members: __call__

Supporting types:
^^^^^^^^^^^^^^^^^


.. tabs::

    .. group-tab:: 2D

        .. autoclass:: abrain.Point2D
            :special-members: __init__

        .. autoclass:: abrain.ANN2D.IBuffer
        .. autoclass:: abrain.ANN2D.OBuffer

    .. group-tab:: 3D

        .. autoclass:: abrain.Point3D
            :special-members: __init__

        .. autoclass:: abrain.ANN3D.IBuffer
        .. autoclass:: abrain.ANN3D.OBuffer

Underlying types:
^^^^^^^^^^^^^^^^^


.. tabs::

    .. group-tab:: 2D

        .. autoclass:: abrain.ANN2D.Neuron
            :exclude-members: Type

            .. autoclass:: abrain.ANN2D.Neuron.Type
                :no-undoc-members:
                :no-inherited-members:
                :exclude-members: name

        .. autoclass:: abrain.ANN2D.Neurons
        .. autoclass:: abrain.ANN2D.Stats

    .. group-tab:: 3D

            .. autoclass:: abrain.ANN3D.Neuron
                :exclude-members: Type

                .. autoclass:: abrain.ANN3D.Neuron.Type
                    :no-undoc-members:
                    :no-inherited-members:
                    :exclude-members: name

            .. autoclass:: abrain.ANN3D.Neurons
            .. autoclass:: abrain.ANN3D.Stats
