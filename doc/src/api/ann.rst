=========================
Artificial Neural Network
=========================

.. todo:: Missing reference to plotly (hacked something in but fails with 404)
.. todo:: Implement read-/write-through I/O Buffers

Main object:
^^^^^^^^^^^^

.. autoclass:: abrain.ANN
    :exclude-members: IBuffer, OBuffer, Neuron, Neurons, Stats
    :special-members: __call__

Rendering tool(s):
^^^^^^^^^^^^^^^^^^

.. autofunction:: abrain.plotly_render


Supporting types:
^^^^^^^^^^^^^^^^^

.. autoclass:: abrain.Point
    :special-members: __init__

.. autoclass:: abrain.ANN.IBuffer
.. autoclass:: abrain.ANN.OBuffer

Underlying types:
^^^^^^^^^^^^^^^^^

.. autoclass:: abrain.ANN.Neuron
    :exclude-members: Type

    .. autoclass:: abrain.ANN.Neuron.Type
        :no-undoc-members:
        :no-inherited-members:
        :exclude-members: name

.. autoclass:: abrain.ANN.Neurons
.. autoclass:: abrain.ANN.Stats
