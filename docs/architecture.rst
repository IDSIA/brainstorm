============
Architecture
============

Network architecture is a dictionary mapping layer names to their properties.
There are two special properties:

  1. ``@type``: a string that specifies the class of the layer
  2. ``@connections``: either a list of sink names to connect to or a dictionary
     mapping the named outputs (sources) of the layer to the lists of sinks
     it connects to. A sink name is a layer-name followed by a ``.`` and the
     name of that layers input. If the layer has only one input the second
     part can be omitted.

There can be more properties that will be passed on to the layer class when
instantiating them.

An example showcasing most features

.. code-block:: python

    architecture = {
        'InputLayer': {
            '@type': 'InputLayer',
            '@connections': ['splitter', 'output'],
            'shape': 20},
        'splitter': {
            '@type': 'SplitLayer',
            '@connections': {
                'left': ['adder.A']
                'right': ['adder.B']},
            'split_at': 10},
        'adder': {
            '@type': 'PointwiseAdditionLayer',
            '@connections': ['output']
        },
        'output': {
            '@type': 'FullyConnectedLayer',
            '@connections': [],
            'shape': 10,
            'activation_function': 'softmax'
        }
    }

Conventions
===========

When naming the extra properties of layers, a couple of conventions should be
met. A property name should:

    * be a valid python identifier
    * be lowercase with underscores
    * be ``shape`` if it controls the size of the layer directly
    * be ``activation_function`` if it controls the activation function

