======
Layout
======
Layouts describe how the memory for the network should be arranged.

.. _buffer_types:

Buffer Types
============
There are three types of buffers that distinguish the way memory should scale
 with the input dimensions:

  0. **Constant Size:** These buffers do not change their size at all.
     The most common usecase are parameters.

  1. **Batch Sized:** These buffers scale with the number of sequences in the
     current batch. An example would be sequence-wise targets, i.e. there is
     should be one target per sequence.

  2. **Time Sized:** Scale with *both* batch-size and sequence-length.
     Most input data, hidden activations and internal states fall into that
     category.

For all types of buffers we specify their size and shape only for the "feature"
dimension, i.e. the dimension that does not change. So only for constant size
buffer like parameters the specified size and shape will actually be equal to
the final size and shape of the buffer.
For time sized buffer there would be two additional dimension added to the
front of the final buffer shape. So if we specify the inputs should be of
shape ``(28, 28)``, then the input buffer will be of shape ``(T, B, 28, 28)``
where ``B`` is the number of sequences, and ``T`` is their (max) length.

The BufferManager for a network will allocate one big buffer for each type,
and resize them in response to input-sizes. That big chunk of memory is also
split up into a tree of named buffers according to the *layout*.

The Layout Specification
========================
The layout specification is a tree of nested dictionaries,
that describe what entries the buffer views should have, how big the arrays
at the leaves are, and their position in the big buffer are.

Nodes
-----
Each node is a dictionary and has to contain a ``layout`` entry.
The ``layout`` is another dictionary of named child-nodes. The keys in that
dictionary (names of the child-nodes) have to be valid python identifiers.

A node can also contain a ``slice`` entry, if the buffers of all child nodes
are of the same type and contiguous in memory.

Nodes can have ``index`` entries to specify the order among their siblings.
If no index is given the order is assumed to be alphabetical.

Example node:

.. code-block:: python

    {'layout': {
        'child_A': {...},
        'child_B': {...}
    }}

Another example including the optional ``slice``:

.. code-block:: python

    {
        'slice': (0, 50, 110),
        'layout': {
            'only_child': {...}
        }
    }

Leafs
-----
Leafs are also dictionaries but instead of a ``layout`` entry they
*must have* a ``slice`` entry.
The slice should be a tuple of three integers ``(buffer_type, start, stop)``.
Where ``buffer_type`` in ``[0, 1, 2]`` refers to one of the :ref:`buffer_types`,
and ``start`` and ``stop`` specify which slice of the big buffer this leaf points to.

Leafs can also contain a ``shape`` entry describing how the feature
dimension of that buffer should be shaped. It defaults to ``(stop-start, )``.

Like nodes, a leaf can have an ``index`` entry to specify the order among its
siblings.

Example leaf for a 4 times 5 weight matrix:

.. code-block:: python

    {'slice': (0, 5, 25),  'shape': (4, 5)}

Example leaf for the output of a layer with 10 hidden units:

.. code-block:: python

    {'slice': (2, 19, 29), 'shape': (10,)}


Full Layout Example
-------------------
We use the following network as an example here:

.. code-block:: python

    mse = MseLayer(10)
    DataLayer(4) - 'input_data' >> RnnLayer(5) >> FullyConnectedLayer(10, name='OutLayer') >> 'net_out' - mse
    DataLayer(10) - 'targets' >> 'targets' - mse
    net = build_net(mse)


.. code-block:: python

    joint_layout = {
        'InputLayer': {'layout': {
            'outputs': {'slice': (2, 0, 14), 'layout': {
                'input_data': {'index': 0, 'slice': (2, 0, 4),   'shape': (4,)},
                'targets':    {'index': 1, 'slice': (2, 10, 14), 'shape': (4,)}
            }},
        }},
        'RnnLayer': {'layout': {
            'parameters': {'slice': (0, 0, 50), 'layout': {
                'W': {'index': 0, 'slice': (0, 0, 20),  'shape': (4, 5)},
                'R': {'index': 1, 'slice': (0, 20, 45), 'shape': (5, 5)},
                'b': {'index': 2, 'slice': (0, 45, 50), 'shape': (5,  )}
            }},
            'inputs': {'slice': (2, 0, 4), 'layout': {
                'default': {'slice': (2, 0, 4), 'shape': (4,)}
            }},
            'outputs': {'slice': (2, 14, 19), 'layout': {
                'default': {'slice': (2, 14, 19), 'shape': (5,)}
            }},
            'internal': {'slice': (2, 30, 35), 'layout': {
                'Ha': {'slice': (2, 30, 35), 'shape': (5,)}
            }},
        }},
        'OutLayer': {'layout': {
            'parameters': {'slice': (0, 50, 110), 'layout': {
                'W': {'index': 0, 'slice': (0, 50, 100),  'shape': (5, 10)},
                'b': {'index': 1, 'slice': (0, 100, 110), 'shape': (10,  )}
            }},
            'inputs': {'slice': (2, 14, 19), 'layout': {
                'default': {'slice': (2, 14, 19), 'shape': (5,)}
            }},
            'outputs': {'slice': (2, 19, 29), 'layout': {
                'default': {'slice': (2, 19, 29), 'shape': (10,)}
            }},
            'internal': {'slice': (2, 35, 45), 'layout': {
                'Ha': {'slice': (2, 35, 55), 'shape': (10,)}
            }}
        }},
        'MseLayer': {'layout': {
            'inputs': {'layout': {
                'net_out': {'index': 0, 'slice': (2, 19, 29), 'shape': (10,)},
                'targets': {'index': 1, 'slice': (2, 10, 14), 'shape': (10,)}
            }},
            'outputs': {'slice': (2, 29, 30), 'layout': {
                'default': {'slice': (2, 29, 30), 'shape': (1,)}
            }},
        }}
    }

    sizes = (45, 0, 110)