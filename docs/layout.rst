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
The layout specification is a tree of nested dictionaries and lists,
that describe what entries the buffer views should have, how big the arrays
at the leaves are, and their position in the big buffer are.

Nodes
-----
Each node is a dictionary and has to contain a ``name`` entry and
(unless it is a leaf) a ``layout`` entry. The name has to be a valid python
identifier and the ``layout`` is just a list of child-nodes.

A node can also contain a ``slice`` entry, if the buffers of all child nodes
are of the same type and contiguous in memory.

Example node:

.. code-block:: python

    {'name': 'InputLayer', 'layout': [...]}

Another example including the optional ``slice``:

.. code-block:: python

    {'name': 'parameters', 'slice': (0, 50, 110), 'layout': [...]}

Leafs
-----
Every leaf is a node and so has to contain a ``name`` (but not a ``layout``).
In addition to that it *has to have* a ``slice`` entry. The slice should be
a tuple of three integers ``(buffer_type, start, stop)``.
Where ``buffer_type`` in ``[0, 1, 2]`` refers to one of the :ref:`buffer_types`,
and ``start`` and ``stop`` specify which slice of the big buffer this leaf points to.

Leafs can also contain a ``shape`` entry describing how the feature
dimension of that buffer should be shaped. It defaults to ``(stop-start, )``.

Example leaf for a 4 times 5 weight matrix:

.. code-block:: python

    {'name': 'W', 'slice': (0, 5, 25),  'shape': (4, 5)}

Example leaf for the output of a layer with 10 hidden units:

.. code-block:: python

    {'name': 'default', 'slice': (2, 19, 29), 'shape': (10,)}


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
        'sizes': (45, 0, 110),
        'layout': [
            {'name': 'InputLayer', 'layout': [
                {'name': 'outputs', 'slice': (2, 0, 14), 'layout': [
                    ('input_data', {'slice': (2, 0, 4),   'shape': (4,)}),
                    ('targets',    {'slice': (2, 10, 14), 'shape': (4,)})
                ]},
            ]},
            {'name': 'RnnLayer', 'layout': [
                {'name': 'parameters', 'slice': (0, 0, 50), 'layout': [
                    {'name': 'W', 'slice': (0, 0, 20),  'shape': (4, 5)},
                    {'name': 'R', 'slice': (0, 20, 45), 'shape': (5, 5)},
                    {'name': 'b', 'slice': (0, 45, 50), 'shape': (5,  )}
                ]},
                {'name': 'inputs', 'slice': (2, 0, 4), 'layout': [
                    ('default', {'slice': (2, 0, 4), 'shape': (4,)})
                ]},
                {'name': 'outputs', 'slice': (2, 14, 19), 'layout': [
                    ('default', {'tslice': (2, 14, 19), 'shape': (5,)})
                ]},
                {'name': 'state', 'slice': (2, 30, 35), 'layout': [
                    ('Ha', {'slice': (2, 30, 35), 'shape': (5,)})
                ]},
            ]},
            {'name': 'OutLayer', 'layout': [
                {'name': 'parameters', 'slice': (0, 50, 110), 'layout': [
                    ('W', {'slice': (0, 50, 100),  'shape': (5, 10)}),
                    ('b', {'slice': (0, 100, 110), 'shape': (10,  )})
                ]},
                {'name': 'inputs', 'slice': (2, 14, 19), 'layout': [
                    ('default', {'slice': (2, 14, 19), 'shape': (5,)})
                ]},
                {'name': 'outputs', 'slice': (2, 19, 29), 'layout': [
                    ('default', {'slice': (2, 19, 29), 'shape': (10,)})
                ]},
                {'name': 'state', 'slice': (2, 35, 45), 'layout': [
                    ('Ha', {'slice': (2, 35, 55), 'shape': (10,)})
                ]}
            ]},
            {'name': 'MseLayer', 'layout': [
                {'name': 'inputs', 'layout': [
                    ('net_out', {'slice': (2, 19, 29), 'shape': (10,)}),
                    ('targets', {'slice': (2, 10, 14), 'shape': (10,)}),
                ]},
                {'name': 'outputs', 'slice': (2, 29, 30), 'layout': [
                    ('default', {'slice': (2, 29, 30), 'shape': (1,)})
                ]},
            ]}
        ]
    }
