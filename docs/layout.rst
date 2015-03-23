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

Shape templates
---------------
When implementing a layer there are many places where a shape of a buffer
needs to be specified. But the size of the time-size and the batch-size are
both unknown at implementation time. So we use so called *shape-templates* to
specify which buffer type you are expecting. So for example for feature size of
three these would be the templatesfor the 3 buffer types:

  * ``(3,)`` => Constant size buffer
  * ``('B', 3)`` => Batch sized buffer
  * ``('T', 'B', 3)`` => time sized buffer

Here ``'T'`` is the placeholder for the sequence-length, and ``'B'`` is the
placeholder for the batchsize.

If the feature size is also unknown (e.g. when specifying the input and output
shapes of a layer) then ``'F'`` can be used as a placeholder for those.

The Layout Specification
========================
The layout specification is a tree of nested dictionaries, containing two
types of nodes: view-nodes and array-nodes
that describe what entries the buffer views should have, how big the arrays
at the leaves are, and their position in the big buffer are.

View-Nodes
----------
View-nodes will be turned into BufferView objects by the BufferManager.
Each of them is a dictionary and has to contain a ``layout`` and a ``index``
entry.
The ``layout`` is another dictionary mapping names to child-nodes.
The names of the child-nodes have to be valid python identifiers.
The ``index`` entry specifies the order among siblings.


A node can also contain a ``slice`` entry, if the buffers of all child nodes
are of the same type and contiguous in memory. The corresponding array will
then be available as ``_full_buffer`` member in the resulting BufferView object.


Example node:

.. code-block:: python

    {
        'index': 0,
        'layout': {
            'child_A': {...},
            'child_B': {...}
    }}

Another example including the optional ``slice``:

.. code-block:: python

    {
        'index': 2,
        'slice': (0, 50, 110),
        'layout': {
            'only_child': {...}
        }
    }

Array-Nodes
-----------
Array-Nodes are also dictionaries but instead of a ``layout`` entry they
*must have* a ``slice`` entry.
Note that this means array-nodes are always leafs.
Array-nodes will be turned into arrays (exact type depends on the handler), by
the buffer manager.

The ``slice`` should be a tuple of three integers ``(buffer_type, start, stop)``.
Where ``buffer_type`` in ``[0, 1, 2]`` refers to one of the :ref:`buffer_types`,
and ``start`` and ``stop`` specify which slice of the big buffer this array points to.

Leafs can also contain a ``shape`` entry describing how the feature
dimension of that buffer should be shaped. It defaults to ``(stop-start, )``.

Like nodes, a leaf needs an ``index`` entry to specify the order among its
siblings.

Example leaf for a 4 times 5 weight matrix:

.. code-block:: python

    {'index': 1, 'slice': (0, 5, 25),  'shape': (4, 5)}

Example leaf for the output of a layer with 10 hidden units:

.. code-block:: python

    {'index': 1, 'slice': (2, 19, 29), 'shape': (10,)}


Full Layout Example
-------------------
We use the following network as an example here:

.. code-block:: python

    mse = MseLayer(10)
    inputs = InputLayer(out_shapes={'input_data': (4,), 'targets':(10,)})
    inputs - 'input_data' >> RnnLayer(5) >> FullyConnectedLayer(10, name='OutLayer') >> 'net_out' - mse
    inputs - 'targets' >> 'targets' - mse
    net = build_net(mse)


.. code-block:: python
.. code-block:: python

    joint_layout = {
        'InputLayer': {
            '@type': 'BufferView',
            '@index': 0,
            'inputs': {'@type': 'BufferView', '@index': 0},
            'outputs': {
                '@type': 'BufferView',
                '@index': 1,
                '@slice': (0, 14),
                'input_data': {'@type': 'array', '@index': 0, '@slice': (0, 4), '@shape': ('T', 'B', 4)},
                'targets':    {'@type': 'array','@index': 1, '@slice': (10, 14), '@shape': ('T', 'B', 4)}
            }},
            'parameters': {'@type': 'BufferView', '@index': 2},
            'internals': {'@type': 'BufferView', '@index': 3},
        },
        'RnnLayer': {
            '@type': 'BufferView',
            '@index': 1,
            'inputs': {
                '@type': 'BufferView',
                '@index': 0,
                '@slice': (0, 4),
                'default': {'@type': 'array', '@index': 0, '@slice': (0, 4), '@shape': ('T', 'B', 4)}
            },
            'outputs': {
                '@type': 'BufferView',
                '@index': 1,
                '@slice': (14, 19),
                'default': {'@type': 'array', '@index': 0, '@slice': (14, 19), '@shape': ('T', 'B', 5)}
            },
            'parameters': {
                '@type': 'BufferView',
                '@index': 2,
                '@slice': (0, 50),
                'W': {'@type': 'array', '@index': 0, '@slice': (0, 20),  '@shape': (4, 5)},
                'R': {'@type': 'array', '@index': 1, '@slice': (20, 45), '@shape': (5, 5)},
                'b': {'@type': 'array', '@index': 2, '@slice': (45, 50), '@shape': (5,  )}
            },
            'internals': {
                '@type': 'BufferView',
                '@index': 3,
                '@slice': (30, 35),
                'Ha': {'@type': 'array', '@index': 0, '@slice': (30, 35), '@shape': ('T', 'B', 5)}
            },
        },
        'OutLayer': {
            '@type': 'BufferView',
            '@index': 2,
            'inputs': {
                '@type': 'BufferView',
                '@index': 0,
                '@slice': (14, 19),
                'default': {'@type': 'array', '@index': 0, '@slice': (14, 19), '@shape': ('T', 'B', 5)}
            },
            'outputs': {
                '@type': 'BufferView',
                '@index': 1,
                '@slice': (19, 29),
                'default': {'@type': 'array', '@index': 0, '@slice': (19, 29), '@shape': ('T', 'B', 10)}
            },
            'parameters': {
                '@type': 'BufferView',
                '@index': 2,
                '@slice': (50, 110),
                'W': {'@type': 'array', '@index': 0, '@slice': (50, 100),  '@shape': (5, 10)},
                'b': {'@type': 'array', '@index': 1, '@slice': (100, 110), '@shape': (10,  )}
            },
            'internals': {
                '@type': 'BufferView',
                '@index': 3,
                '@slice': (35, 45),
                'Ha': {'@type': 'array', '@index': 0, '@slice': (35, 55), '@shape': ('T', 'B', 10)}
            }
        },
        'MseLayer': {
            '@type': 'BufferView',
            '@index': 3,
            'inputs': {
                '@type': 'BufferView',
                '@index': 0,
                'net_out': {'@type': 'array', '@index': 0, '@slice': (19, 29), '@shape': ('T', 'B', 10)},
                'targets': {'@type': 'array', '@index': 1, '@slice': (10, 14), '@shape': ('T', 'B', 10)}
            },
            'outputs': {
                '@type': 'BufferView',
                '@index': 1,
                '@slice': (29, 30),
                'default': {'@type': 'array', '@index': 0, '@slice': (29, 30), '@shape': ('T', 'B', 1)}
            },
            'parameters': {'@type': 'BufferView', '@index': 2},
            'internals': {'@type': 'BufferView', '@index': 3},
        }}
    }