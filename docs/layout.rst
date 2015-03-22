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
    DataLayer(4) - 'input_data' >> RnnLayer(5) >> FullyConnectedLayer(10, name='OutLayer') >> 'net_out' - mse
    DataLayer(10) - 'targets' >> 'targets' - mse
    net = build_net(mse)


.. code-block:: python

    joint_layout = {
        'InputLayer': {'index': 0, 'layout': {
            'inputs': {'index': 0, 'layout': {}},
            'outputs': {'index': 1, 'slice': (2, 0, 14), 'layout': {
                'input_data': {'index': 0, 'slice': (2, 0, 4),   'shape': (4,)},
                'targets':    {'index': 1, 'slice': (2, 10, 14), 'shape': (4,)}
            }},
            'parameters': {'index': 2, 'layout': {}},
            'internals': {'index': 3, 'layout': {}},
        }},
        'RnnLayer': {'index': 1, 'layout': {
            'inputs': {'index': 0, 'slice': (2, 0, 4), 'layout': {
                'default': {'index': 0, 'slice': (2, 0, 4), 'shape': (4,)}
            }},
            'outputs': {'index': 1, 'slice': (2, 14, 19), 'layout': {
                'default': {'index': 0, 'slice': (2, 14, 19), 'shape': (5,)}
            }},
            'parameters': {'index': 2, 'slice': (0, 0, 50), 'layout': {
                'W': {'index': 0, 'slice': (0, 0, 20),  'shape': (4, 5)},
                'R': {'index': 1, 'slice': (0, 20, 45), 'shape': (5, 5)},
                'b': {'index': 2, 'slice': (0, 45, 50), 'shape': (5,  )}
            }},
            'internals': {'index': 3, 'slice': (2, 30, 35), 'layout': {
                'Ha': {'index': 0, 'slice': (2, 30, 35), 'shape': (5,)}
            }},
        }},
        'OutLayer': {'index': 2, 'layout': {
            'inputs': {'index': 0, 'slice': (2, 14, 19), 'layout': {
                'default': {'index': 0, 'slice': (2, 14, 19), 'shape': (5,)}
            }},
            'outputs': {'index': 1, 'slice': (2, 19, 29), 'layout': {
                'default': {'index': 0, 'slice': (2, 19, 29), 'shape': (10,)}
            }},
            'parameters': {'index': 2, 'slice': (0, 50, 110), 'layout': {
                'W': {'index': 0, 'slice': (0, 50, 100),  'shape': (5, 10)},
                'b': {'index': 1, 'slice': (0, 100, 110), 'shape': (10,  )}
            }},
            'internals': {'index': 3, 'slice': (2, 35, 45), 'layout': {
                'Ha': {'index': 0, 'slice': (2, 35, 55), 'shape': (10,)}
            }}
        }},
        'MseLayer': {'index': 3, 'layout': {
            'inputs': {'index': 0, 'layout': {
                'net_out': {'index': 0, 'slice': (2, 19, 29), 'shape': (10,)},
                'targets': {'index': 1, 'slice': (2, 10, 14), 'shape': (10,)}
            }},
            'outputs': {'index': 1, 'slice': (2, 29, 30), 'layout': {
                'default': {'index': 0, 'slice': (2, 29, 30), 'shape': (1,)}
            }},
            'parameters': {'index': 2, 'layout': {}},
            'internals': {'index': 3, 'layout': {}},
        }}
    }

    sizes = (45, 0, 110)

Alternative
-----------
Another alternative to consider, is to remove the layout entries and "inline"
them. To distinguish child-nodes from other entries we would need to mark them.
For example with an ``@`` sign like in the architecture description:

.. code-block:: python

    joint_layout = {
        'InputLayer': {
            '@index': 0,
            'inputs': {'@index': 0},
            'outputs': {
                '@index': 1,
                '@slice': (2, 0, 14),
                'input_data': {'@index': 0, '@slice': (2, 0, 4),   '@shape': (4,)},
                'targets':    {'@index': 1, '@slice': (2, 10, 14), '@shape': (4,)}
            }},
            'parameters': {'@index': 2},
            'internals': {'@index': 3},
        },
        'RnnLayer': {
            '@index': 1,
            'inputs': {
                '@index': 0,
                '@slice': (2, 0, 4),
                'default': {'@index': 0, '@slice': (2, 0, 4), '@shape': (4,)}
            },
            'outputs': {
                '@index': 1,
                '@slice': (2, 14, 19),
                'default': {'@index': 0, '@slice': (2, 14, 19), '@shape': (5,)}
            },
            'parameters': {
                '@index': 2,
                '@slice': (0, 0, 50),
                'W': {'@index': 0, '@slice': (0, 0, 20),  '@shape': (4, 5)},
                'R': {'@index': 1, '@slice': (0, 20, 45), '@shape': (5, 5)},
                'b': {'@index': 2, '@slice': (0, 45, 50), '@shape': (5,  )}
            },
            'internals': {
                '@index': 3,
                '@slice': (2, 30, 35),
                'Ha': {'@index': 0, '@slice': (2, 30, 35), '@shape': (5,)}
            },
        },
        'OutLayer': {
            '@index': 2,
            'inputs': {
                '@index': 0,
                '@slice': (2, 14, 19),
                'default': {'@index': 0, '@slice': (2, 14, 19), '@shape': (5,)}
            },
            'outputs': {
                '@index': 1,
                '@slice': (2, 19, 29),
                'default': {'@index': 0, '@slice': (2, 19, 29), '@shape': (10,)}
            },
            'parameters': {
                '@index': 2,
                '@slice': (0, 50, 110),
                'W': {'@index': 0, '@slice': (0, 50, 100),  '@shape': (5, 10)},
                'b': {'@index': 1, '@slice': (0, 100, 110), '@shape': (10,  )}
            },
            'internals': {
                '@index': 3,
                '@slice': (2, 35, 45),
                'Ha': {'@index': 0, '@slice': (2, 35, 55), '@shape': (10,)}
            }
        },
        'MseLayer': {
            '@index': 3,
            'inputs': {
                '@index': 0,
                'net_out': {'@index': 0, '@slice': (2, 19, 29), '@shape': (10,)},
                'targets': {'@index': 1, '@slice': (2, 10, 14), '@shape': (10,)}
            },
            'outputs': {
                '@index': 1,
                '@slice': (2, 29, 30),
                'default': {'@index': 0, '@slice': (2, 29, 30), '@shape': (1,)}
            },
            'parameters': {'@index': 2},
            'internals': {'@index': 3},
        }}
    }