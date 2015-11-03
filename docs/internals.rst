#########
Internals
#########

********
Overview
********

Like many deep learning frameworks, a key philosophy behind Brainstorm is **modularity**.
 * Key abstractions in brainstorm are **layers** that are connected to form a **network** and define which computations should be performed and what their parameters are.
 * Layers are implemented in terms of operations provided by the **handler**, which makes independent from how and on which device the operations are actually implemented (CPU or GPU).
 * The **trainer** coordinates applying an iterative optimization algorithm (**stepper**) with other concerns like monitoring and early stopping (done via so called **hooks**).
 * **data iterators** control the way the data is loaded, augmented, and chunked up.
 * an important design feature is that each of these parts can easily be changed by user-code.
 * So custom layers for example can simply be shared as a single file and imported by others, so there is no need to change brainstorm to support them.

Handler
=======
A :class:`~brainstorm.handlers.base_handler.Handler` implements basic mathematical operations on arrays of a certain type and can allocate memory.
For example, the ``NumpyHandler`` allocates CPU memory and performs computations on the CPU mainly through Numpy.
The ``PyCudaHandler`` on the other hand allocates GPU memory and implements its operations with ``pycuda``.
This intermediate abstraction allows changing the low-level implementation of operations without breaking the layers or any other part of brainstorm.
Our Handler operations have a more restricted feature set than numpy, which makes it easier to implement new handlers, but of course this comes at the price of less convenience in implementing layers.
For the future we plan to provide also a handler that CuDNN4, and one that builds upon NervanaGPU.

Layers
======
A :class:`~brainstorm.layers.base_layer.Layer` defines how to compute a forward pass and a backward pass and which parameters it uses to do so.
Layers don't own any of the memory they use themselves.
The memory layout and managing is done by the network using the ``BufferManager``.
So a layer only needs to specify how it interacts with other layers and how much memory it needs.
This design makes implementing a new layer rather simple, and allows for automated testing of layers.

Note that the ``...LayerImpl`` objects are different from the objects used during network creation with the ``>>`` operator.
The latter are used only to define an architecture, which is then in turn used to instantiate the actual layer objects and create the network.

BufferManager
=============
The BufferManager is part of the Network and is responsible for allocating and structuring the memory needed by all the layers.
It computes and allocates the required total amount of memory based on the current batch-size and sequence-length.
This memory is then chunked-up, reshaped and structured distributed according to the  memory :ref:`_layout` of the network.

Network
=======
The network is a central abstraction that coordinates the layers and the buffer manager, provides an interface for the user to inspect the internals.
It is designed to allow for comprehensive inspection and ease of use.
Each network holds an ordered dictionary of layers (``net.layers``) sorted topologically.
There has to be always exactly one Input layer which is also called ``Input``, and the connections between layers have to form a connected acyclic graph.

The network gives access to the internal memory layout created by the BufferManager through ``net.buffer``.
This interface can be used to inspect(and influence) exactly what is happening inside the network.
Note that if using a GPU handler the returned memory views will be on the device.
If you want to get a copy of any buffer, use the ``net.get(BUFFER_PATH)`` method.

The network provides a powerful initialize method that allows to control precisely how all the parameters are initialized.
Furthermore it keeps track of so called weight modifiers and gradient modifiers.
These simple modifications that will regularly be applied to the weights (e.g. constrain their L2 norm) or the gradient (e.g. clip their values).
Similar to initializers these can be added to individually to individual parameters via the ``net.set_weight_modifiers`` and ``net.set_gradint_modifiers`` methods.

The Network usually operates in three steps:
  1. ``net.provide_external_data()`` is called first to populate the buffers of the Input layer with values for the input data and the targets (and possibly others)
  2. ``net.forward_pass()`` runs the forward pass of all layers populating all the output buffers
  3. ``net.backward_pass()`` performs a backward pass, thus calculating all the deltas and gradients


Trainer
=======
The Trainer is the second central abstraction in Brainstorm which coordinates the training and collects logs.
It provides the Network with data from DataIterators and has its parameters updated by an optimization Stepper.
Furthermore it calls the Hooks, collects their logs and prints them if appropriate.

When ``trainer.train`` is called, the trainer will iterate over the data it gets from the training data iterator and provide it to the network.
It will then call the Stepper which is responsible for updating the parameters based on the gradients.
The trainer also keeps list of Hooks which will be called regularly (on a timescale they can specify) and which can interact with the training.
Some by monitoring certain quantities, some by saving the network or stopping the training if some condition occurs.
Hooks that monitor something report their logs back to the trainer, which prints, collects, and stores them.


Stepper
=======
Is responsible for updating the parameters of a Network.
Can be Stochastic Gradient Descent or RMSProp.

Hooks
=====
 * Interact with the training.
 * Will be called by the trainer based on their timescale and interval.
   ("epoch" or "update", and arbitrary interval)
 * Different stereotypical kinds of Hooks:
   1. Monitors
   2. Savers
   3. Stoppers
   4. Visualizers


Scorers
=======

ValueModifiers
==============

Initializers
============



Let's say we have some data (usually some *input* data and perhaps some *target* data), and we know what mathematical operations we'd like to do on it (compute the outputs, adjust the parameters etc.).
We tell Brainstorm about these operations by building a directed acyclic graph of layers.
Brainstorm then computes the memory that is needed for the required operations, and slices it up into parts needed by each layer creating a **memory layout**.
This memory can now be allocated, and each layer gets access to the parts of the memory relevant for it (where its parameters, inputs, outputs etc. live).

2. A **Buffer Manager** allocates the memory (using the handler), and decides how to prepare the memory layout for efficient processing.
Again, this works independently from how the layers or handlers are implemented.

This means that one can now easily write a new Handler which uses a different array type, and performs basic mathematical operations differently.
The rest of Brainstorm simply works with it.
Similarly, one may chose a different way of allocating memory given a description of layers, without affecting the rest of the components.

Such a design implies that important components of the library can be improved individually and in principle *hot-swapped* as needed.
If a new numerical computations library is available, one can easily integrate it into Brainstorm as long as it satisfies some basic requirement.
If we realize that there is a better way to allocate memory for a certain network connectivity, this can be easily be incorporated.


Here you can find some details about the internal design of brainstorm.
This description is however very much work in progress and by no means
complete.

***********
Conventions
***********

When naming the extra properties of layers, a couple of conventions should be
met. A property name should:

    * be a valid python identifier
    * be in snake_case (lowercase with underscores)
    * be called ``size`` if it controls the size of the layer directly
    * be ``activation`` if it controls the activation function


************
Architecture
************

Network architecture is a dictionary mapping layer names to their properties.
There are two special properties:

  1. ``@type``: a string that specifies the class of the layer
  2. ``@outgoing_connections``: a dictionary mapping the named outputs
     of the layer to a lists of named inputs of other layers it connects to.
     For specifying the input we use dotted notation: ``LAYER_NAME.INPUT_NAME``.
     If the name of the input is ``default`` it can be omitted.

There can be more properties that will be passed on to the layer class when
instantiating them.

A basic example showcasing most features

.. code-block:: python

    architecture = {
        'Input': {
            '@type': 'Input',
            '@outgoing_connections': {
                'default': ['hidden_layer'],
                'targets': ['output_layer.targets']
            },
            'out_shapes': {
                'default': ('T', 'B', 784),
                'targets': ('T', 'B', 1)
            }
        },
        'hidden_layer': {
            '@type': 'FullyConnected',
            '@outgoing_connections': {
                'default': ['output_projection']
            },
            'activation': 'rel',
            'size': 100
        },
        'output_projection': {
            '@type': 'FullyConnected',
            '@outgoing_connections': {
                'default': ['output_layer']
            },
            'activation': 'linear',
            'size': (10,)
        },
        'output_layer': {
            '@type': 'SoftmaxCE'
            '@outgoing_connections': {
                'loss': ['loss_layer']
            },
        },
        'loss_layer': {
            '@outgoing_connections': {},
            '@type': 'Loss',
            'importance': 1.0
        }
    }


.. _layout:

******
Layout
******

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
===============
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
Each node has to have an ``@type`` field that is either ``BufferView`` or
``array``.

View-Nodes
==========
View-nodes will be turned into BufferView objects by the BufferManager.
Each of them is a dictionary and has to contain a  and a ``@index``
entry. The entries not starting with an ``@`` are the child-nodes.
The ``@index`` entry specifies the order among siblings.


A node can also contain a ``@slice`` entry, if the buffers of all child nodes
are of the same type and contiguous in memory. The corresponding array will
then be available as ``_full_buffer`` member in the resulting BufferView object.


Example node:

.. code-block:: python

    {
        '@type': 'BufferView',
        '@index': 0,

        'child_A': {...},
        'child_B': {...}
    }

Another example including the optional ``@slice``:

.. code-block:: python

    {
        '@type': 'BufferView',
        '@index': 2,
        '@slice': (0, 50, 110),

        'only_child': {...}
    }

Array-Nodes
===========
Array-nodes will be turned into arrays (exact type depends on the handler), by
the buffer manager.
Array-Nodes are also dictionaries but they *must have* a ``@slice`` and a
``@shape`` entry, and they cannot have any children.
Like view-nodes, an array-node needs an ``@index`` entry to specify the order among its
siblings.

The ``@slice`` should be a tuple of two integers ``(start, stop)``.
Where ``start`` and ``stop`` specify which slice of the big buffer this array
is a view of points to.

The ``@shape`` entry is a shape-template and describes the dimensionality of
the array.

If an array-node has a shape of a type 2 buffer (time-scaled) it can
(optionally) contain a ``@context_size`` entry. This determines how many extra
time steps are added to the end of that buffer. Notice that this way you can
access the context slices using negative indexing.


Example leaf for a 4 times 5 weight matrix:

.. code-block:: python

    {'@index': 1, '@slice': (5, 25),  '@shape': (4, 5)}

Example leaf for the output of a layer with 10 hidden units:

.. code-block:: python

    {'@index': 1, '@slice': (19, 29), '@shape': ('T', 'B', 10)}


Full Layout Example
===================
We use the following network as an example here:

.. code-block:: python

    mse = MseLayer(10)
    inputs = Input(out_shapes={'input_data': (4,), 'targets':(10,)})
    inputs - 'input_data' >> Rnn(5) >> FullyConnected(10, name='OutLayer') >> 'net_out' - mse
    inputs - 'targets' >> 'targets' - mse
    net = build_net(mse)


.. code-block:: python
.. code-block:: python

    joint_layout = {
        'Input': {
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
        'Rnn': {
            '@type': 'BufferView',
            '@index': 1,
            'inputs': {
                '@type': 'BufferView',
                '@index': 0,
                '@slice': (0, 4),
                'default': {'@type': 'array', '@index': 0, '@slice': (0, 4), '@shape': ('T', 'B', 4), '@context_size':1}
            },
            'outputs': {
                '@type': 'BufferView',
                '@index': 1,
                '@slice': (14, 19),
                'default': {'@type': 'array', '@index': 0, '@slice': (14, 19), '@shape': ('T', 'B', 5), '@context_size':1}
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
                'Ha': {'@type': 'array', '@index': 0, '@slice': (30, 35), '@shape': ('T', 'B', 5), '@context_size':1}
            },
        },
        'Out': {
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
        'Mse': {
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