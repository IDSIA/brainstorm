#######
Network
#######
Networks are the central structure in brainstorm. They contain and manage a
directed acyclic graph of layers, manage the memory and provide access to all
the internal states.

********
Creating
********
There are essentially 4 different ways of creating a network in brainstorm.

  1. with the ``create_network_from_spec`` tool
  2. using layer wiring in python (with and without helpers)
  3. writing an architecture yourself (advanced)
  4. instantiating the layers by hand and setting up a layout (don't do this)

Setting the Handler
===================
If you want to run on CPU in 32bit mode you don't need to do anything.
For GPU you need to do:

.. code-block:: python

    from brainstorm.handlers import PyCudaHandler
    net.set_handler(PyCudaHandler())


Initializing
============

Just use the :meth:`~brainstorm.structure.network.Network.initialize` method.


Weight and Gradient Modifiers
=============================

``net.set_weight_modifiers()``

``net.set_gradient_modifiers()``

*******
Running
*******
Normally a trainer will run the network for you. But if you want to run a
network yourself you have to do this in order:

  1. ``net.provide_external_data(my_data)``
  2. ``net.forward_pass()``
  3. (optional) ``net.backward_pass()``

*******************
Accessing Internals
*******************

The recommended way is to always use ``net.get(PATH)`` because that returns
a copy of the buffer in numpy format. If you for some reason want to tamper
with the memory that the network is actually using you can get access with:
``net.buffer[PATH]``.

Parameters
==========
  * ``'parameters'`` for an array of all parameters
  * ``'LAYER_NAME.parameters.PARAM_NAME'`` for a specific parameter buffer

For the corresponding derivatives calculated during the backward pass:
  * ``'gradients'``
  * ``'LAYER_NAME.gradients.GRAD_NAME'``

Inputs and Outputs
==================
To access the inputs that have been passed to the network you can use the
shortcut ``net.get_inputs(IN_NAME)``.

To access inputs and outputs of layers use the following paths:

  * ``'LAYER_NAME.inputs.IN_NAME'`` (often IN_NAME = default)
  * ``'LAYER_NAME.outputs.OUT_NAME'`` (often OUT_NAME = default)

For the corresponding derivatives calculated during the backward pass:

  * ``'LAYER_NAME.input_deltas.IN_NAME'``
  * ``'LAYER_NAME.output_deltas.OUT_NAME'``

Internals
=========
Some layers also expose some internal buffers. You can access them with this
path:

  * ``'LAYER_NAME.internals.INTERNAL_NAME'``


******************
Loading and Saving
******************

``net.save_as_hdf5(filename)``

``net = Network.from_hdf5(filename)``
