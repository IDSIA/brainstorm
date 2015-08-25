=======
Network
=======


Buffer Access
=============
We should decide on the API of the buffer management. I thought up 3 variants,
that you could maybe have a look at.

Variant 1
---------
Here ``net.buffer`` would be the BufferManager that keeps a ``forward`` and a
``backward`` BufferView containing all the sub-buffers.

.. code-block:: python

    net.buffer.A.outputs.default   # output of layer A
    net.buffer.B.parameters.W      # weight matrix of layer B

    net.buffer.A.output_deltas.default  # out-deltas of layer A
    net.buffer.B.gradients.W     # gradient matrix for layer B

Variant 2
---------
Almost like variant 1 but with two buffer managers. This will make dotted
access impossible though:

.. code-block:: python

    net.forward_buffer['A'].outputs.default   # output of layer A
    net.forward_buffer['B'].parameters.W      # weight matrix of layer B

    net.backward_buffer['A'].outputs.default  # out-deltas of layer A
    net.backward_buffer['B'].parameters.W     # gradient matrix for layer B


Variant 2
---------
We could move the buffers to the layers, like this. But it would couple the
layers and the BufferManager, and make it harder to access all buffers
together in one common-place:

.. code-block:: python

    net.layers['A'].forward_buffer.outputs.default  # output of layer A
    net.layers['B'].forward_buffer.parameters.W     # weight matrix of layer B

    net.layers['A'].backward_buffer.outputs.default  # out-deltas of layer A
    net.layers['B'].backward_buffer.parameters.W    # gradient matrix for layer B



