.. _data_format:

###########
Data Format
###########

***********
Data Shapes
***********
All data passed to a network in Brainstorm by a data iterator must match
the template ``(T, B, ...)`` where ``T`` is the maximum sequence length and
``B`` is the number of sequences (or batch size, in other words).

To simplify handling both sequential and non-sequential data,
these shapes should also be used when the data is not sequential. In such cases
the shape simply becomes ``(1, B, ...)``. As an example, the MNIST training images
for classification with an MLP should be shaped ``(1, 60000, 784)`` and the
corresponding targets should be shaped ``(1, 60000, 1)``.

Data for images/videos should be stored in the ``TNHWC`` format. For
example, the training images for CIFAR-10 should be shaped
``(1, 50000, 32, 32, 3)`` and the targets should be shaped ``(1, 50000, 1)``.

*******
Example
*******

A network in brainstorm accepts a dictionary of named data items as input.
The keys of this dictionary and the shapes of the data should match those
which were specified when the network was built.

Consider a simple network built as follows:

.. code-block:: python

    import numpy as np
    from brainstorm import Network, layers

    inp = layers.Input({'my_inputs': ('T', 'B', 50),
                        'my_targets': ('T','B', 2)})
    hid = layers.FullyConnected(100, name='Hidden')
    out = layers.SoftmaxCE(name='Output')
    loss = layers.Loss()
    inp - 'my_inputs' >> hid >> out
    inp - 'my_targets' >> 'targets' - out - 'loss' >> loss
    network = Network.from_layer(loss)

The same network can be quickly build


Here's how you can provide some data to a network in brainstorm and run a
forward pass on it.

***********
File Format
***********
There is no requirement on how to store the data in ``brainstorm``, but we
highly recommend the HDF5 format using the h5py library.

It is very simple to create hdf5 files:

.. code-block:: python

    import h5py
    import numpy as np

    with h5py.File('demo.hdf5', 'w') as f:
        f['training/input_data'] = np.random.randn(7, 100, 15)
        f['training/targets'] = np.random.randn(7, 100, 2)
        f['training/static_data'] = np.random.randn(1, 100, 4)

Having such a file available you can then set-up your data iterator like this:

.. code-block:: python

    import h5py
    import brainstorm as bs

    ds = h5py.File('demo.hdf5', 'r')

    online_train_iter = bs.Online(**ds['training'])
    minibatch_train_iter = bs.Minibatches(100, **ds['training'])

These iterators will then provide named data items (a dictionary) to the
network with names 'input_data', 'targets' and 'static_data'.

H5py offers many more features, which can be utilized to improve data
storage and access such as chunking and compression.
