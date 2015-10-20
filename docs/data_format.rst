###########
Data Format
###########

*****
Shape
*****
All data passed to the iterators has to match the shape ``(T, B, ...)`` where
``T`` is the maximum sequence length and ``B`` is the number of sequences.
This has to be true even if the data is not sequential. In that case it would
be ``(1, B, ...)``.

***********
File Format
***********
There is no requirement on how to store the data in ``brainstorm``, but we
highly recommend the HDF5 format using the h5py library.

It's amazingly simple to create these files:

.. code-block:: python

    import h5py
    import numpy as np

    with h5py.File('demo.hdf5', 'w') as f:
        f['training/input_data'] = np.random.randn(7, 100, 15)
        f['training/targets'] = np.random.randn(7, 100, 2)
        f['training/non_sequential'] = np.random.randn(1, 100, 4)

And given that file you could use your iterator like this:

.. code-block:: python

    import h5py
    import brainstorm as bs

    ds = h5py.File('demo.hdf5', 'r')

    train_iter = bs.Online(**ds['training'])


.. note::

    H5py offers a lot more features than that. So check them out if you need
    more specialized features.

