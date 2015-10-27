###########
Walkthrough
###########
Let's walk through the ``cifar10_cnn.py`` example in the ``examples``
directory, which will quickly explain the basics of using brainstorm.

Prior to running this example, you will need to prepare the CIFAR-10 dataset
for brainstorm. This can be done using the script ``create_cifar10.py`` in the
``data`` directory. To understand how to prepare data for brainstorm, see
:ref:`data_format`.

To start, let's do our imports first:

.. code-block:: python

    from __future__ import division, print_function, unicode_literals

    import os

    import h5py

    import brainstorm as bs
    from brainstorm.data_iterators import Minibatches
    from brainstorm.handlers import PyCudaHandler
    from brainstorm.initializers import Gaussian

We now set the seed for global random number generation in Brainstorm. This
way we can be sure that our experiment is reproducible.

.. code-block:: python

    bs.global_rnd.set_seed(42)

Let's load the dataset from the HDF5 file prepared earlier. A Minibatches
iterator is then set up for both the training and validation sets. We specify
that we want to use a batch size of 100, and that the image data and targets
should be named "default" and "targets" respectively.

.. code-block:: python

    data_dir = os.environ.get('BRAINSTORM_DATA_DIR', '../data')
    data_file = os.path.join(data_dir, 'CIFAR-10.hdf5')
    ds = h5py.File(data_file, 'r')['normalized_split']

    getter_tr = Minibatches(100, default=ds['training']['default'][:], targets=ds['training']['targets'][:])
    getter_va = Minibatches(100, default=ds['validation']['default'][:], targets=ds['validation']['targets'][:])

Now we use a helper tool to easily create two important layers. The first
is an ``Input`` layer which takes external inputs named "default" and "targets"
(these names are used by the tool by default, but can be specified if
different). Every brainstorm layer has a name, and this layer will simply be
named 'Input'.

The second is the final fully-connected layer which produces 10 outputs, and is
given the name 'Output_projection' by the tool by default.

In the background, the tool adds a ``SoftmaxCE`` layer (named 'Output' by
default) which will apply the softmax function and compute the appropriate
cross-entropy loss using the targets. It also wires this loss to a ``Loss``
layer, marking that this is a loss that will be minimized.

.. code-block:: python

    inp, fc = bs.tools.get_in_out_layers('classification', (32, 32, 3), 10)

We can wire up our network using the ``>>`` operator. The layer syntaxes
should be self-explanatory. Any layer connected to other layers can now be
passed to ``from_layer`` to create a new network. Note that we also name each
layer. These names will be used later.

.. code-block:: python

    (inp >>
        bs.layers.Convolution2D(32, kernel_size=(5, 5), padding=2, name='Conv1') >>
        bs.layers.Pooling2D(type="max", kernel_size=(3, 3), stride=(2, 2)) >>
        bs.layers.Convolution2D(32, kernel_size=(5, 5), padding=2, name='Conv2') >>
        bs.layers.Pooling2D(type="max", kernel_size=(3, 3), stride=(2, 2)) >>
        bs.layers.Convolution2D(64, kernel_size=(5, 5), padding=2, name='Conv3') >>
        bs.layers.Pooling2D(type="max", kernel_size=(3, 3), stride=(2, 2)) >>
        bs.layers.FullyConnected(64, name='FC') >>
        fc)

    network = bs.Network.from_layer(fc)

We would like to use CUDA to speed up our network training, so we simply change
set the network's handler to be a ``PyCudaHandler``. This is not needed if we
do not have, or want to use the GPU -- the default handler is ``NumpyHandler``.

.. code-block:: python

    network.set_handler(PyCudaHandler())

We now initialize the weights of our network with a simple dictionary,
using the names of the layers provided earlier. Note that we can use wildcards
here!

We specify that:
- For each layer name beginning with 'Conv', the 'W' parameter should be
initialized using a Gaussian distribution with std. dev. 0.01, and the 'bias'
parameter should be set to all zeros.
- The layers named 'FC' and 'Output_projection' should be initialized
similarly, but using a std. dev. of 0.1 for 'W'.

Note that 'Output_projection' is the default name of the final projection layer
 created by the helper.

.. code-block:: python

    network.initialize({'Conv*': {'W': Gaussian(0.01), 'bias': 0},
                        'FC': {'W': Gaussian(0.1), 'bias': 0},
                        'Output_projection': {'W': Gaussian(0.1), 'bias': 0}})

Next we create the trainer, specifying that we'd like to use SGD with momentum.

To this trainer, we add a **hook** which will produce progress bar during each
epoch.

.. code-block:: python

    trainer = bs.Trainer(bs.training.MomentumStepper(learning_rate=0.01, momentum=0.9))
    trainer.add_hook(bs.hooks.ProgressBar())

We'd like to check the accuracy of the network on our validation set after each
epoch, and there's a hook for that. We inform it that the trainer will
provide access to a data iterator named 'valid_getter' for this.

The layer named 'Output' produces an output named 'probabilities' (the other
output it produces is named 'loss'). We tell the ``Accuracy`` scorer that
this output should be used for computing the accuracy using the dotted
notation ``<layer_name>.<view_type>.<view_name>``.


.. code-block:: python

    scorers = [bs.scorers.Accuracy(out_name='Output.outputs.probabilities')]
    trainer.train_scorers = scorers
    trainer.add_hook(bs.hooks.MonitorScores('valid_getter', scorers, name='validation'))

We'd also like to save the network every time the validation accuracy drops, so
we add a hook for this too. Note that we tell the hook that another hook named
'validation' is logging something called 'Accuracy' and the network should be
saved at any time that its value is at its maximum.

.. code-block:: python

    trainer.add_hook(bs.hooks.SaveBestNetwork('validation.Accuracy',
                                              filename='cifar10_cnn_best.hdf5',
                                              name='best weights',
                                              criterion='max'))

Finally, we add a hook to stop training after 20 epochs.

.. code-block:: python

    trainer.add_hook(bs.hooks.StopAfterEpoch(20))

Now we're ready to train! We provide the trainer with the network to train,
the training data iterator, and the validation data iterator (to be used by the
hook for monitoring the validation accuracy).

.. code-block:: python

    trainer.train(network, getter_tr, valid_getter=getter_va)

All quantities logged by the hooks are collected by the trainer, so
post-training we may examine them.

.. code-block:: python

    print("Best validation accuracy:", max(trainer.logs["validation"]["Accuracy"]))
