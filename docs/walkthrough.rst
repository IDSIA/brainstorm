###########
Walkthrough
###########
In this walkthrough we go over the ``cifar10_cnn.py`` example in the ``examples``
directory. This example explains the basics of using brainstorm and helps you get started
with your own project.

Prior to running this example, you will need to prepare the CIFAR-10 dataset
for brainstorm. This can be done by running the ``create_cifar10.py`` script in the
``data`` directory. A detailed description of how to prepare your data for brainstorm can be found in
:ref:`data_format`.

We start the ``cifar10_cnn.py`` example by importing the essential features we will need later:

.. code-block:: python

    from __future__ import division, print_function, unicode_literals

    import os

    import h5py

    import brainstorm as bs
    from brainstorm.data_iterators import Minibatches
    from brainstorm.handlers import PyCudaHandler
    from brainstorm.initializers import Gaussian

Next we set the seed for the global random number generator in Brainstorm. By doing so we are
sure that our experiment is reproducible.

.. code-block:: python

    bs.global_rnd.set_seed(42)

Let's now load the CIFAR-10 dataset from the HDF5 file, which we prepared earlier. Next we create a
``Minibatches`` iterator for the training set and validation set. Here we specify
that we want to use a batch size of 100, and that the image data and targets
should be named 'default' and 'targets' respectively.

.. code-block:: python

    data_dir = os.environ.get('BRAINSTORM_DATA_DIR', '../data')
    data_file = os.path.join(data_dir, 'CIFAR-10.hdf5')
    ds = h5py.File(data_file, 'r')['normalized_split']

    getter_tr = Minibatches(100, default=ds['training']['default'][:], targets=ds['training']['targets'][:])
    getter_va = Minibatches(100, default=ds['validation']['default'][:], targets=ds['validation']['targets'][:])

In the next step we use a simple helper tool to create two important layers. The first layer
is an ``Input`` layer which takes external inputs named 'default' and 'targets'
(these names are the default names used by this tool and can be altered by specifying
different names). Every layer in brainstorm has a name, and by default this layer will simply be
named 'Input'.

The second layer is a fully-connected layer which produces 10 outputs, and is
assigned the name 'Output_projection' by default. In the background, a ``SoftmaxCE`` layer
(named 'Output' by default) is added, which will apply the softmax function and compute the appropriate
cross-entropy loss using the targets. At the same time this loss is wired to a ``Loss``
layer, which marks that this is a value to be minimized.

.. code-block:: python

    inp, fc = bs.tools.get_in_out_layers('classification', (32, 32, 3), 10)

In brainstorm we can wire up our network by using the ``>>`` operator. The layer syntaxes below
should be self-explanatory. Any layer connected to other layers can now be
passed to ``from_layer`` to create a new network. Note that each
layer is assigned a name, which will be used later.

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

We would like to use CUDA to speed up our network training, so we simply
set the network's handler to be the ``PyCudaHandler``. This line is not needed if we
do not have, or do not want to use the GPU -- the default handler is the ``NumpyHandler``.

.. code-block:: python

    network.set_handler(PyCudaHandler())

In the next line we initialize the weights of our network with a simple dictionary, using
the names that were assigned to the layers before. Note that we can use wildcards
here!

We specify that:

- For each layer name beginning with 'Conv', the 'W' parameter should be
  initialized using a Gaussian distribution with std. dev. 0.01, and the 'bias'
  parameter should be set to zero.
- The parameter 'W' of the layers named 'FC' and 'Output_projection' should be
  initialized using a Gaussian distribution with std. dev. 0.1. The 'bias' parameter
  of these layers should be set to zero.

Note that 'Output_projection' is the default name of the final layer created by
the helper over which the softmax is computed.

.. code-block:: python

    network.initialize({'Conv*': {'W': Gaussian(0.01), 'bias': 0},
                        'FC': {'W': Gaussian(0.1), 'bias': 0},
                        'Output_projection': {'W': Gaussian(0.1), 'bias': 0}})

Next we create the trainer for which we specify that we would like to use
stochastic gradient descent (SGD) with momentum.

Additionally we add a **hook** to the trainer, which will produce a progress bar during each
epoch, to keep track of training.

.. code-block:: python

    trainer = bs.Trainer(bs.training.MomentumStepper(learning_rate=0.01, momentum=0.9))
    trainer.add_hook(bs.hooks.ProgressBar())

We would like to check the accuracy of the network on our validation set after each
epoch. In order to do so we will make use of a hook.

The ``SoftmaxCE`` layer named 'Output' produces an output named 'probabilities' (the other
output it produces is named 'loss'). We tell the ``Accuracy`` scorer that
this output should be used for computing the accuracy using the dotted
notation ``<layer_name>.<view_type>.<view_name>``.

Next we set the scorers in the trainer and create a ``MonitorScores`` hook. Here we specify
that the trainer will provide access to a data iterator named 'valid_getter', as well as the
scorers which will make use of this data.

.. code-block:: python

    scorers = [bs.scorers.Accuracy(out_name='Output.outputs.probabilities')]
    trainer.train_scorers = scorers
    trainer.add_hook(bs.hooks.MonitorScores('valid_getter', scorers, name='validation'))

Additionally we would like to save the network every time the validation accuracy improves, so
we add a hook for this too. We tell the hook that another hook named 'validation'
is logging something called 'Accuracy' and that the network should be
saved whenever this value is at its maximum.

.. code-block:: python

    trainer.add_hook(bs.hooks.SaveBestNetwork('validation.Accuracy',
                                              filename='cifar10_cnn_best.hdf5',
                                              name='best weights',
                                              criterion='max'))

Finally, we add a hook to stop training after 20 epochs.

.. code-block:: python

    trainer.add_hook(bs.hooks.StopAfterEpoch(20))

We are now ready to train! We provide the trainer with the network to train,
the training data iterator, and the validation data iterator (to be used by the
hook for monitoring the validation accuracy).

.. code-block:: python

    trainer.train(network, getter_tr, valid_getter=getter_va)

All quantities logged by the hooks are collected by the trainer, which we can examine
post training.

.. code-block:: python

    print("Best validation accuracy:", max(trainer.logs["validation"]["Accuracy"]))
