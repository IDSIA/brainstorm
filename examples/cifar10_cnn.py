#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from brainstorm.handlers import PyCudaHandler
import brainstorm as bs
import h5py
import os
bs.global_rnd.set_seed(42)

# ----------------------------- Set up Iterators ---------------------------- #

data_dir = os.environ.get('BRAINSTORM_DATA_DIR', '.')
data_file = os.path.join(data_dir, 'CIFAR-10.hdf5')
ds = h5py.File(data_file, 'r')['normalized_split']
x_tr, y_tr = ds['training']['default'].value, ds['training']['targets'].value
x_va, y_va = ds['validation']['default'].value, ds['validation']['targets'].value
x_te, y_te = ds['test']['default'].value, ds['test']['targets'].value

getter_tr = bs.Minibatches(100, verbose=True, default=x_tr, targets=y_tr)
getter_va = bs.Minibatches(100, verbose=True, default=x_va, targets=y_va)
getter_te = bs.Minibatches(100, verbose=True, default=x_te, targets=y_te)

# ------------------------------ Set up Network ----------------------------- #

inp, out = bs.get_in_out_layers_for_classification((3, 32, 32), 10)

inp >> \
    bs.layers.Convolution2D(32, kernel_size=(5, 5), padding=2, name='conv1') >> \
    bs.layers.Pooling2D(type="max", kernel_size=(3, 3), stride=(2, 2), name='pool1') >> \
    bs.layers.Convolution2D(32, kernel_size=(5, 5), padding=2, name='conv2') >> \
    bs.layers.Pooling2D(type="max", kernel_size=(3, 3), stride=(2, 2), name='pool2') >> \
    bs.layers.Convolution2D(64, kernel_size=(5, 5), padding=2, name='conv3') >> \
    bs.layers.Pooling2D(type="max", kernel_size=(3, 3), stride=(2, 2), name='pool3') >> \
    bs.layers.FullyConnected(64, name='fc') >> \
    out

network = bs.Network.from_layer(out)
network.set_memory_handler(PyCudaHandler())
network.initialize({'conv*': {'W': bs.Gaussian(0.01), 'bias': 0},
                    'fc': {'W': bs.Gaussian(0.1), 'bias': 0},
                    'Output': {'W': bs.Gaussian(0.1), 'bias': 0},
                    })

# ------------------------------ Set up Trainer ----------------------------- #

trainer = bs.Trainer(bs.MomentumStep(learning_rate=0.001, momentum=0.9,
                                     scale_learning_rate=False),
                     double_buffering=False)
trainer.add_hook(bs.hooks.StopAfterEpoch(20))
trainer.add_hook(bs.hooks.MonitorAccuracy("valid_getter", "Output.output",
                                          name="validation",
                                          verbose=True))
trainer.add_hook(bs.hooks.SaveBestNetwork("validation.accuracy",
                                          filename='cifar10_cnn_best.hdf5',
                                          name="best weights",
                                          criterion="max"))

# --------------------------------- Train ----------------------------------- #

trainer.train(network, getter_tr, valid_getter=getter_va)
print("\nBest validation accuracy: ", max(trainer.logs["validation"]["accuracy"]))
