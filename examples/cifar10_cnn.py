#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from brainstorm.handlers import PyCudaHandler
import brainstorm as bs
from brainstorm.data_iterators import Minibatches
from brainstorm.initializers import Gaussian
import h5py
import os
bs.global_rnd.set_seed(42)

# ----------------------------- Set up Iterators ---------------------------- #

data_dir = os.environ.get('BRAINSTORM_DATA_DIR', '../data')
data_file = os.path.join(data_dir, 'CIFAR-10.hdf5')
ds = h5py.File(data_file, 'r')['normalized_split']

getter_tr = Minibatches(100, default=ds['training']['default'][:],
                        targets=ds['training']['targets'][:])
getter_va = Minibatches(100, default=ds['validation']['default'][:],
                        targets=ds['validation']['targets'][:])

# ------------------------------ Set up Network ----------------------------- #

inp, out = bs.tools.get_in_out_layers_for_classification((3, 32, 32), 10)

(inp >>
    bs.layers.Convolution2D(32, kernel_size=(5, 5), padding=2, name='conv1') >>
    bs.layers.Pooling2D(type="max", kernel_size=(3, 3), stride=(2, 2)) >>
    bs.layers.Convolution2D(32, kernel_size=(5, 5), padding=2, name='conv2') >>
    bs.layers.Pooling2D(type="max", kernel_size=(3, 3), stride=(2, 2)) >>
    bs.layers.Convolution2D(64, kernel_size=(5, 5), padding=2, name='conv3') >>
    bs.layers.Pooling2D(type="max", kernel_size=(3, 3), stride=(2, 2)) >>
    bs.layers.FullyConnected(64, name='fc') >>
    out)

network = bs.Network.from_layer(out)
network.set_memory_handler(PyCudaHandler())
network.initialize({'conv*': {'W': Gaussian(0.01), 'bias': 0},
                    'fc': {'W': Gaussian(0.1), 'bias': 0},
                    'Output': {'W': Gaussian(0.1), 'bias': 0}})

# ------------------------------ Set up Trainer ----------------------------- #

trainer = bs.Trainer(bs.training.MomentumStep(learning_rate=0.001,
                                              momentum=0.9,
                                              scale_learning_rate=False),
                     double_buffering=False)
trainer.add_hook(bs.hooks.StopAfterEpoch(20))
scorers = [bs.scorers.Accuracy(out_name='Output.output')]
trainer.add_hook(bs.hooks.MonitorScores('valid_getter', scorers,
                                        name='validation'))
trainer.add_hook(bs.hooks.SaveBestNetwork("validation.accuracy",
                                          filename='cifar10_cnn_best.hdf5',
                                          name="best weights",
                                          criterion="max"))

# --------------------------------- Train ----------------------------------- #

trainer.train(network, getter_tr, valid_getter=getter_va)
print("\nBest validation accuracy: ",
      max(trainer.logs["validation"]["accuracy"]))
