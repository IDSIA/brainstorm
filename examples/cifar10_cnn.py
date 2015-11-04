#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

import os

import h5py

import brainstorm as bs
from brainstorm.data_iterators import Minibatches
from brainstorm.handlers import PyCudaHandler
from brainstorm.initializers import Gaussian

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

inp, fc = bs.tools.get_in_out_layers('classification', (32, 32, 3), 10)

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
# Uncomment next line to use GPU
# network.set_handler(PyCudaHandler())
network.initialize({'Conv*': {'W': Gaussian(0.01), 'bias': 0},
                    'FC': {'W': Gaussian(0.1), 'bias': 0},
                    'Output_projection': {'W': Gaussian(0.1), 'bias': 0}})

# ------------------------------ Set up Trainer ----------------------------- #

trainer = bs.Trainer(bs.training.MomentumStepper(learning_rate=0.01, momentum=0.9))
trainer.add_hook(bs.hooks.ProgressBar())
scorers = [bs.scorers.Accuracy(out_name='Output.outputs.predictions')]
trainer.train_scorers = scorers
trainer.add_hook(bs.hooks.MonitorScores('valid_getter', scorers,
                                        name='validation'))
trainer.add_hook(bs.hooks.SaveBestNetwork('validation.Accuracy',
                                          filename='cifar10_cnn_best.hdf5',
                                          name='best weights',
                                          criterion='max'))
trainer.add_hook(bs.hooks.StopAfterEpoch(20))

# --------------------------------- Train ----------------------------------- #

trainer.train(network, getter_tr, valid_getter=getter_va)
print("Best validation accuracy:", max(trainer.logs["validation"]["Accuracy"]))
