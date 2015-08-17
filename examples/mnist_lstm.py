#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

import brainstorm as bs
from brainstorm.handlers import PyCudaHandler
import numpy as np
import os
import gzip
import pickle
import sys
if sys.version_info < (3,):
    from urllib import urlretrieve
else:
    from urllib.request import urlretrieve

# ------------------------------ Get the data ------------------------------- #

url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
mnist_file = 'mnist.pkl.gz'

if not os.path.exists(mnist_file):
    urlretrieve(url, mnist_file)
with gzip.open(mnist_file, 'rb') as f:
    if sys.version_info < (3,):
        ds = pickle.load(f)
    else:
        ds = pickle.load(f, encoding='latin1')

train_inputs = ds[0][0].T.flatten().reshape((784, 50000, 1))
train_targets = np.zeros((784, 50000, 1))
train_targets[-1, :, :] = ds[0][1].reshape((1, 50000, 1))
train_mask = np.zeros((784, 50000, 1))
train_mask[-1, :, :] = 1

valid_inputs = ds[1][0].T.flatten().reshape((784, 10000, 1))
valid_targets = np.zeros((784, 10000, 1))
valid_targets[-1, :, :] = ds[1][1].reshape((1, 10000, 1))
valid_mask = np.zeros((784, 10000, 1))
valid_mask[-1, :, :] = 1

test_inputs = ds[2][0].T.flatten().reshape((784, 10000, 1))
test_targets = np.zeros((784, 10000, 1))
test_targets[-1, :, :] = ds[2][1].reshape((1, 10000, 1))
test_mask = np.zeros((784, 10000, 1))
test_mask[-1, :, :] = 1

# ---------------------------- Set up Iterators ----------------------------- #

train_getter = bs.Minibatches(batch_size=10, verbose=True, mask=train_mask,
                              default=train_inputs, targets=train_targets)
valid_getter = bs.Minibatches(batch_size=10, verbose=True, mask=valid_mask,
                              default=valid_inputs, targets=valid_targets)
test_getter = bs.Minibatches(batch_size=10, verbose=True, mask=test_mask,
                             default=test_inputs, targets=test_targets)

# ----------------------------- Set up Network ------------------------------ #
inp_layer = bs.InputLayer(out_shapes={'default': ('T', 'B', 1),
                                      'targets': ('T', 'B', 1),
                                      'mask': ('T', 'B', 1)})
out_layer = bs.ClassificationLayer(10, name="out")
mask_layer = bs.MaskLayer()

inp_layer >> \
    bs.LstmLayer(100, name='lstm') >> \
    out_layer - "loss" >> \
    mask_layer >> \
    bs.LossLayer()

inp_layer - 'mask' >> 'mask' - mask_layer

network = bs.Network.from_layer(inp_layer - 'targets' >> 'targets' - out_layer)
network.set_memory_handler(PyCudaHandler())
network.initialize({"default": bs.Gaussian(0.01),
                    "lstm": {'bf': 1}}, seed=42)

# ----------------------------- Set up Trainer ------------------------------ #

trainer = bs.Trainer(bs.SgdStep(learning_rate=0.1), double_buffering=False)
trainer.add_monitor(bs.MaxEpochsSeen(500))
trainer.add_monitor(bs.MonitorAccuracy("valid_getter",
                                       output="out.output", mask_name="mask",
                                       name="validation accuracy",
                                       verbose=True))
trainer.add_monitor(bs.SaveBestWeights("validation accuracy",
                                       name="best weights",
                                       criterion="max"))

# -------------------------------- Train ------------------------------------ #

trainer.train(network, train_getter, valid_getter=valid_getter)
print("\nBest validation accuracy: ", max(trainer.logs["validation accuracy"]))
network.buffer.forward.parameters = trainer.monitors["best weights"].weights
