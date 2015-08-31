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
    print("Downloading MNIST data ...")
    urlretrieve(url, mnist_file)
print("Extracting data ...")
with gzip.open(mnist_file, 'rb') as f:
    if sys.version_info < (3,):
        ds = pickle.load(f)
    else:
        ds = pickle.load(f, encoding='latin1')
print("Done.")

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

# ----------------------------- Set up Network ------------------------------ #

inp, out = bs.get_in_out_layers_for_classification(1, 10, outlayer_name='out',
                                                   mask_name='mask')
inp >> bs.layers.Lstm(100, name='lstm') >> out
network = bs.Network.from_layer(out)

network.set_memory_handler(PyCudaHandler())
network.initialize({"default": bs.Gaussian(0.01),
                    "lstm": {'bf': 4, 'bi': 4, 'bo': 4}}, seed=42)
network.set_gradient_modifiers({"lstm": bs.ClipValues(low=-1., high=1)})

# ---------------------------- Set up Iterators ----------------------------- #

train_getter = bs.Minibatches(batch_size=100, verbose=True, mask=train_mask,
                              default=train_inputs, targets=train_targets,
                              seed=45252)
valid_getter = bs.Minibatches(batch_size=200, verbose=True, mask=valid_mask,
                              default=valid_inputs, targets=valid_targets)
test_getter = bs.Minibatches(batch_size=200, verbose=True, mask=test_mask,
                             default=test_inputs, targets=test_targets)

# ----------------------------- Set up Trainer ------------------------------ #

trainer = bs.Trainer(bs.SgdStep(learning_rate=0.1), double_buffering=False)
trainer.add_hook(bs.hooks.StopAfterEpoch(500))
trainer.add_hook(bs.hooks.MonitorAccuracy("valid_getter",
                                          output="out.output",
                                          mask_name="mask",
                                          name="validation accuracy",
                                          verbose=True))
trainer.add_hook(bs.hooks.SaveBestNetwork("validation accuracy",
                                          filename='mnist_lstm_best.hdf5',
                                          name="best weights",
                                          criterion="max"))
trainer.add_hook(bs.hooks.MonitorLayerParameters('lstm'))
trainer.add_hook(bs.hooks.MonitorLayerGradients('lstm', timescale='update'))

# -------------------------------- Train ------------------------------------ #

trainer.train(network, train_getter, valid_getter=valid_getter)
print("\nBest validation accuracy: ", max(trainer.logs["validation accuracy"]))
