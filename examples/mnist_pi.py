#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

import brainstorm as bs
from brainstorm.handlers import PyCudaHandler
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

train_inputs, train_targets = \
    ds[0][0].reshape((1, 50000, 784)), ds[0][1].reshape((1, 50000, 1))
valid_inputs, valid_targets = \
    ds[1][0].reshape((1, 10000, 784)), ds[1][1].reshape((1, 10000, 1))
test_inputs, test_targets = \
    ds[2][0].reshape((1, 10000, 784)), ds[2][1].reshape((1, 10000, 1))

# ---------------------------- Set up Iterators ----------------------------- #

train_getter = bs.Minibatches(batch_size=100, verbose=True, seed=42,
                              default=train_inputs, targets=train_targets)
valid_getter = bs.Minibatches(batch_size=500, verbose=True,
                              default=valid_inputs, targets=valid_targets)
test_getter = bs.Minibatches(batch_size=500, verbose=True,
                             default=test_inputs, targets=test_targets)

# ----------------------------- Set up Network ------------------------------ #

inp = bs.layers.Input(out_shapes={'default': ('T', 'B', 784),
                                  'targets': ('T', 'B', 1)})
out = bs.layers.Classification(10, name="out")

inp >> \
    bs.layers.FullyConnected(1000, name='hid1', activation_function='rel') >> \
    bs.layers.FullyConnected(1000, name='hid2', activation_function='rel') >> \
    out - "loss" >> bs.layers.Loss()

network = bs.Network.from_layer(inp - 'targets' >> 'targets' - out)
network.set_memory_handler(PyCudaHandler())
network.initialize(bs.Gaussian(0.01), seed=42)
network.set_weight_modifiers({"out": bs.ConstrainL2Norm(1)})

# ----------------------------- Set up Trainer ------------------------------ #

trainer = bs.Trainer(bs.SgdStep(learning_rate=0.1), double_buffering=False)
trainer.add_hook(bs.hooks.StopAfterEpoch(100))
trainer.add_hook(bs.hooks.MonitorAccuracy("valid_getter", "out.output",
                                          name="validation accuracy",
                                          verbose=True))
trainer.add_hook(bs.hooks.SaveBestNetwork("validation accuracy",
                                          name="best weights",
                                          criterion="max"))

# -------------------------------- Train ------------------------------------ #

trainer.train(network, train_getter, valid_getter=valid_getter)
print("\nBest validation accuracy: ", max(trainer.logs["validation accuracy"]))
network.buffer.parameters = trainer.hooks["best weights"].weights
