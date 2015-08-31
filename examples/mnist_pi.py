#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

import brainstorm as bs
from brainstorm.handlers import PyCudaHandler
import os
import gzip
import pickle
import sys
bs.global_rnd.set_seed(42)
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

# ----------------------------- Set up Network ------------------------------ #

inp, out = bs.get_in_out_layers_for_classification(784, 10,
                                                   outlayer_name='out')
inp >> \
    bs.layers.Dropout(drop_prob=0.2) >> \
    bs.layers.FullyConnected(1200, name='hid1', activation_function='rel') >> \
    bs.layers.Dropout(drop_prob=0.5) >> \
    bs.layers.FullyConnected(1200, name='hid2', activation_function='rel') >> \
    bs.layers.Dropout(drop_prob=0.5) >> \
    out
network = bs.Network.from_layer(out)

network.set_memory_handler(PyCudaHandler())
network.initialize(bs.Gaussian(0.01))
network.set_weight_modifiers({"out": bs.ConstrainL2Norm(1)})

# ---------------------------- Set up Iterators ----------------------------- #

train_getter = bs.Minibatches(batch_size=100, verbose=True,
                              default=train_inputs, targets=train_targets)
valid_getter = bs.Minibatches(batch_size=500, verbose=True,
                              default=valid_inputs, targets=valid_targets)
test_getter = bs.Minibatches(batch_size=500, verbose=True,
                             default=test_inputs, targets=test_targets)

# ----------------------------- Set up Trainer ------------------------------ #

trainer = bs.Trainer(bs.MomentumStep(learning_rate=0.1, momentum=0.9),
                     double_buffering=False)
trainer.add_hook(bs.hooks.StopAfterEpoch(500))
trainer.add_hook(bs.hooks.MonitorAccuracy("valid_getter", "out.output",
                                          name="validation accuracy",
                                          verbose=True))
trainer.add_hook(bs.hooks.SaveBestNetwork("validation accuracy",
                                          filename='mnist_pi_best.hdf5',
                                          name="best weights",
                                          criterion="max"))

# -------------------------------- Train ------------------------------------ #

trainer.train(network, train_getter, valid_getter=valid_getter)
print("\nBest validation accuracy: ", max(trainer.logs["validation accuracy"]))
