#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

import brainstorm as bs
from brainstorm.handlers import PyCudaHandler, NumpyHandler
import numpy as np
import os
import tarfile
import sys
import six

from six.moves.urllib.request import urlretrieve

# ------------------------------- Get the data ------------------------------ #

url = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
cifar10_file = 'cifar-10-binary.tar.gz'

if not os.path.exists(cifar10_file):
    urlretrieve(url, cifar10_file)
archive_paths = [
    'cifar-10-batches-bin/data_batch_1.bin',
    'cifar-10-batches-bin/data_batch_2.bin',
    'cifar-10-batches-bin/data_batch_3.bin',
    'cifar-10-batches-bin/data_batch_4.bin',
    'cifar-10-batches-bin/data_batch_5.bin',
    'cifar-10-batches-bin/test_batch.bin'
]
with tarfile.open(cifar10_file) as f:
    res = []
    for fn in archive_paths:
        buf = f.extractfile(fn).read()
        tmp = np.fromstring(buf, dtype=np.uint8).reshape(-1, 1024*3+1)
        res.append(tmp)
ds = np.concatenate(res)
# ds[:, 1:] /= 255   # normalize to 0-1 range
num_tr = 40000
x_tr = ds[:num_tr, 1:].reshape((1, num_tr, 3, 32, 32))
y_tr = ds[:num_tr, 0].reshape((1, num_tr, 1))
tr_mean = x_tr.reshape((num_tr * 3, 32, 32)).mean(axis=0)
x_tr = (x_tr - tr_mean)

x_va = (ds[40000:50000, 1:].reshape((1, 10000, 3, 32, 32)) - tr_mean)
y_va = ds[40000:50000, 0].reshape((1, 10000, 1))
x_te = (ds[50000:, 1:].reshape((1, 10000, 3, 32, 32)) - tr_mean)
y_te = ds[50000:, 0].reshape((1, 10000, 1))

# ----------------------------- Set up Iterators ---------------------------- #

getter_tr = bs.Minibatches(100, verbose=True, default=x_tr, targets=y_tr,
                           seed=42)
getter_va = bs.Minibatches(100, verbose=True, default=x_va, targets=y_va,
                           seed=42)
getter_te = bs.Minibatches(100, verbose=True, default=x_te, targets=y_te,
                           seed=42)

# ------------------------------ Set up Network ----------------------------- #

inp = bs.layers.Input(out_shapes={'default': ('T', 'B', 3, 32, 32),
                                  'targets': ('T', 'B', 1)})
out = bs.layers.Classification(10, name="out")
act = 'rel'
# this network is similar to the 'cifar10-quick' example in caffe
inp >> \
    bs.layers.Convolution2D(32, kernel_size=(5, 5), padding=2, name='conv1', activation_function=act) >> \
    bs.layers.Pooling2D(32, type="max", kernel_size=(3, 3), stride=(2, 2), name='pool1') >> \
    bs.layers.Convolution2D(32, kernel_size=(5, 5), padding=2, name='conv2', activation_function=act) >> \
    bs.layers.Pooling2D(32, type="max", kernel_size=(3, 3), stride=(2, 2), name='pool2') >> \
    bs.layers.Convolution2D(64, kernel_size=(5, 5), padding=2, name='conv3',activation_function=act) >> \
    bs.layers.Pooling2D(64, type="max", kernel_size=(3, 3), stride=(2, 2), name='pool3') >> \
    bs.layers.FullyConnected(64, name='fc1', activation_function=act) >> \
    out - "loss" >> bs.layers.Loss()

network = bs.Network.from_layer(inp - 'targets' >> 'targets' - out)
network.set_memory_handler(PyCudaHandler())
network.initialize({'conv1': {'W': bs.Gaussian(0.0001), 'bias': 0},
                    'conv2': {'W': bs.Gaussian(0.01), 'bias': 0},
                    'conv3': {'W': bs.Gaussian(0.01), 'bias': 0},
                    'fc1': {'W': bs.Gaussian(0.1), 'bias': 0},
                    'out': {'W': bs.Gaussian(0.1), 'b': 0},
                    },
                   seed=42)
# network.set_weight_modifiers({"out": bs.ConstrainL2Norm(1)})
# ------------------------------ Set up Trainer ----------------------------- #

trainer = bs.Trainer(bs.SgdStep(learning_rate=0.001), double_buffering=False)
trainer.add_hook(bs.hooks.StopAfterEpoch(10))
trainer.add_hook(bs.hooks.MonitorAccuracy("valid_getter", "out.output",
                                          name="validation accuracy",
                                          verbose=True,
                                          interval=100))
#trainer.add_hook(bs.hooks.SaveBestNetwork("validation accuracy",
#                                          filename='cifar10_cnn_best.hdf5',
#                                          name="best weights",
#                                          criterion="max"))

# --------------------------------- Train ----------------------------------- #

trainer.train(network, getter_tr, valid_getter=getter_va)
print("\nBest validation accuracy: ", max(trainer.logs["validation accuracy"]))
# network.buffer.forward.parameters = trainer.hooks["best weights"].weights
