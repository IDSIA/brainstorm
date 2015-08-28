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
ds[:, 1:] /= 255   # normalize to 0-1 range

x_tr = ds[:40000, 1:].reshape((1, 40000, 3, 32, 32))
y_tr = ds[:40000, 0].reshape((1, 40000, 1))
x_va = ds[40000:50000, 1:].reshape((1, 10000, 3, 32, 32))
y_va = ds[40000:50000, 0].reshape((1, 10000, 1))
x_te = ds[50000:, 1:].reshape((1, 10000, 3, 32, 32))
y_te = ds[50000:, 0].reshape((1, 10000, 1))

# ----------------------------- Set up Iterators ---------------------------- #

getter_tr = bs.Minibatches(64, verbose=True, default=x_tr, targets=y_tr)
getter_va = bs.Minibatches(512, verbose=True, default=x_va, targets=y_va)
getter_te = bs.Minibatches(512, verbose=True, default=x_te, targets=y_te)

# ------------------------------ Set up Network ----------------------------- #

inp = bs.layers.Input(out_shapes={'default': ('T', 'B', 3, 32, 32),
                                'targets': ('T', 'B', 1)})
out = bs.layers.Classification(10, name="out")

# this network is similar to the 'cifar10-quick' example in caffe
inp >> \
    bs.layers.Convolution2D(32, kernel_size=(5, 5), padding=2, name='conv1', activation_function='rel') >> \
    bs.layers.Pooling2D(32, kernel_size=(3, 3), stride=(2, 2), name='pool1') >> \
    bs.layers.Convolution2D(32, kernel_size=(5, 5), padding=2, name='conv2', activation_function='rel') >> \
    bs.layers.Pooling2D(32, kernel_size=(3, 3), stride=(2, 2), name='pool2') >> \
    bs.layers.Convolution2D(64, kernel_size=(5, 5), padding=2, name='conv3', activation_function='rel') >> \
    bs.layers.Pooling2D(64, kernel_size=(3, 3), stride=(2, 2), name='pool3') >> \
    bs.layers.FullyConnected(64, name='fc1', activation_function='rel') >> \
    out - "loss" >> bs.layers.Loss()

network = bs.Network.from_layer(inp - 'targets' >> 'targets' - out)
network.set_memory_handler(PyCudaHandler())
network.initialize(bs.Gaussian(0.001), seed=42)
network.set_weight_modifiers({"out": bs.ConstrainL2Norm(1)})

# ------------------------------ Set up Trainer ----------------------------- #

trainer = bs.Trainer(bs.SgdStep(learning_rate=0.01), double_buffering=False)
trainer.add_hook(bs.hooks.StopAfterEpoch(100))
trainer.add_hook(bs.hooks.MonitorAccuracy("valid_getter", "out.output",
                                          name="validation accuracy",
                                          verbose=True))
#trainer.add_hook(bs.hooks.SaveBestNetwork("validation accuracy",
#                                          filename='cifar10_cnn_best.hdf5',
#                                          name="best weights",
#                                          criterion="max"))

# --------------------------------- Train ----------------------------------- #

trainer.train(network, getter_tr, valid_getter=getter_va)
print("\nBest validation accuracy: ", max(trainer.logs["validation accuracy"]))
network.buffer.forward.parameters = trainer.hooks["best weights"].weights
