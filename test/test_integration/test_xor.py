#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np
import pytest
from brainstorm import Network
from brainstorm.training import Trainer, SgdStep
from brainstorm.initializers import Gaussian
from brainstorm.data_iterators import Undivided
from brainstorm.layers import *
from brainstorm.hooks import StopAfterEpoch
# from brainstorm.handlers.pycuda_handler import PyCudaHandler


@pytest.mark.slow
def test_learn_xor_function():
    # set up the network
    inp = Input(out_shapes={'default': ('T', 'B', 2),
                            'targets': ('T', 'B', 1)})
    error_func = BinomialCrossEntropy()

    (inp >>
     FullyConnected(2, activation_function='sigmoid') >>
     FullyConnected(1, activation_function='sigmoid', name='OutLayer') >>
     error_func >>
     Loss())

    net = Network.from_layer(inp - 'targets' >> 'targets' - error_func)
    # net.set_memory_handler(PyCudaHandler())
    net.initialize(Gaussian(1.0), seed=42)  # high weight-init needed
    print(net.buffer.parameters)

    # set up the trainer
    tr = Trainer(SgdStep(learning_rate=4.0), verbose=False,
                 double_buffering=False)
    tr.add_hook(StopAfterEpoch(300))

    # generate the data
    data = np.array([
        [0., 0.],
        [0., 1.],
        [1., 0.],
        [1., 1.]
    ]).reshape((1, 4, 2))
    targets = np.array([0., 1., 1., 0.]).reshape((1, 4, 1))

    tr.train(net, Undivided(default=data, targets=targets))

    out = net.buffer.OutLayer.outputs.default
    print('Network output:', out.flatten())
    print('Rounded output:', np.round(out.flatten()))
    print('Targets       :', targets.flatten())
    assert np.all(np.round(out) == targets)
    assert min(tr.logs['training']['Loss']) < 0.5
