#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

import numpy as np
import pytest

from brainstorm import Network, Trainer
from brainstorm.data_iterators import Undivided
from brainstorm.hooks import StopAfterEpoch
from brainstorm.initializers import Gaussian
from brainstorm.layers import *
from brainstorm.training import SgdStepper


@pytest.mark.slow
def test_learn_xor_function():
    # set up the network
    inp = Input(out_shapes={'default': ('T', 'B', 2),
                            'targets': ('T', 'B', 1)})
    error_func = BinomialCrossEntropy()

    (inp >>
     FullyConnected(2, activation='sigmoid') >>
     FullyConnected(1, activation='sigmoid', name='OutLayer') >>
     error_func >>
     Loss())

    net = Network.from_layer(inp - 'targets' >> 'targets' - error_func)
    # net.set_handler(PyCudaHandler())
    net.initialize(Gaussian(1.0), seed=42)  # high weight-init needed
    # print(net.buffer.parameters)

    # set up the trainer
    tr = Trainer(SgdStepper(learning_rate=4.0), verbose=False)
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
    success = np.all(np.round(out) == targets)
    if not success:
        print('Network output:', out.flatten())
        print('Rounded output:', np.round(out.flatten()))
        print('Targets       :', targets.flatten())
        raise AssertionError("Network training did not succeed.")
    assert min(tr.logs['rolling_training']['total_loss']) < 0.5
