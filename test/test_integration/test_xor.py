#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np
import pytest
from brainstorm import (
    InputLayer, FullyConnectedLayer, LossLayer, BinomialCrossEntropyLayer,
    build_net, Gaussian, Trainer, SgdStep, MaxEpochsSeen, Undivided)


@pytest.mark.slow
def test_learn_xor_function():
    # set up the network
    inp = InputLayer(out_shapes={'default': ('T', 'B', 2),
                                 'targets': ('T', 'B', 1)})
    error_func = BinomialCrossEntropyLayer()

    (inp >>
     FullyConnectedLayer(2, activation_function='sigmoid') >>
     FullyConnectedLayer(1, activation_function='sigmoid', name='OutLayer') >>
     error_func >>
     LossLayer())

    net = build_net(inp - 'targets' >> 'targets' - error_func)
    net.initialize(Gaussian(1.0), seed=42)  # high weight-init needed
    print(net.buffer.forward.parameters)

    # set up the trainer
    tr = Trainer(SgdStep(learning_rate=4.0), verbose=False)
    tr.add_monitor(MaxEpochsSeen(300))

    # generate the data
    data = np.array([
        [0., 0.],
        [0., 1.],
        [1., 0.],
        [1., 1.]
    ]).reshape((1, 4, 2))
    targets = np.array([0., 1., 1., 0.]).reshape((1, 4, 1))

    tr.train(net, Undivided(default=data, targets=targets))

    out = net.buffer.forward.OutLayer.outputs.default
    print('Network output:', out.flatten())
    print('Rounded output:', np.round(out.flatten()))
    print('Targets       :', targets.flatten())
    assert np.all(np.round(out) == targets)
    assert min(tr.logs['training_errors'][1:]) < 0.5
