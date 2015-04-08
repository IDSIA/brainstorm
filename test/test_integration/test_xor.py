#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np
from brainstorm import (
    InputLayer,  FullyConnectedLayer, LossLayer, BinomialCrossEntropyLayer,
    build_net, Gaussian, Trainer, SgdStep, MaxEpochsSeen, Online,
    SquaredDifferenceLayer)


def test_learn_xor_function():
    # set up the network
    inp = InputLayer(out_shapes={'default': ('T', 'B', 2),
                                 'targets': ('T', 'B', 1)})
    error_func = BinomialCrossEntropyLayer()

    (inp >>
     FullyConnectedLayer(4, activation_function='tanh') >>
     FullyConnectedLayer(1, activation_function='sigmoid') >>
     error_func >>
     LossLayer())

    net = build_net(inp - 'targets' >> 'targets' - error_func)
    net.initialize(Gaussian(0.1))

    # set up the trainer
    tr = Trainer(SgdStep(learning_rate=0.001))
    tr.add_monitor(MaxEpochsSeen(100))

    # generate the data
    data = np.array([
        [0., 0.],
        [0., 1.],
        [1., 0.],
        [1., 1.]
    ]).reshape((1, 4, 2))
    targets = np.array([0., 1., 1., 0.]).reshape((1, 4, 1))

    tr.train(net, Online(default=data, targets=targets))
    assert False
