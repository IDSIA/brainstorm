#!/usr/bin/env python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
from brainstorm.layers.fully_connected_layer import FullyConnectedLayer
from brainstorm.layers.squared_difference_layer import SquaredDifferenceLayer
from brainstorm.handlers import NumpyHandler
from helpers import run_layer_test
import numpy as np
np.random.seed(1234)


def test_fully_connected_layer():

    eps = 1e-4
    time_steps = 3
    batch_size = 2
    input_shape = 3
    layer_shape = 2

    in_shapes = {'default': ('T', 'B', input_shape,)}
    layer = FullyConnectedLayer('TestLayer', in_shapes, [], [],
                                shape=layer_shape,
                                activation_function='sigmoid')
    layer.set_handler(NumpyHandler(np.float64))
    print("\n---------- Testing FullyConnectedLayer ----------")
    run_layer_test(layer, time_steps, batch_size, eps)


def test_framewise_mse_layer():

    eps = 1e-4
    time_steps = 3
    batch_size = 2
    in_shapes = {'inputs_1': ('T', 'B', 3, 2),
                 'inputs_2': ('T', 'B', 3, 2)
                 }

    layer = SquaredDifferenceLayer('TestLayer', in_shapes, [], [])
    layer.set_handler(NumpyHandler(np.float64))

    print("\n---------- Testing FramewiseMSELayer ----------")
    run_layer_test(layer, time_steps, batch_size, eps)
