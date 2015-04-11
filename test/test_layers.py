#!/usr/bin/env python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
from brainstorm.layers.classification_layer import ClassificationLayerImpl
from brainstorm.layers.fully_connected_layer import FullyConnectedLayerImpl
from brainstorm.layers.squared_difference_layer import \
    SquaredDifferenceLayerImpl
from brainstorm.layers.binomial_cross_entropy_layer import \
    BinomialCrossEntropyLayerImpl
from brainstorm.handlers import NumpyHandler
from .helpers import run_layer_test
import numpy as np
from brainstorm.structure.shapes import ShapeTemplate
from layers.rnn_layer import RnnLayerImpl
import pytest

np.random.seed(1234)

NO_CON = set()

def test_fully_connected_layer():

    eps = 1e-6
    time_steps = 3
    batch_size = 2
    input_shape = 3
    layer_shape = 2

    in_shapes = {'default': ShapeTemplate('T', 'B', input_shape,)}
    layer = FullyConnectedLayerImpl('TestLayer', in_shapes, NO_CON, NO_CON,
                                    size=layer_shape,
                                    activation_function='sigmoid')
    layer.set_handler(NumpyHandler(np.float64))
    print("\n---------- Testing FullyConnectedLayer ----------")
    run_layer_test(layer, time_steps, batch_size, eps)


def test_squared_difference_layer():

    eps = 1e-4
    time_steps = 3
    batch_size = 2
    in_shapes = {'inputs_1': ShapeTemplate('T', 'B', 3, 2),
                 'inputs_2': ShapeTemplate('T', 'B', 3, 2)
                 }

    layer = SquaredDifferenceLayerImpl('TestLayer', in_shapes, NO_CON, NO_CON)
    layer.set_handler(NumpyHandler(np.float64))

    print("\n---------- Testing SquaredDifferenceLayer ----------")
    # run_layer_test(layer, time_steps, batch_size, eps)


def test_binomial_crossentropy_layer():
    eps = 1e-5
    time_steps = 3
    batch_size = 2
    feature_shape = (5,)
    shape = (time_steps, batch_size) + feature_shape
    default = np.random.rand(*shape)
    targets = np.random.randint(0, 2, shape)
    print(default)
    print(targets)
    in_shapes = {'default': ShapeTemplate('T', 'B', *feature_shape),
                 'targets': ShapeTemplate('T', 'B', *feature_shape)}

    layer = BinomialCrossEntropyLayerImpl('TestLayer', in_shapes, NO_CON, NO_CON)


    #run_layer_test(layer, time_steps, batch_size, eps,
    #               skip_inputs=['targets'], default=default, targets=targets)


def classification_layer():
    time_steps = 3
    batch_size = 2
    feature_dim = 5
    shape = (time_steps, batch_size, 1)
    targets = np.random.randint(0, feature_dim, shape)
    in_shapes = {'default': ShapeTemplate('T', 'B', feature_dim),
                 'targets': ShapeTemplate('T', 'B', 1)}

    layer = ClassificationLayerImpl('ClassificationLayer', in_shapes, NO_CON,
                                    NO_CON, size=feature_dim)
    return layer, {
        'time_steps': time_steps,
        'batch_size': batch_size,
        'skip_inputs': ['targets'],
        'skip_output': ['output'],
        'targets': targets
    }


def rnn_layer():
    in_shapes = {'default': ShapeTemplate('T', 'B', 5)}
    layer = RnnLayerImpl('RnnLayer', in_shapes, NO_CON, NO_CON,
                         size=7)
    return layer, {}

layers_to_test = [
    classification_layer,
    rnn_layer
]

ids = [f.__name__ for f in layers_to_test]


@pytest.fixture(params=layers_to_test, ids=ids)
def layer_specs(request):
    layer, specs = request.param()
    return layer, specs


def test_layer(layer_specs):
    layer, specs = layer_specs
    print("\n---------- Testing {} ----------".format(layer.name))
    layer.set_handler(NumpyHandler(np.float64))

    run_layer_test(layer, 3, 2, 1e-5, **specs)