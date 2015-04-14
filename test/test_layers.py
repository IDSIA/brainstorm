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
from .helpers import run_deltas_test, setup_buffers, run_gradients_test
import numpy as np
from brainstorm.structure.shapes import ShapeTemplate
from layers.rnn_layer import RnnLayerImpl
import pytest

np.random.seed(1234)

NO_CON = set()
HANDLER = NumpyHandler(np.float64)


def test_fully_connected_layer():
    in_shapes = {'default': ShapeTemplate('T', 'B', 5)}
    layer = FullyConnectedLayerImpl('FullyConnectedLayer', in_shapes,
                                    NO_CON, NO_CON,
                                    size=4,
                                    activation_function='sigmoid')
    return layer, {}


def squared_difference_layer():
    in_shapes = {'inputs_1': ShapeTemplate('T', 'B', 3, 2),
                 'inputs_2': ShapeTemplate('T', 'B', 3, 2)
                 }

    layer = SquaredDifferenceLayerImpl('SquaredDifferenceLayer',
                                       in_shapes, NO_CON, NO_CON)
    return layer, {}


def binomial_crossentropy_layer():
    time_steps = 3
    batch_size = 2
    size = 5
    shape = (time_steps, batch_size, size)
    default = np.random.rand(*shape)
    targets = np.random.randint(0, 2, shape)
    in_shapes = {'default': ShapeTemplate('T', 'B', size),
                 'targets': ShapeTemplate('T', 'B', size)}

    layer = BinomialCrossEntropyLayerImpl('BinomialCrossEntropyError',
                                          in_shapes, NO_CON, NO_CON)
    return layer, {
        'time_steps': time_steps,
        'batch_size': batch_size,
        'default': default,
        'targets': targets,
        'skip_inputs': ['targets']
    }


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
    layer = RnnLayerImpl('RnnLayer',
                         {'default': ShapeTemplate('T', 'B', 5)},
                         NO_CON, NO_CON,
                         size=7)
    return layer, {}

layers_to_test = [
    binomial_crossentropy_layer,
    classification_layer,
    rnn_layer,
    squared_difference_layer
]

ids = [f.__name__ for f in layers_to_test]


@pytest.fixture(params=layers_to_test, ids=ids)
def layer_specs(request):
    layer, specs = request.param()
    return layer, specs


def test_deltas_for_layer(layer_specs):
    layer, specs = layer_specs
    print("\n---------- Testing Deltas for: {} ----------".format(layer.name))
    layer.set_handler(HANDLER)
    time_steps = specs.get('time_steps', 3)
    batch_size = specs.get('batch_size', 2)
    eps = specs.get('eps', 1e-5)
    fwd_buffers, bwd_buffers = setup_buffers(time_steps, batch_size, layer)

    for key, value in fwd_buffers.inputs.items():
        if key in specs:
            print("Using special input:", key)
            HANDLER.set_from_numpy(fwd_buffers.inputs[key], specs[key])

    run_deltas_test(layer, fwd_buffers, bwd_buffers, eps,
                    skip_inputs=specs.get('skip_inputs', []),
                    skip_outputs=specs.get('skip_outputs', []))


def test_gradients_for_layer(layer_specs):
    layer, specs = layer_specs
    print("\n---------- Testing Gradients for: {} ----------".format(layer.name))
    layer.set_handler(HANDLER)
    time_steps = specs.get('time_steps', 3)
    batch_size = specs.get('batch_size', 2)
    eps = specs.get('eps', 1e-5)
    fwd_buffers, bwd_buffers = setup_buffers(time_steps, batch_size, layer)

    for key, value in fwd_buffers.inputs.items():
        if key in specs:
            print("Using special input:", key)
            HANDLER.set_from_numpy(fwd_buffers.inputs[key], specs[key])

    run_gradients_test(layer, fwd_buffers, bwd_buffers, eps,
                       skip_parameters=specs.get('skip_parameters', []),
                       skip_outputs=specs.get('skip_outputs', []))