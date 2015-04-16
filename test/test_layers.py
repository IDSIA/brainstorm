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
from brainstorm.layers.rnn_layer import RnnLayerImpl
import pytest

np.random.seed(1234)

NO_CON = set()
HANDLER = NumpyHandler(np.float64)


def fully_connected_layer():
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
    fully_connected_layer,
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


def set_up_layer(layer, specs):
    layer.set_handler(HANDLER)
    time_steps = specs.get('time_steps', 3)
    batch_size = specs.get('batch_size', 2)

    fwd_buffers, bwd_buffers = setup_buffers(time_steps, batch_size, layer)

    for key, value in fwd_buffers.inputs.items():
        if key in specs:
            print("Using special input:", key)
            HANDLER.set_from_numpy(fwd_buffers.inputs[key], specs[key])

    return fwd_buffers, bwd_buffers


def test_deltas_for_layer(layer_specs):
    layer, specs = layer_specs
    print("\n---------- Testing Deltas for: {} ----------".format(layer.name))
    fwd_buffers, bwd_buffers = set_up_layer(layer, specs)
    eps = specs.get('eps', 1e-5)

    run_deltas_test(layer, fwd_buffers, bwd_buffers, eps,
                    skip_inputs=specs.get('skip_inputs', []),
                    skip_outputs=specs.get('skip_outputs', []))


def test_gradients_for_layer(layer_specs):
    layer, specs = layer_specs
    print("\n-------- Testing Gradients for: {} --------".format(layer.name))
    fwd_buffers, bwd_buffers = set_up_layer(layer, specs)
    eps = specs.get('eps', 1e-5)

    run_gradients_test(layer, fwd_buffers, bwd_buffers, eps,
                       skip_parameters=specs.get('skip_parameters', []),
                       skip_outputs=specs.get('skip_outputs', []))


def test_layer_insensitive_to_internal_state_init(layer_specs):
    layer, specs = layer_specs
    print("\n----- Testing Internal State Insensitivity for: {} -----".format(
        layer.name))
    fwd_buffers, bwd_buffers = set_up_layer(layer, specs)
    eps = specs.get('eps', 1e-8)
    layer.forward_pass(fwd_buffers)

    # get outputs after normal forward pass
    outputs = {}
    for key, value in fwd_buffers.outputs.items():
        outputs[key] = HANDLER.get_numpy_copy(value)

    # randomize internal state
    for key, value in fwd_buffers.internals.items():
        HANDLER.set_from_numpy(value, np.random.randn(*value.shape))

        # compare new output
        layer.forward_pass(fwd_buffers)
        for key, value in fwd_buffers.outputs.items():
            assert np.allclose(outputs[key], value, rtol=eps, atol=eps)


def test_layer_add_to_deltas(layer_specs):
    layer, specs = layer_specs
    print("\n----- Testing Internal State Insensitivity for: {} -----".format(
        layer.name))
    fwd_buffers, bwd_buffers = set_up_layer(layer, specs)
    eps = specs.get('eps', 1e-8)
    for key in bwd_buffers.outputs.keys():
        HANDLER.fill(bwd_buffers.outputs[key], 1.0)

    layer.forward_pass(fwd_buffers)
    layer.backward_pass(fwd_buffers, bwd_buffers)

    # get deltas
    deltas = {}
    for key, value in bwd_buffers.inputs.items():
        deltas[key] = HANDLER.get_numpy_copy(value)

    # clear all bwd buffers except inputs and outputs
    for key, value in bwd_buffers.internals.items():
        HANDLER.fill(value, 0)
    for key, value in bwd_buffers.parameters.items():
        HANDLER.fill(value, 0)

    # do a second backward pass
    layer.backward_pass(fwd_buffers, bwd_buffers)

    # assert all input deltas are doubled
    for key, value in bwd_buffers.inputs.items():
        assert np.allclose(deltas[key] * 2, value, rtol=eps, atol=eps), key