#!/usr/bin/env python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
from brainstorm.layers.classification_layer import ClassificationLayerImpl
from brainstorm.layers.fully_connected_layer import FullyConnectedLayerImpl
from brainstorm.layers.squared_difference_layer import \
    SquaredDifferenceLayerImpl
from brainstorm.layers.binomial_cross_entropy_layer import \
    BinomialCrossEntropyLayerImpl

from .helpers import run_gradients_test, run_deltas_test, set_up_layer, \
    HANDLER, approx_fprime
import numpy as np
from brainstorm.structure.shapes import ShapeTemplate
from brainstorm.layers.rnn_layer import RnnLayerImpl
from brainstorm.layers.noop_layer import NoOpLayerImpl
from brainstorm.layers.loss_layer import LossLayerImpl
from brainstorm.layers.lstm_layer import LstmLayerImpl
from brainstorm.layers.mask_layer import MaskLayerImpl
from brainstorm.layers.convolution_layer_2d import ConvolutionLayer2DImpl

import pytest

np.random.seed(1234)

NO_CON = set()


def noop_layer(spec):
    in_shapes = {'default': ShapeTemplate('T', 'B', 5)}
    layer = NoOpLayerImpl('NoOpLayer', in_shapes, NO_CON, NO_CON)
    return layer, spec


def loss_layer(spec):
    in_shapes = {'default': ShapeTemplate('T', 'B', 5)}
    layer = LossLayerImpl('LossLayer', in_shapes, NO_CON, NO_CON)
    return layer, spec


def fully_connected_layer(spec):
    in_shapes = {'default': ShapeTemplate('T', 'B', 5)}
    layer = FullyConnectedLayerImpl('FullyConnectedLayer', in_shapes,
                                    NO_CON, NO_CON,
                                    size=4,
                                    activation_function=spec['act_func'])
    return layer, spec


def squared_difference_layer(spec):
    in_shapes = {'inputs_1': ShapeTemplate('T', 'B', 3, 2),
                 'inputs_2': ShapeTemplate('T', 'B', 3, 2)
                 }

    layer = SquaredDifferenceLayerImpl('SquaredDifferenceLayer',
                                       in_shapes, NO_CON, NO_CON)
    return layer, spec


def binomial_crossentropy_layer(spec):
    time_steps = spec.get('time_steps', 3)
    batch_size = spec.get('batch_size', 2)
    size = 5
    shape = (time_steps, batch_size, size)
    default = np.random.rand(*shape)
    targets = np.random.randint(0, 2, shape)
    in_shapes = {'default': ShapeTemplate('T', 'B', size),
                 'targets': ShapeTemplate('T', 'B', size)}

    layer = BinomialCrossEntropyLayerImpl('BinomialCrossEntropyError',
                                          in_shapes, NO_CON, NO_CON)

    spec['default'] = default
    spec['targets'] = targets
    spec['skip_inputs'] = ['targets']
    return layer, spec


def classification_layer(spec):
    time_steps = spec.get('time_steps', 3)
    batch_size = spec.get('batch_size', 2)
    feature_dim = 5
    shape = (time_steps, batch_size, 1)
    targets = np.random.randint(0, feature_dim, shape)
    in_shapes = {'default': ShapeTemplate('T', 'B', feature_dim),
                 'targets': ShapeTemplate('T', 'B', 1)}

    layer = ClassificationLayerImpl('ClassificationLayer', in_shapes, NO_CON,
                                    NO_CON, size=feature_dim)

    spec['skip_inputs'] = ['targets']
    spec['skip_outputs'] = ['output']
    spec['target'] = targets
    return layer, spec


def rnn_layer(spec):
    layer = RnnLayerImpl('RnnLayer',
                         {'default': ShapeTemplate('T', 'B', 5)},
                         NO_CON, NO_CON,
                         size=7,
                         activation_function=spec['act_func'])
    return layer, spec


def lstm_layer(spec):
    layer = LstmLayerImpl('LstmLayer',
                          {'default': ShapeTemplate('T', 'B', 5)},
                          NO_CON, NO_CON,
                          size=7,
                          activation_function=spec['act_func'])
    return layer, spec


def mask_layer(spec):
    layer = MaskLayerImpl('MaskLayer',
                          {'default': ShapeTemplate('T', 'B', 3, 2),
                           'mask': ShapeTemplate('T', 'B', 1)},
                          NO_CON, NO_CON)
    spec['skip_inputs'] = ['mask']
    return layer, spec


def convolution_layer_2d(spec, input_shape=(1, 4, 4),
                         num_filters=1, kernel_size=(2, 2), stride=(1, 1)):
    x = ShapeTemplate('T', 'B', *input_shape)
    layer = ConvolutionLayer2DImpl('ConvolutionLayer2D', {'default': x},
                                   NO_CON, NO_CON, num_filters=num_filters,
                                   kernel_size=kernel_size, stride=stride,
                                   activation_function=spec['act_func'])
    return layer, spec


def convolution_layer_2d_a(spec):
    return convolution_layer_2d(spec, input_shape=(2, 3, 4))


def convolution_layer_2d_b(spec):
    return convolution_layer_2d(spec, input_shape=(2, 3, 4), num_filters=2)


def convolution_layer_2d_c(spec):
    return convolution_layer_2d(spec, input_shape=(2, 3, 4), num_filters=2,
                                kernel_size=(2, 3))

layers_to_test = [
    noop_layer,
    loss_layer,
    fully_connected_layer,
    binomial_crossentropy_layer,
    classification_layer,
    rnn_layer,
    squared_difference_layer,
    lstm_layer,
    mask_layer,
    convolution_layer_2d_a,
    convolution_layer_2d_b,
    convolution_layer_2d_c
]

ids = [f.__name__ for f in layers_to_test]

spec_list = [
    (1, 1, 'tanh'),
    (3, 2, 'tanh'),
    (2, 3, 'sigmoid'),
    (5, 5, 'rel'),
    (1, 4, 'linear')]
spec_ids = ['{}{}{}'.format(*p) for p in spec_list]


@pytest.fixture(params=spec_list, ids=spec_ids)
def spec(request):
    time_steps, batch_size, act_func = request.param
    return {
        'time_steps': time_steps,
        'batch_size': batch_size,
        'act_func': act_func
    }


@pytest.fixture(params=layers_to_test, ids=ids)
def layer_specs(request, spec):
    layer, specs = request.param(spec)
    return layer, specs


def test_deltas_calculation_of_layer(layer_specs):
    layer, specs = layer_specs
    print("\n========= Testing Deltas for: '{}' =========".format(layer.name))

    skip_outputs = specs.get('skip_outputs', [])
    skip_inputs = specs.get('skip_inputs', [])
    successful = True
    for outputs_name in layer.out_shapes:
        if outputs_name in skip_outputs:
            continue
        print("----------- WRT Output: '{}' ----------- ".format(outputs_name))

        for inputs_name in layer.in_shapes:
            if inputs_name in skip_inputs:
                continue
            successful &= run_deltas_test(layer, specs, inputs_name,
                                          outputs_name)

        assert successful, "Deltas check failed for {}".format(layer.name)


def test_gradients_for_layer(layer_specs):
    layer, specs = layer_specs
    print("\n======== Testing Gradients for: '{}' ========".format(layer.name))

    skip_outputs = specs.get('skip_outputs', [])
    skip_parameters = specs.get('skip_parameters', [])
    successful = True
    for outputs_name in layer.out_shapes:
        if outputs_name in skip_outputs:
            continue
        print("----------- WRT Output: '{}' ----------- ".format(outputs_name))

        for inputs_name in layer.get_parameter_structure():
            if inputs_name in skip_parameters:
                continue
            successful &= run_gradients_test(layer, specs, inputs_name,
                                             outputs_name)

        assert successful, "Gradients check failed for {}".format(layer.name)


def test_layer_forward_pass_insensitive_to_internal_state_init(layer_specs):
    layer, specs = layer_specs
    print("\n========= Testing Internal State Insensitivity for: {} ========="
          .format(layer.name))
    layer_buffers = set_up_layer(layer, specs)
    time_steps = specs.get('time_steps', 3)

    eps = specs.get('eps', 1e-8)
    layer.forward_pass(layer_buffers)

    # get outputs after normal forward pass
    outputs = {}
    for key, value in layer_buffers.outputs.items():
        outputs[key] = HANDLER.get_numpy_copy(value)

    # randomize internal state
    for internal, shape_template in layer.get_internal_structure().items():
        value = layer_buffers.internals[internal]
        if shape_template.scales_with_time:
            # exclude context slice
            HANDLER.set_from_numpy(value[:time_steps], np.random.randn(time_steps, *value.shape[1:]))
        else:
            HANDLER.set_from_numpy(value, np.random.randn(*value.shape))

        # compare new output
        layer.forward_pass(layer_buffers)
        for key, value in layer_buffers.outputs.items():
            assert np.allclose(outputs[key], value, rtol=eps, atol=eps), internal


def test_layer_backward_pass_insensitive_to_internal_state_init(layer_specs):
    layer, specs = layer_specs
    print("\n========= Testing Internal State Insensitivity for: {} ========="
          .format(layer.name))
    layer_buffers = set_up_layer(layer, specs)
    time_steps = specs.get('time_steps', 3)
    eps = specs.get('eps', 1e-8)
    layer.forward_pass(layer_buffers)
    layer.backward_pass(layer_buffers)

    # get deltas after normal backward pass
    deltas = {}
    for key, value in layer_buffers.input_deltas.items():
        deltas[key] = HANDLER.get_numpy_copy(value)

    # randomize internal state
    for internal, shape_template in layer.get_internal_structure().items():
        value = layer_buffers.internals[internal]
        if shape_template.scales_with_time:
            # exclude context slice
            HANDLER.set_from_numpy(value[:time_steps], np.random.randn(time_steps, *value.shape[1:]))
        else:
            HANDLER.set_from_numpy(value, np.random.randn(*value.shape))

        # clear deltas
        for k, v in layer_buffers.input_deltas.items():
            HANDLER.fill(v, 0.0)

        # compare new deltas
        layer.forward_pass(layer_buffers)
        layer.backward_pass(layer_buffers)
        for key, value in layer_buffers.input_deltas.items():
            assert np.allclose(deltas[key], value, rtol=eps, atol=eps)


def test_layer_add_to_deltas(layer_specs):
    layer, specs = layer_specs
    print("\n----- Testing Internal State Insensitivity for: {} -----".format(
        layer.name))
    layer_buffers = set_up_layer(layer, specs)
    eps = specs.get('eps', 1e-8)
    for key in layer_buffers.output_deltas.keys():
        HANDLER.fill(layer_buffers.output_deltas[key], 1.0)

    layer.forward_pass(layer_buffers)
    layer.backward_pass(layer_buffers)

    # get deltas
    deltas = {}
    for key, value in layer_buffers.input_deltas.items():
        deltas[key] = HANDLER.get_numpy_copy(value)

    # clear all bwd buffers except inputs and outputs
    for key, value in layer_buffers.internals.items():
        HANDLER.fill(value, 0)
    for key, value in layer_buffers.gradients.items():
        HANDLER.fill(value, 0)
    # set all bwd_buffer inputs to 1.0
    for key, value in layer_buffers.input_deltas.items():
        HANDLER.fill(value, 1.0)

    # do a second backward pass
    layer.backward_pass(layer_buffers)

    # assert all input deltas are 1.0 bigger
    for key, value in layer_buffers.input_deltas.items():
        assert np.allclose(deltas[key] + 1.0, value, rtol=eps, atol=eps), key


def test_elementwise_act_func_gradients():
    pairs_to_test = [(HANDLER.sigmoid, HANDLER.sigmoid_deriv),
                     (HANDLER.tanh, HANDLER.tanh_deriv),
                     (HANDLER.rel, HANDLER.rel_deriv)]

    for fwd, bwd in pairs_to_test:
        print("------------------")
        print("Testing", fwd.__name__)
        inputs = HANDLER.create_from_numpy(np.random.randn(3, 2, 4))
        outputs = HANDLER.create_from_numpy(np.zeros_like(inputs))
        doutputs = HANDLER.create_from_numpy(np.ones_like(inputs))
        dinputs = HANDLER.create_from_numpy(np.zeros_like(inputs))
        fwd(inputs, outputs)
        bwd(inputs, outputs, doutputs, dinputs)
        grad_calc = HANDLER.get_numpy_copy(dinputs)

        size = HANDLER.size(inputs)
        x0 = HANDLER.get_numpy_copy(inputs).reshape((size,))

        def f(x):
            flat_inputs = HANDLER.reshape(inputs, (size,))
            HANDLER.set_from_numpy(flat_inputs, x)
            HANDLER.fill(outputs, 0.)
            fwd(inputs, outputs)
            return HANDLER.get_numpy_copy(outputs).sum()

        grad_approx = approx_fprime(x0, f, 1e-5).reshape(grad_calc.shape)

        assert np.allclose(grad_approx, grad_calc, rtol=1e-4, atol=1e-4)
