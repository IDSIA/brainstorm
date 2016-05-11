#!/usr/bin/env python
# coding=utf-8

from __future__ import division, print_function, unicode_literals

import numpy as np
import pytest

from brainstorm.layers.base_layer import get_layer_class_from_typename
from brainstorm.layers.batch_normalization_layer import BatchNormLayerImpl
from brainstorm.layers.binomial_cross_entropy_layer import \
    BinomialCrossEntropyLayerImpl
from brainstorm.layers.softmax_ce_layer import SoftmaxCELayerImpl
from brainstorm.layers.sigmoid_ce_layer import SigmoidCELayerImpl
from brainstorm.layers.convolution_layer_2d import Convolution2DLayerImpl
from brainstorm.layers.elementwise_layer import ElementwiseLayerImpl
from brainstorm.layers.fully_connected_layer import FullyConnectedLayerImpl
from brainstorm.layers.highway_layer import HighwayLayerImpl
from brainstorm.layers.input_layer import InputLayerImpl
from brainstorm.layers.l1_decay import L1DecayLayerImpl
from brainstorm.layers.l2_decay import L2DecayLayerImpl
from brainstorm.layers.loss_layer import LossLayerImpl
from brainstorm.layers.lstm_layer import LstmLayerImpl
from brainstorm.layers.mask_layer import MaskLayerImpl
from brainstorm.layers.noop_layer import NoOpLayerImpl
from brainstorm.layers.pooling_layer_2d import Pooling2DLayerImpl
from brainstorm.layers.recurrent_layer import RecurrentLayerImpl
from brainstorm.layers.squared_error_layer import SquaredErrorLayerImpl
from brainstorm.layers.squared_difference_layer import \
    SquaredDifferenceLayerImpl
from brainstorm.structure.architecture import Connection
from brainstorm.structure.buffer_structure import BufferStructure
from brainstorm.utils import LayerValidationError
from brainstorm.layers.clockwork_layer import ClockworkLayerImpl
from brainstorm.layers.clockwork_lstm_layer import ClockworkLstmLayerImpl
from brainstorm.layers.merge_layer import MergeLayerImpl

from brainstorm.tests.helpers import (HANDLER, approx_fprime, run_deltas_test,
                                      run_gradients_test, set_up_layer)
from layers.softmax_fiddle_layer import SoftmaxFiddleLayerImpl

np.random.seed(1234)

NO_CON = set()


def noop_layer(spec):
    in_shapes = {'default': BufferStructure('T', 'B', 5)}
    layer = NoOpLayerImpl('NoOpLayer', in_shapes, NO_CON, NO_CON)
    return layer, spec


def loss_layer(spec):
    in_shapes = {'default': BufferStructure('T', 'B', 5)}
    layer = LossLayerImpl('LossLayer', in_shapes, NO_CON, NO_CON)
    return layer, spec


def fully_connected_layer(spec):
    in_shapes = {'default': BufferStructure('T', 'B', 5)}
    layer = FullyConnectedLayerImpl('FullyConnectedLayer', in_shapes,
                                    NO_CON, NO_CON,
                                    size=4,
                                    activation=spec['activation'])
    return layer, spec


def fully_connected_layer_2d(spec):
    in_shapes = {'default': BufferStructure('T', 'B', 2, 3)}
    layer = FullyConnectedLayerImpl('FullyConnectedLayer', in_shapes,
                                    NO_CON, NO_CON,
                                    size=(3, 3, 1),
                                    activation=spec['activation'])
    return layer, spec


def highway_layer(spec):
    in_shapes = {'H': BufferStructure('T', 'B', 2, 3),
                 'T': BufferStructure('T', 'B', 2, 3),
                 'x': BufferStructure('T', 'B', 2, 3)}
    layer = HighwayLayerImpl('HighwayLayer', in_shapes, NO_CON, NO_CON)
    return layer, spec


def squared_difference_layer(spec):
    in_shapes = {'inputs_1': BufferStructure('T', 'B', 3, 2),
                 'inputs_2': BufferStructure('T', 'B', 3, 2)
                 }

    layer = SquaredDifferenceLayerImpl('SquaredDifferenceLayer',
                                       in_shapes, NO_CON, NO_CON)
    return layer, spec


def squared_error_layer(spec):
    in_shapes = {'default': BufferStructure('T', 'B', 3, 2),
                 'targets': BufferStructure('T', 'B', 3, 2)
                 }

    layer = SquaredErrorLayerImpl('SquaredErrorLayer',
                                  in_shapes, NO_CON, NO_CON)
    return layer, spec


def binomial_crossentropy_layer(spec):
    time_steps = spec.get('time_steps', 3)
    batch_size = spec.get('batch_size', 2)
    size = 5
    shape = (time_steps, batch_size, size)
    default = np.random.rand(*shape)
    targets = np.random.randint(0, 2, shape)
    in_shapes = {'default': BufferStructure('T', 'B', size),
                 'targets': BufferStructure('T', 'B', size)}

    layer = BinomialCrossEntropyLayerImpl('BinomialCrossEntropyError',
                                          in_shapes, NO_CON, NO_CON)

    spec['default'] = default
    spec['targets'] = targets
    return layer, spec


def softmax_ce_layer(spec):
    time_steps = spec.get('time_steps', 3)
    batch_size = spec.get('batch_size', 2)
    feature_dim = (2, 3, 5)
    target_shape = (time_steps, batch_size, 2, 3, 1)
    targets = np.random.randint(0, feature_dim[-1], target_shape)
    in_shapes = {'default': BufferStructure('T', 'B', *feature_dim),
                 'targets': BufferStructure('T', 'B', *target_shape[2:])}

    layer = SoftmaxCELayerImpl('SoftmaxCELayer', in_shapes, NO_CON,
                               NO_CON)

    spec['targets'] = targets
    return layer, spec


def sigmoid_ce_layer(spec):
    time_steps = spec.get('time_steps', 3)
    batch_size = spec.get('batch_size', 2)
    feature_dim = (2, 3, 5)
    target_shape = (time_steps, batch_size) + feature_dim
    targets = np.random.randint(0, 2, target_shape)
    in_shapes = {'default': BufferStructure('T', 'B', *feature_dim),
                 'targets': BufferStructure('T', 'B', *target_shape[2:])}

    layer = SigmoidCELayerImpl('SigmoidCELayer', in_shapes, NO_CON,
                               NO_CON)

    spec['targets'] = targets
    return layer, spec


def softmax_fiddle_layer(spec):
    time_steps = spec.get('time_steps', 3)
    batch_size = spec.get('batch_size', 2)
    feature_dim = (4,)
    target_shape = (time_steps, batch_size) + feature_dim
    targets = np.random.randint(0, 2, target_shape).astype(np.float)
    targets /= np.clip(targets.sum(2)[:, :, None], 1, 10000)
    print('TARGETS:', targets)
    in_shapes = {'default': BufferStructure('T', 'B', *feature_dim),
                 'targets': BufferStructure('T', 'B', *target_shape[2:])}

    layer = SoftmaxFiddleLayerImpl('SoftmaxFiddleLayer', in_shapes, NO_CON,
                                    NO_CON)

    spec['targets'] = targets
    return layer, spec


def rnn_layer(spec):
    layer = RecurrentLayerImpl('RnnLayer',
                               {'default': BufferStructure('T', 'B', 3)},
                               NO_CON, NO_CON,
                               size=4,
                               activation=spec['activation'])
    return layer, spec


def rnn_layer_2d(spec):
    layer = RecurrentLayerImpl('RnnLayer',
                               {'default': BufferStructure('T', 'B', 2, 1, 2)},
                               NO_CON, NO_CON,
                               size=3,
                               activation=spec['activation'])
    return layer, spec


def lstm_layer(spec):
    layer = LstmLayerImpl('LstmLayer',
                          {'default': BufferStructure('T', 'B', 3)},
                          NO_CON, NO_CON,
                          size=4,
                          activation=spec['activation'])
    return layer, spec


def lstm_layer_2d(spec):
    layer = LstmLayerImpl('LstmLayer',
                          {'default': BufferStructure('T', 'B', 2, 2, 1)},
                          NO_CON, NO_CON,
                          size=3,
                          activation=spec['activation'])
    return layer, spec


def mask_layer(spec):
    layer = MaskLayerImpl('MaskLayer',
                          {'default': BufferStructure('T', 'B', 3, 2),
                           'mask': BufferStructure('T', 'B', 1)},
                          NO_CON, NO_CON)
    return layer, spec


def convolution_layer_2d(spec, input_shape=(4, 4, 1),
                         num_filters=1, kernel_size=(2, 2), stride=(1, 1)):
    x = BufferStructure('T', 'B', *input_shape)
    layer = Convolution2DLayerImpl('Convolution2DLayer', {'default': x},
                                   NO_CON, NO_CON, num_filters=num_filters,
                                   kernel_size=kernel_size, stride=stride,
                                   activation=spec['activation'])
    return layer, spec


def convolution_layer_2d_a(spec):
    return convolution_layer_2d(spec, input_shape=(3, 4, 2))


def convolution_layer_2d_b(spec):
    return convolution_layer_2d(spec, input_shape=(3, 4, 2), num_filters=2)


def convolution_layer_2d_c(spec):
    return convolution_layer_2d(spec, input_shape=(3, 4, 2), num_filters=2,
                                kernel_size=(2, 3))


def maxpooling_layer_2d(spec):
    layer = Pooling2DLayerImpl('Pooling2DLayer',
                               {'default':
                                BufferStructure('T', 'B', 4, 4, 1)},
                               NO_CON, NO_CON,
                               kernel_size=(2, 2), stride=(1, 1),
                               type="max")
    return layer, spec


def avgpooling_layer_2d(spec):
    layer = Pooling2DLayerImpl('Pooling2DLayer',
                               {'default':
                                BufferStructure('T', 'B', 4, 4, 1)},
                               NO_CON, NO_CON,
                               kernel_size=(2, 2), stride=(1, 1),
                               type="avg")
    return layer, spec


def batch_norm_layer_fc(spec):
    layer = BatchNormLayerImpl('BatchNorm',
                               {'default': BufferStructure('T', 'B', 3)},
                               NO_CON, NO_CON)
    return layer, spec


def batch_norm_layer_nhwc(spec):
    layer = BatchNormLayerImpl('BatchNorm',
                               {'default': BufferStructure('T', 'B', 3, 2, 4)},
                               NO_CON, NO_CON)
    return layer, spec


def elementwise_layer(spec):
    layer = ElementwiseLayerImpl('Elementwise',
                                 {'default': BufferStructure('T', 'B', 3, 2)},
                                 NO_CON, NO_CON,
                                 activation=spec['activation'])
    return layer, spec


def l1_decay_layer(spec):
    layer = L1DecayLayerImpl('L1Decay',
                             {'default': BufferStructure('T', 'B', 3, 2)},
                             NO_CON, NO_CON)
    return layer, spec


def l2_decay_layer(spec):
    layer = L2DecayLayerImpl('L2Decay',
                             {'default': BufferStructure('T', 'B', 3, 2)},
                             NO_CON, NO_CON)
    return layer, spec


def clockwork_layer(spec):
    layer = ClockworkLayerImpl('ClockworkRnn',
                               {'default': BufferStructure('T', 'B', 3)},
                               NO_CON, NO_CON,
                               size=7,
                               activation=spec['activation'])
    spec['inits'] = {'timing': np.array([1, 1, 2, 2, 3, 3, 5])}
    return layer, spec


def clockwork_layer_2d(spec):
    layer = ClockworkLayerImpl('ClockworkRnn',
                               {'default': BufferStructure('T', 'B', 2, 1, 2)},
                               NO_CON, NO_CON,
                               size=7,
                               activation=spec['activation'])
    spec['inits'] = {'timing': np.array([1, 1, 2, 2, 3, 3, 5])}
    return layer, spec


def clockwork_lstm_layer(spec):
    layer = ClockworkLstmLayerImpl('ClockworkLstm',
                                   {'default': BufferStructure('T', 'B', 3)},
                                   NO_CON, NO_CON,
                                   size=4,
                                   activation=spec['activation'])

    spec['inits'] = {'timing': np.array([1, 2, 2, 3])}
    return layer, spec


def clockwork_lstm_layer_2d(spec):
    layer = ClockworkLstmLayerImpl(
        'ClockworkLstm',
        {'default': BufferStructure('T', 'B', 1, 2, 2)},
        NO_CON, NO_CON,
        size=3,
        activation=spec['activation'])

    spec['inits'] = {'timing': np.array([1, 2, 3])}
    return layer, spec


def merge(spec):
    in_shapes = {'inputs_1': BufferStructure('T', 'B', 3, 2),
                 'inputs_2': BufferStructure('T', 'B', 3, 4)}

    layer = MergeLayerImpl('Merge',
                           in_shapes, NO_CON, NO_CON)
    return layer, spec

layers_to_test = [
    noop_layer,
    loss_layer,
    fully_connected_layer,
    fully_connected_layer_2d,
    highway_layer,
    binomial_crossentropy_layer,
    softmax_ce_layer,
    softmax_fiddle_layer,
    sigmoid_ce_layer,
    rnn_layer,
    rnn_layer_2d,
    squared_difference_layer,
    squared_error_layer,
    lstm_layer,
    lstm_layer_2d,
    mask_layer,
    convolution_layer_2d_a,
    convolution_layer_2d_b,
    convolution_layer_2d_c,
    convolution_layer_2d,
    maxpooling_layer_2d,
    avgpooling_layer_2d,
    batch_norm_layer_fc,
    batch_norm_layer_nhwc,
    elementwise_layer,
    l1_decay_layer,
    l2_decay_layer,
    clockwork_layer,
    clockwork_layer_2d,
    clockwork_lstm_layer,
    clockwork_lstm_layer_2d,
    merge
]

ids = [f.__name__ for f in layers_to_test]

spec_list = [
    (1, 1, 'tanh'),
    (4, 1, 'sigmoid'),
    (2, 3, 'rel'),
    (1, 3, 'linear'),
]
spec_ids = ['{}{}{}'.format(*p) for p in spec_list]


@pytest.fixture(params=spec_list, ids=spec_ids)
def spec(request):
    time_steps, batch_size, activation = request.param
    return {
        'time_steps': time_steps,
        'batch_size': batch_size,
        'activation': activation
    }


@pytest.fixture(params=layers_to_test, ids=ids)
def layer_specs(request, spec):
    layer, specs = request.param(spec)
    return layer, specs


def test_deltas_calculation_of_layer(layer_specs):
    layer, specs = layer_specs
    successful = True
    for outputs_name in layer.out_shapes:
        if outputs_name in layer.takes_no_output_deltas_from:
            continue

        for inputs_name in layer.in_shapes:
            if inputs_name in layer.computes_no_input_deltas_for:
                continue
            successful &= run_deltas_test(layer, specs, inputs_name,
                                          outputs_name)

        assert successful, "Deltas check failed for {}".format(layer.name)


def test_gradients_for_layer(layer_specs):
    layer, specs = layer_specs
    successful = True
    for outputs_name in layer.out_shapes:
        if outputs_name in layer.takes_no_output_deltas_from:
            continue

        for param_name in layer.parameter_shapes:
            if param_name in layer.computes_no_gradients_for:
                continue
            successful &= run_gradients_test(layer, specs, param_name,
                                             outputs_name)

        assert successful, "Gradients check failed for {}".format(layer.name)


def test_layer_forward_pass_insensitive_to_internal_state_init(layer_specs):
    layer, specs = layer_specs
    layer_buffers = set_up_layer(layer, specs)
    time_steps = specs.get('time_steps', 3)

    eps = specs.get('eps', 1e-8)
    layer.forward_pass(layer_buffers)

    # get outputs after normal forward pass
    outputs = {}
    for key, value in layer_buffers.outputs.items():
        outputs[key] = HANDLER.get_numpy_copy(value)

    # randomize internal state
    for internal, shape_template in layer.internal_shapes.items():
        value = layer_buffers.internals[internal]
        if shape_template.scales_with_time:
            # exclude context slice
            HANDLER.set_from_numpy(
                value[:time_steps],
                np.random.randn(time_steps, *value.shape[1:]))
        else:
            HANDLER.set_from_numpy(value, np.random.randn(*value.shape))

        # compare new output
        layer.forward_pass(layer_buffers)
        for key, value in layer_buffers.outputs.items():
            assert np.allclose(outputs[key], HANDLER.get_numpy_copy(value),
                               rtol=eps, atol=eps), internal


def test_layer_backward_pass_insensitive_to_internal_state_init(layer_specs):
    layer, specs = layer_specs
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
    for internal, shape_template in layer.internal_shapes.items():
        value = layer_buffers.internals[internal]
        if shape_template.scales_with_time:
            # exclude context slice
            HANDLER.set_from_numpy(
                value[:time_steps],
                np.random.randn(time_steps, *value.shape[1:]))
        else:
            HANDLER.set_from_numpy(value, np.random.randn(*value.shape))

        # clear deltas
        for k, v in layer_buffers.input_deltas.items():
            HANDLER.fill(v, 0.0)

        # compare new deltas
        layer.forward_pass(layer_buffers)
        layer.backward_pass(layer_buffers)
        for key, value in layer_buffers.input_deltas.items():
            assert np.allclose(deltas[key], HANDLER.get_numpy_copy(value),
                               rtol=eps, atol=eps), \
                "Failed for internal.{} when inspecting {}".format(internal,
                                                                   key)


def test_layer_add_to_deltas(layer_specs):
    layer, specs = layer_specs
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
    # output_deltas buffer may have changed due to inplace activation. Reset.
    for key in layer_buffers.output_deltas.keys():
        HANDLER.fill(layer_buffers.output_deltas[key], 1.0)

    # do a second forward/backward pass
    layer.forward_pass(layer_buffers)
    layer.backward_pass(layer_buffers)

    # assert all input deltas are 1.0 bigger
    for key, value in layer_buffers.input_deltas.items():
        obtained = HANDLER.get_numpy_copy(value)
        passed = np.allclose(deltas[key] + 1.0, obtained,
                             rtol=eps, atol=eps)
        if not passed:
            print("Adding deltas test failed for {}!".format(key))
            print("Calculated Deltas:\n", obtained)
            print("Expected Deltas:\n", deltas[key] + 1.0)
            print("Difference:\n", deltas[key] + 1.0 - obtained)
        assert passed, key


def test_elementwise_act_func_gradients():
    pairs_to_test = [(HANDLER.sigmoid, HANDLER.sigmoid_deriv),
                     (HANDLER.tanh, HANDLER.tanh_deriv),
                     (HANDLER.rel, HANDLER.rel_deriv)]
    test_shape = (3, 2, 4)

    for fwd, bwd in pairs_to_test:
        inputs = HANDLER.create_from_numpy(np.random.randn(*test_shape))
        outputs = HANDLER.zeros(test_shape)
        doutputs = HANDLER.ones(test_shape)
        dinputs = HANDLER.zeros(test_shape)
        fwd(inputs, outputs)
        bwd(inputs, outputs, doutputs, dinputs)
        grad_calc = HANDLER.get_numpy_copy(dinputs)

        size = inputs.size
        x0 = HANDLER.get_numpy_copy(inputs).reshape((size,))

        def f(x):
            flat_inputs = inputs.reshape((size,))
            HANDLER.set_from_numpy(flat_inputs, x)
            HANDLER.fill(outputs, 0.)
            fwd(inputs, outputs)
            return HANDLER.get_numpy_copy(outputs).sum()

        grad_approx = approx_fprime(x0, f, 1e-5).reshape(grad_calc.shape)

        close = np.allclose(grad_approx, grad_calc, rtol=1e-4, atol=1e-4)
        if not close:
            print("-----------------------------")
            print("Testing", fwd.__name__)
            print('-- Approximated Gradient ----')
            print(grad_approx)
            print('---- Calculated Gradient ----')
            print(grad_calc)
            print('------------- Difference ----')
            print(grad_approx - grad_calc)
        assert close


def test_get_layer_class_from_typename():
    assert get_layer_class_from_typename('InputLayerImpl') == InputLayerImpl
    assert get_layer_class_from_typename('NoOpLayerImpl') == NoOpLayerImpl


def test_get_layer_class_from_typename_raises_typeerror():
    with pytest.raises(TypeError):
        get_layer_class_from_typename('NonexistentLayer')


def test_layer_constructor():
    a = Connection('l', 'default', 'A', 'default')
    b = Connection('l', 'default', 'B', 'default')
    c = Connection('l', 'default', 'C', 'default')

    l = FullyConnectedLayerImpl('LayerName',
                                {'default': BufferStructure('T', 'B', 5)},
                                {c},
                                {a, b},
                                size=8)
    expected = {'default': BufferStructure('T', 'B', 8)}
    assert l.out_shapes == expected
    assert l.in_shapes == {'default': BufferStructure('T', 'B', 5)}
    assert l.incoming == {c}
    assert l.outgoing == {a, b}
    assert l.kwargs == {'size': 8}


def test_nooplayer_raises_on_size_mismatch():
    with pytest.raises(LayerValidationError):
        l = NoOpLayerImpl('LayerName', {'default': ('T', 'B', 5,)}, NO_CON,
                          NO_CON, size=8)


def test_inputlayer_raises_on_in_size():
    with pytest.raises(LayerValidationError):
        l = InputLayerImpl('LayerName', {'default': ('T', 'B', 5,)}, NO_CON,
                           NO_CON, out_shapes={'default': ('T', 'B', 5,)})


@pytest.mark.parametrize("LayerClass", [
    NoOpLayerImpl, FullyConnectedLayerImpl
])
def test_raises_on_unexpected_kwargs(LayerClass):
    with pytest.raises(LayerValidationError) as excinfo:
        l = LayerClass('LayerName', {'default': BufferStructure(5,)},
                       NO_CON, NO_CON, some_foo=16)
    assert 'some_foo' in excinfo.value.args[0]
