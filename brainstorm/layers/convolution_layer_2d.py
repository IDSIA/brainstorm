#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

from collections import OrderedDict
from brainstorm.layers.base_layer import Layer
from brainstorm.structure.buffer_structure import (BufferStructure,
                                                   StructureTemplate)
from brainstorm.structure.construction import ConstructionWrapper
from brainstorm.utils import flatten_time


def Convolution2D(num_filters, kernel_size, stride=(1, 1), padding=0,
                  activation='rel', name=None):
    """Create a 2D Convolution layer."""
    return ConstructionWrapper.create(Convolution2DLayerImpl,
                                      num_filters=num_filters,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=padding,
                                      activation=activation,
                                      name=name)


class Convolution2DLayerImpl(Layer):

    expected_inputs = {'default': StructureTemplate('T', 'B', '...')}
    expected_kwargs = {'num_filters', 'kernel_size', 'stride', 'padding',
                       'activation'}

    def setup(self, kwargs, in_shapes):
        self.activation = kwargs.get('activation', 'tanh')
        assert 'num_filters' in kwargs, "num_filters must be specified " \
                                        " for ConvolutionLayer"
        assert 'kernel_size' in kwargs, "kernel_size must be specified " \
                                        "for ConvolutionLayer"
        num_filters = kwargs['num_filters']
        kernel_size = kwargs['kernel_size']
        stride = kwargs.get('stride', (1, 1))
        padding = kwargs.get('padding', 0)
        assert type(padding) is int and padding >= 0, \
            "Invalid padding: {}".format(padding)
        assert type(kernel_size) in [list, tuple] and \
            len(kernel_size) == 2, "Kernel size must be list or tuple  of " \
                                   "length 2: {}".format(kernel_size)
        assert type(stride) in [list, tuple] and len(stride) == 2, \
            "Stride must be list or tuple of length 2: {}".format(stride)
        in_shape = self.in_shapes['default'].feature_shape
        assert stride[0] >= 0 and stride[1] >= 0, \
            "Invalid stride: {}".format(stride)
        assert isinstance(in_shape, tuple) and len(in_shape) == 3, \
            "ConvolutionLayer2D must have 3 dimensional input but input " \
            "shape was {}".format(in_shape)
        self.num_filters = num_filters
        self.kernel_size = tuple(kernel_size)
        self.stride = tuple(stride)
        self.padding = padding
        kernel_x, kernel_y = self.kernel_size
        num_input_maps = in_shape[2]

        output_height = ((in_shape[0] + 2 * padding - kernel_x) //
                         stride[0]) + 1
        output_width = ((in_shape[1] + 2 * padding - kernel_y) //
                        stride[1]) + 1
        out_shape = (output_height, output_width, num_filters)

        outputs = OrderedDict()
        outputs['default'] = BufferStructure('T', 'B', *out_shape)

        parameters = OrderedDict()
        parameters['W'] = BufferStructure(num_filters, kernel_x, kernel_y,
                                          num_input_maps)
        parameters['bias'] = BufferStructure(num_filters)

        internals = OrderedDict()
        return outputs, parameters, internals

    def forward_pass(self, buffers, training_pass=True):
        # prepare
        _h = self.handler
        W, bias = buffers.parameters
        inputs = buffers.inputs.default
        outputs = buffers.outputs.default

        # reshape
        flat_inputs = flatten_time(inputs)
        flat_outputs = flatten_time(outputs)

        # calculate outputs
        _h.conv2d_forward_batch(flat_inputs, W, bias, flat_outputs,
                                self.padding, self.stride)
        _h.inplace_act_func[self.activation](outputs)

    def backward_pass(self, buffers):
        # prepare
        _h = self.handler
        W, bias = buffers.parameters
        dW, dbias = buffers.gradients
        inputs = buffers.inputs.default
        outputs = buffers.outputs.default
        in_deltas = buffers.input_deltas.default
        out_deltas = buffers.output_deltas.default

        # reshape
        flat_inputs = flatten_time(inputs)
        flat_in_deltas = flatten_time(in_deltas)
        flat_out_deltas = flatten_time(out_deltas)

        # calculate in_deltas and gradients
        _h.inplace_act_func_deriv[self.activation](outputs, out_deltas)
        _h.conv2d_backward_batch(flat_inputs, W, self.padding, self.stride,
                                 flat_in_deltas, flat_out_deltas, dW, dbias)
