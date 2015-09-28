#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from collections import OrderedDict
from brainstorm.structure.construction import ConstructionWrapper
from brainstorm.layers.base_layer import LayerBaseImpl
from brainstorm.structure.shapes import StructureTemplate, BufferStructure
from brainstorm.utils import flatten_time


def Convolution2D(num_filters, kernel_size, stride=(1, 1), padding=0,
                  activation_function='rel', name=None):
    """Create a 2D Convolution layer."""
    return ConstructionWrapper.create('Convolution2D',
                                      num_filters=num_filters,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=padding,
                                      activation_function=activation_function,
                                      name=name)


class Convolution2DLayerImpl(LayerBaseImpl):

    expected_inputs = {'default': StructureTemplate('T', 'B', '...')}
    expected_kwargs = {'num_filters', 'kernel_size', 'stride', 'padding',
                       'activation_function'}

    def setup(self, kwargs, in_shapes):
        self.act_func = None
        self.act_func_deriv = None
        assert 'num_filters' in kwargs, "num_filters must be specified " \
                                        " for ConvolutionLayer"
        assert 'kernel_size' in kwargs, "kernel_size must be specified " \
                                        "for ConvolutionLayer"
        self.num_filters = kwargs['num_filters']
        self.kernel_size = kwargs['kernel_size']
        self.stride = kwargs.get('stride', (1, 1))
        self.padding = kwargs.get('padding', 0)
        assert type(self.padding) is int and self.padding >= 0, \
            "Invalid padding: {}".format(self.padding)
        assert type(self.kernel_size) in [list, tuple] and \
            len(self.kernel_size) == 2, "Kernel size must be list or " \
                                        "tuple  of length 2: {}".format(
                                        self.kernel_size)
        assert type(self.stride) in [list, tuple] and len(self.stride) == 2, \
            "Stride must be list or tuple of length 2: {}".format(self.stride)
        in_shape = self.in_shapes['default'].feature_shape
        assert self.stride[0] >= 0 and self.stride[1] >= 0, \
            "Invalid stride: {}".format(self.stride)
        assert isinstance(in_shape, tuple) and len(in_shape) == 3, \
            "ConvolutionLayer2D must have 3 dimensional input but input " \
            "shape was %s" % in_shape
        num_input_maps = in_shape[0]
        num_filters = self.num_filters
        kernel_x, kernel_y = self.kernel_size
        padding, stride = self.padding, self.stride
        output_height = ((in_shape[1] + 2 * padding - kernel_x) //
                         stride[0]) + 1
        output_width = ((in_shape[2] + 2 * padding - kernel_y) //
                        stride[1]) + 1
        out_shape = (num_filters, output_height, output_width)

        outputs = OrderedDict()
        outputs['default'] = BufferStructure('T', 'B', *out_shape)

        parameters = OrderedDict()
        parameters['W'] = BufferStructure(num_filters, num_input_maps,
                                          kernel_x, kernel_y)
        parameters['bias'] = BufferStructure(num_filters)

        internals = OrderedDict()
        internals['H'] = BufferStructure('T', 'B', *out_shape)
        internals['dH'] = BufferStructure('T', 'B', *out_shape,
                                          is_backward_only=True)

        return outputs, parameters, internals

    def set_handler(self, new_handler):
        super(Convolution2DLayerImpl, self).set_handler(new_handler)

        # Assign act_func and act_dunc_derivs
        activation_functions = {
            'sigmoid': (self.handler.sigmoid, self.handler.sigmoid_deriv),
            'tanh': (self.handler.tanh, self.handler.tanh_deriv),
            'linear': (lambda x, y: self.handler.copy_to(y, x),
                       lambda x, y, dy, dx: self.handler.copy_to(dx, dy)),
            'rel': (self.handler.rel, self.handler.rel_deriv)
        }

        self.act_func, self.act_func_deriv = activation_functions[
            self.kwargs.get('activation_function', 'rel')]

    def forward_pass(self, buffers, training_pass=True):
        # prepare
        _h = self.handler
        W, bias = buffers.parameters
        inputs = buffers.inputs.default
        outputs = buffers.outputs.default
        H = buffers.internals.H

        # reshape
        flat_inputs = flatten_time(inputs)
        flat_H = flatten_time(H)

        # calculate outputs
        _h.conv2d_forward_batch(flat_inputs, W, bias, flat_H,
                                self.padding, self.stride)
        self.act_func(H, outputs)

    def backward_pass(self, buffers):
        # prepare
        _h = self.handler
        W, bias = buffers.parameters
        dW, dbias = buffers.gradients
        inputs = buffers.inputs.default
        outputs = buffers.outputs.default
        in_deltas = buffers.input_deltas.default
        out_deltas = buffers.output_deltas.default
        H, dH = buffers.internals

        # reshape
        flat_inputs = flatten_time(inputs)
        flat_in_deltas = flatten_time(in_deltas)
        flat_dH = flatten_time(dH)

        # calculate in_deltas and gradients
        self.act_func_deriv(H, outputs, out_deltas, dH)
        _h.conv2d_backward_batch(flat_inputs, W, self.padding, self.stride,
                                 flat_in_deltas, flat_dH, dW, dbias)
