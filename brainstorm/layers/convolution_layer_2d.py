#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from collections import OrderedDict
from brainstorm.utils import LayerValidationError
from brainstorm.layers.base_layer import LayerBaseImpl
from brainstorm.structure.shapes import ShapeTemplate


class ConvolutionLayer2DImpl(LayerBaseImpl):
    expected_kwargs = {'num_filters', 'kernel_size', 'stride', 'pad',
                       'activation_function'}

    def __init__(self, name, in_shapes, incoming_connections,
                 outgoing_connections, **kwargs):
        super(ConvolutionLayer2DImpl, self).__init__(
            name, in_shapes, incoming_connections, outgoing_connections,
            **kwargs)
        self.act_func = None
        self.act_func_deriv = None
        self.kwargs = kwargs
        assert 'num_filters' in kwargs, "num_filters must be specified for " \
                                        "ConvolutionLayer"
        assert 'kernel_size' in kwargs, "kernel_size must be specified for " \
                                        "ConvolutionLayer"
        self.num_filters = kwargs['num_filters']
        self.kernel_size = kwargs['kernel_size']
        self.pad = kwargs.get('pad', (0, 0))
        self.stride = kwargs.get('stride', (0, 0))

    def set_handler(self, new_handler):
        super(ConvolutionLayer2DImpl, self).set_handler(new_handler)

        # Assign act_func and act_dunc_derivs
        activation_functions = {
            'sigmoid': (self.handler.sigmoid, self.handler.sigmoid_deriv),
            'tanh': (self.handler.tanh, self.handler.tanh_deriv),
            'linear': (lambda x, y: self.handler.copy_to(y, x),
                       lambda x, y, dy, dx: self.handler.copy_to(dx, dy)),
            'rel': (self.handler.rel, self.handler.rel_deriv)
        }

        self.act_func, self.act_func_deriv = activation_functions[
            self.kwargs.get('activation_function', 'linear')]

    def get_parameter_structure(self):
        in_shape = self.in_shapes['default'].feature_size
        num_input_maps = in_shape[0]
        num_filters = self.num_filters
        kernel_x = self.kernel_size[0]
        kernel_y = self.kernel_size[1]

        parameters = OrderedDict()
        parameters['W'] = ShapeTemplate(num_input_maps, num_filters,
                                        kernel_x, kernel_y)
        parameters['b'] = ShapeTemplate(num_filters)
        return parameters

    def get_internal_structure(self):
        output_shape = self.out_shapes['default'].feature_size

        internals = OrderedDict()
        internals['Ha'] = ShapeTemplate('T', 'B', output_shape)
        return internals

    def _get_output_shapes(self):
        kernel_size = self.kernel_size
        pad = self.pad
        stride = self.stride
        num_filters = self.num_filters
        in_shape = self.in_shapes['default'].feature_size
        assert len(in_shape) == 3, "The shape of input must be 3 for " \
                                   "ConvolutionLayer2D"

        output_height = ((in_shape[1] + 2 * pad[0] - kernel_size[0]) /
                         stride[0]) + 1
        output_width = ((in_shape[2] + 2 * pad[1] - kernel_size[1]) /
                        stride[1]) + 1
        output_shape = (num_filters, output_height, output_width)
        return {'default': ShapeTemplate('T', 'B', output_shape)}

    def forward_pass(self, forward_buffers, training_pass=True):
        # prepare
        _h = self.handler
        WX, W_bias = forward_buffers.parameters
        input = forward_buffers.inputs.default
        output = forward_buffers.outputs.default
        Ha = forward_buffers.internals.Ha

        # calculate outputs
        _h.conv2d_forward_batch(input, WX, Ha)
        self.act_func(Ha, output)

    def backward_pass(self, forward_buffers, backward_buffers):

        # prepare
        _h = self.handler
        WX, W_bias = forward_buffers.parameters
        dWX, dW_bias = backward_buffers.parameters
        inputs = forward_buffers.inputs.default
        outputs = forward_buffers.outputs.default
        in_deltas = backward_buffers.inputs.default
        out_deltas = backward_buffers.outputs.default
        Ha = forward_buffers.internals.Ha
        dHa = backward_buffers.internals.Ha

        # calculate in_deltas and gradients
        self.act_func_deriv(Ha, outputs, out_deltas, dHa)
        _h.conv2d_backward_batch(dHa, inputs, in_deltas,
                                 WX, W_bias, dWX, dW_bias)
