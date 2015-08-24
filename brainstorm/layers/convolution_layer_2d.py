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
    inputs = {'default': ShapeTemplate('T', 'B', '...')}
    outputs = {'default': ShapeTemplate('T', 'B', '...')}

    def _setup_hyperparameters(self):
        self.act_func = None
        self.act_func_deriv = None
        self.kwargs = self.kwargs
        assert 'num_filters' in self.kwargs, "num_filters must be specified " \
                                             " for ConvolutionLayer"
        assert 'kernel_size' in self.kwargs, "kernel_size must be specified " \
                                             "for ConvolutionLayer"
        self.num_filters = self.kwargs['num_filters']
        self.kernel_size = self.kwargs['kernel_size']
        self.pad = self.kwargs.get('pad', 0)
        self.stride = self.kwargs.get('stride', (1, 1))
        assert type(self.pad) is int and self.pad >= 0
        assert type(self.stride) is tuple and self.stride[0] >= 0 and \
            self.stride >= 0

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
        in_shape = self.in_shapes['default'].feature_shape
        num_input_maps = in_shape[0]
        num_filters = self.num_filters
        kernel_x = self.kernel_size[0]
        kernel_y = self.kernel_size[1]

        parameters = OrderedDict()
        parameters['W'] = ShapeTemplate(num_filters, num_input_maps,
                                        kernel_x, kernel_y)
        parameters['bias'] = ShapeTemplate(num_filters)
        return parameters

    def get_internal_structure(self):
        output_shape = self.out_shapes['default'].feature_shape

        internals = OrderedDict()
        internals['H'] = ShapeTemplate('T', 'B', *output_shape)
        internals['dH'] = ShapeTemplate('T', 'B', *output_shape,
                                        is_backward_only=True)
        return internals

    def _get_output_shapes(self):
        kernel_size = self.kernel_size
        pad = self.pad
        stride = self.stride
        num_filters = self.num_filters
        in_shape = self.in_shapes['default'].feature_shape
        assert isinstance(in_shape, tuple) and len(in_shape) == 3, \
            "ConvolutionLayer2D must have 3 dimensional input but input " \
            "shape was %s" % in_shape

        output_height = ((in_shape[1] + 2 * pad - kernel_size[0]) //
                         stride[0]) + 1
        output_width = ((in_shape[2] + 2 * pad - kernel_size[1]) //
                        stride[1]) + 1
        output_shape = (num_filters, output_height, output_width)
        return {'default': ShapeTemplate('T', 'B', *output_shape)}

    def forward_pass(self, buffers, training_pass=True):
        # prepare
        _h = self.handler
        W, bias = buffers.parameters
        inputs = buffers.inputs.default
        outputs = buffers.outputs.default
        H = buffers.internals.H

        # reshape
        t, b, c, h, w = inputs.shape
        flat_inputs = _h.reshape(inputs, (t * b, c, h, w))
        flat_H = _h.reshape(H, (t * b,) + self.out_shapes['default'][2:])

        # calculate outputs
        _h.conv2d_forward_batch(flat_inputs, W, bias, flat_H,
                                self.pad, self.stride)
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
        t, b, c, h, w = inputs.shape
        flat_inputs = _h.reshape(inputs, (t * b, c, h, w))
        flat_in_deltas = _h.reshape(in_deltas, (t * b, c, h, w))
        flat_dH = _h.reshape(dH, (t * b,) + self.out_shapes['default'][2:])

        # calculate in_deltas and gradients
        self.act_func_deriv(H, outputs, out_deltas, dH)
        _h.conv2d_backward_batch(flat_inputs, W, self.pad, self.stride,
                                 flat_in_deltas, flat_dH, dW, dbias)
