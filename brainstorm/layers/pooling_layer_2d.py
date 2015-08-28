#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from collections import OrderedDict
from brainstorm.layers.base_layer import LayerBaseImpl
from brainstorm.structure.shapes import ShapeTemplate


class Pooling2DLayerImpl(LayerBaseImpl):
    expected_kwargs = {'num_filters', 'kernel_size', 'stride', 'padding',
                       'activation_function'}
    inputs = {'default': ShapeTemplate('T', 'B', '...')}
    outputs = {'default': ShapeTemplate('T', 'B', '...')}

    def _setup_hyperparameters(self):
        assert 'num_filters' in self.kwargs, "num_filters must be specified " \
                                             " for PoolingLayer2D"
        assert 'kernel_size' in self.kwargs, "kernel_size must be specified " \
                                             "for PoolingLayer2D"
        self.num_filters = self.kwargs['num_filters']
        self.kernel_size = self.kwargs['kernel_size']
        self.padding = self.kwargs.get('padding', 0)
        self.stride = self.kwargs.get('stride', (1, 1))
        assert type(self.padding) is int and self.padding >= 0
        assert type(self.stride) is tuple and self.stride[0] >= 0 and \
            self.stride[1] >= 0

    def get_internal_structure(self):
        argmax_shape = self.out_shapes['default'].feature_shape + (2, )
        internals = OrderedDict()
        internals['argmax'] = ShapeTemplate('T', 'B', *argmax_shape)
        return internals

    def _get_output_shapes(self):
        kernel_size = self.kernel_size
        padding = self.padding
        stride = self.stride
        num_filters = self.num_filters
        in_shape = self.in_shapes['default'].feature_shape
        assert isinstance(in_shape, tuple) and len(in_shape) == 3, \
            "PoolingLayer2D must have 3 dimensional input but input " \
            "shape was %s" % in_shape

        output_height = ((in_shape[1] + 2 * padding - kernel_size[0]) //
                         stride[0]) + 1
        output_width = ((in_shape[2] + 2 * padding - kernel_size[1]) //
                        stride[1]) + 1
        output_shape = (num_filters, output_height, output_width)
        return {'default': ShapeTemplate('T', 'B', *output_shape)}

    def forward_pass(self, forward_buffers, training_pass=True):
        # prepare
        _h = self.handler
        inputs = forward_buffers.inputs.default
        outputs = forward_buffers.outputs.default
        argmax = forward_buffers.internals.argmax

        # reshape
        t, b, c, h, w = inputs.shape
        flat_inputs = _h.reshape(inputs, (t * b, c, h, w))
        flat_outputs = _h.reshape(outputs, (t * b,) + outputs.shape[2:])
        flat_argmax = _h.reshape(argmax, (t * b,) + argmax.shape[2:])

        # calculate outputs
        _h.pool2d_forward_batch(flat_inputs, self.kernel_size, flat_outputs,
                                self.padding, self.stride, flat_argmax)

    def backward_pass(self, forward_buffers, backward_buffers):

        # prepare
        _h = self.handler
        argmax = forward_buffers.internals.argmax
        inputs = forward_buffers.inputs.default
        outputs = forward_buffers.outputs.default
        in_deltas = backward_buffers.inputs.default
        out_deltas = backward_buffers.outputs.default

        # reshape
        t, b, c, h, w = inputs.shape
        flat_inputs = _h.reshape(inputs, (t * b, c, h, w))
        flat_in_deltas = _h.reshape(in_deltas, (t * b, c, h, w))
        flat_out_deltas = _h.reshape(out_deltas, (t * b,)
                                     + out_deltas.shape[2:])
        flat_outputs = _h.reshape(outputs, (t * b,) + outputs.shape[2:])
        flat_argmax = _h.reshape(argmax, (t * b,) + argmax.shape[2:])

        _h.pool2d_backward_batch(flat_inputs, self.kernel_size,
                                 flat_outputs, self.padding, self.stride,
                                 flat_argmax, flat_in_deltas, flat_out_deltas)
