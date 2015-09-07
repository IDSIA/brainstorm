#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from collections import OrderedDict
from brainstorm.layers.base_layer import LayerBaseImpl
from brainstorm.structure.shapes import ShapeTemplate


class Pooling2DLayerImpl(LayerBaseImpl):
    expected_kwargs = {'num_filters', 'kernel_size', 'type', 'stride',
                       'padding', 'activation_function'}
    inputs = {'default': ShapeTemplate('T', 'B', '...')}
    outputs = {'default': ShapeTemplate('T', 'B', '...')}

    def _setup_hyperparameters(self):
        assert 'num_filters' in self.kwargs, "num_filters must be specified " \
                                             " for PoolingLayer2D"
        assert 'kernel_size' in self.kwargs, "kernel_size must be specified " \
                                             "for PoolingLayer2D"
        assert 'type' in self.kwargs, "type must be specified " \
                                      "for PoolingLayer2D"
        self.num_filters = self.kwargs['num_filters']
        self.kernel_size = self.kwargs['kernel_size']
        self.type = self.kwargs['type']
        self.padding = self.kwargs.get('padding', 0)
        self.stride = self.kwargs.get('stride', (1, 1))
        assert self.type in ('max', 'avg')
        assert type(self.padding) is int and self.padding >= 0
        assert type(self.stride) is tuple and self.stride[0] >= 0 and \
            self.stride[1] >= 0

    def get_internal_structure(self):
        argmax_shape = self.out_shapes['default'].feature_shape + (2, )
        internals = OrderedDict()
        if self.type == 'max':
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

    def forward_pass(self, buffers, training_pass=True):
        # prepare
        _h = self.handler
        inputs = buffers.inputs.default
        outputs = buffers.outputs.default

        # reshape
        t, b, c, h, w = inputs.shape
        flat_inputs = inputs.reshape((t * b, c, h, w))
        flat_outputs = outputs.reshape((t * b,) + outputs.shape[2:])

        # calculate outputs
        if self.type == 'max':
            argmax = buffers.internals.argmax
            flat_argmax = argmax.reshape((t * b,) + argmax.shape[2:])
            _h.maxpool2d_forward_batch(flat_inputs, self.kernel_size,
                                       flat_outputs, self.padding, self.stride,
                                       flat_argmax)
        elif self.type == 'avg':
            _h.avgpool2d_forward_batch(flat_inputs, self.kernel_size,
                                       flat_outputs, self.padding, self.stride)

    def backward_pass(self, buffers):

        # prepare
        _h = self.handler
        inputs = buffers.inputs.default
        outputs = buffers.outputs.default
        in_deltas = buffers.input_deltas.default
        out_deltas = buffers.output_deltas.default

        # reshape
        t, b, c, h, w = inputs.shape
        flat_inputs = inputs.reshape((t * b, c, h, w))
        flat_in_deltas = in_deltas.reshape((t * b, c, h, w))
        flat_out_deltas = out_deltas.reshape((t * b,)
                                             + out_deltas.shape[2:])
        flat_outputs = outputs.reshape((t * b,) + outputs.shape[2:])


        if self.type == 'max':
            argmax = buffers.internals.argmax
            flat_argmax = argmax.reshape((t * b,) + argmax.shape[2:])
            _h.maxpool2d_backward_batch(flat_inputs, self.kernel_size,
                                        flat_outputs, self.padding,
                                        self.stride, flat_argmax,
                                        flat_in_deltas, flat_out_deltas)
        elif self.type == 'avg':
            _h.avgpool2d_backward_batch(flat_inputs, self.kernel_size,
                                        flat_outputs, self.padding,
                                        self.stride,
                                        flat_in_deltas, flat_out_deltas)
