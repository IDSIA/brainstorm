#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

from collections import OrderedDict

from brainstorm.layers.base_layer import Layer
from brainstorm.structure.buffer_structure import (BufferStructure,
                                                   StructureTemplate)
from brainstorm.structure.construction import ConstructionWrapper
from brainstorm.utils import flatten_time


def Pooling2D(kernel_size, type='max', stride=(1, 1), padding=0, name=None):
    """Create a 2D Pooling layer."""
    return ConstructionWrapper.create(Pooling2DLayerImpl,
                                      kernel_size=kernel_size,
                                      type=type, stride=stride,
                                      padding=padding, name=name)


class Pooling2DLayerImpl(Layer):

    expected_inputs = {'default': StructureTemplate('T', 'B', '...')}
    expected_kwargs = {'kernel_size', 'type', 'stride',
                       'padding', 'activation_function'}

    def setup(self, kwargs, in_shapes):
        assert 'kernel_size' in kwargs, "kernel_size must be specified for " \
                                        "Pooling2D"
        assert 'type' in kwargs, "type must be specified for Pooling2D"
        kernel_size = kwargs['kernel_size']
        ptype = kwargs['type']
        padding = kwargs.get('padding', 0)
        stride = kwargs.get('stride', (1, 1))
        in_shape = self.in_shapes['default'].feature_shape
        assert ptype in ('max', 'avg')
        assert type(padding) is int and padding >= 0, \
            "Invalid padding: {}".format(padding)
        assert type(kernel_size) in [list, tuple] and \
            len(kernel_size) == 2, "Kernel size must be list or " \
                                   "tuple  of length 2: {}".format(
                                   kernel_size)
        assert type(stride) in [list, tuple] and len(stride) == 2, \
            "Stride must be list or tuple of length 2: {}".format(stride)
        assert stride[0] >= 0 and stride[1] >= 0, \
            "Invalid stride: {}".format(stride)
        assert isinstance(in_shape, tuple) and len(in_shape) == 3, \
            "PoolingLayer2D must have 3 dimensional input but input " \
            "shape was %s" % in_shape

        self.kernel_size = tuple(kernel_size)
        self.type = ptype
        self.padding = padding
        self.stride = tuple(stride)
        output_height = ((in_shape[0] + 2 * padding - kernel_size[0]) //
                         stride[0]) + 1
        output_width = ((in_shape[1] + 2 * padding - kernel_size[1]) //
                        stride[1]) + 1
        assert output_height > 0 and output_width > 0, \
            "Evaluated output height and width must be positive but were " \
            "({}, {})".format(output_height, output_width)
        output_shape = (output_height, output_width, in_shape[2])

        outputs = OrderedDict()
        outputs['default'] = BufferStructure('T', 'B', *output_shape)

        internals = OrderedDict()
        if self.type == 'max':
            argmax_shape = outputs['default'].feature_shape
            internals['argmax'] = BufferStructure('T', 'B', *argmax_shape)
        return outputs, OrderedDict(), internals

    def forward_pass(self, buffers, training_pass=True):
        # prepare
        _h = self.handler
        inputs = buffers.inputs.default
        outputs = buffers.outputs.default

        # reshape
        flat_inputs = flatten_time(inputs)
        flat_outputs = flatten_time(outputs)

        # calculate outputs
        if self.type == 'max':
            argmax = buffers.internals.argmax
            flat_argmax = flatten_time(argmax)
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
        flat_inputs = flatten_time(inputs)
        flat_in_deltas = flatten_time(in_deltas)
        flat_out_deltas = flatten_time(out_deltas)
        flat_outputs = flatten_time(outputs)

        if self.type == 'max':
            argmax = buffers.internals.argmax
            flat_argmax = flatten_time(argmax)
            _h.maxpool2d_backward_batch(flat_inputs, self.kernel_size,
                                        flat_outputs, self.padding,
                                        self.stride, flat_argmax,
                                        flat_in_deltas, flat_out_deltas)
        elif self.type == 'avg':
            _h.avgpool2d_backward_batch(flat_inputs, self.kernel_size,
                                        flat_outputs, self.padding,
                                        self.stride,
                                        flat_in_deltas, flat_out_deltas)
