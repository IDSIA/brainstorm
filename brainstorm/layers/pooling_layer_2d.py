#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from collections import OrderedDict
from brainstorm.structure.construction import ConstructionWrapper
from brainstorm.layers.base_layer import BaseLayerImpl
from brainstorm.structure.buffer_structure import (StructureTemplate,
                                                   BufferStructure)


def Pooling2D(kernel_size, type='max', stride=(1, 1), padding=0, name=None):
    """Create a 2D Pooling layer."""
    return ConstructionWrapper.create('Pooling2D', kernel_size=kernel_size,
                                      type=type, stride=stride,
                                      padding=padding, name=name)


class Pooling2DLayerImpl(BaseLayerImpl):

    expected_inputs = {'default': StructureTemplate('T', 'B', '...')}
    expected_kwargs = {'kernel_size', 'type', 'stride',
                       'padding', 'activation_function'}

    def setup(self, kwargs, in_shapes):
        assert 'kernel_size' in kwargs, "kernel_size must be specified for " \
                                        "Pooling2D"
        assert 'type' in kwargs, "type must be specified for Pooling2D"
        self.kernel_size = kwargs['kernel_size']
        self.type = kwargs['type']
        self.padding = kwargs.get('padding', 0)
        self.stride = kwargs.get('stride', (1, 1))
        in_shape = self.in_shapes['default'].feature_shape
        assert self.type in ('max', 'avg')
        assert type(self.padding) is int and self.padding >= 0, \
            "Invalid padding: {}".format(self.padding)
        assert type(self.kernel_size) in [list, tuple] and \
            len(self.kernel_size) == 2, "Kernel size must be list or " \
                                        "tuple  of length 2: {}".format(
                                        self.kernel_size)
        assert type(self.stride) in [list, tuple] and len(self.stride) == 2, \
            "Stride must be list or tuple of length 2: {}".format(self.stride)
        assert self.stride[0] >= 0 and self.stride[1] >= 0, \
            "Invalid stride: {}".format(self.stride)
        assert isinstance(in_shape, tuple) and len(in_shape) == 3, \
            "PoolingLayer2D must have 3 dimensional input but input " \
            "shape was %s" % in_shape

        kernel_size = self.kernel_size
        padding = self.padding
        stride = self.stride
        output_height = ((in_shape[1] + 2 * padding - kernel_size[0]) //
                         stride[0]) + 1
        output_width = ((in_shape[2] + 2 * padding - kernel_size[1]) //
                        stride[1]) + 1
        assert output_height > 0 and output_width > 0, \
            "Evaluated output height and width must be positive but were " \
            "({}, {})".format(output_height, output_width)
        output_shape = (in_shape[0], output_height, output_width)

        outputs = OrderedDict()
        outputs['default'] = BufferStructure('T', 'B', *output_shape)

        internals = OrderedDict()
        if self.type == 'max':
            argmax_shape = self.out_shapes['default'].feature_shape + (2, )
            internals['argmax'] = BufferStructure('T', 'B', *argmax_shape)
        return outputs, OrderedDict(), internals

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
        flat_out_deltas = out_deltas.reshape((t * b,) + out_deltas.shape[2:])
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
