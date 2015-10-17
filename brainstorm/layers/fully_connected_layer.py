#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

from collections import OrderedDict

from brainstorm.layers.base_layer import Layer
from brainstorm.structure.buffer_structure import (BufferStructure,
                                                   StructureTemplate)
from brainstorm.structure.construction import ConstructionWrapper
from brainstorm.utils import (LayerValidationError, flatten_time,
                              flatten_time_and_features)


def FullyConnected(size=None, activation='rel', name=None):
    """Create a Fully Connected (inner product) layer."""
    if size is None:
        ConstructionWrapper.create(FullyConnectedLayerImpl, name=name,
                                   activation=activation)
    else:
        return ConstructionWrapper.create(FullyConnectedLayerImpl, size=size,
                                          name=name, activation=activation)


class FullyConnectedLayerImpl(Layer):

    expected_inputs = {'default': StructureTemplate('T', 'B', '...')}
    expected_kwargs = {'size', 'activation'}

    def setup(self, kwargs, in_shapes):
        self.activation = kwargs.get('activation', 'rel')
        self.size = kwargs.get('size', self.in_shapes['default'].feature_shape)
        if isinstance(self.size, int):
            self.size = (self.size,)

        if not isinstance(self.size, (tuple, list)):
            raise LayerValidationError('size must be int but was {}'.
                                       format(self.size))
        in_size = in_shapes['default'].feature_size

        outputs = OrderedDict()
        outputs['default'] = BufferStructure('T', 'B', *self.size)
        out_size = outputs['default'].feature_size

        parameters = OrderedDict()
        parameters['W'] = BufferStructure(out_size, in_size)
        parameters['bias'] = BufferStructure(out_size)

        internals = OrderedDict()
        internals['H'] = BufferStructure('T', 'B', out_size)
        internals['dH'] = BufferStructure('T', 'B', out_size,
                                          is_backward_only=True)
        return outputs, parameters, internals

    def forward_pass(self, buffers, training_pass=True):
        # prepare
        _h = self.handler
        W, bias = buffers.parameters
        inputs = flatten_time_and_features(buffers.inputs.default)
        outputs = flatten_time_and_features(buffers.outputs.default)
        H = flatten_time(buffers.internals.H)

        # calculate outputs
        _h.dot_mm(inputs, W, H, transb=True)
        _h.add_mv(H, bias.reshape((1, bias.shape[0])), H)
        _h.act_func[self.activation](H, outputs)

    def backward_pass(self, buffers):
        # prepare
        _h = self.handler
        W, bias = buffers.parameters
        dW, dbias = buffers.gradients
        inputs = flatten_time_and_features(buffers.inputs.default)
        outputs = flatten_time_and_features(buffers.outputs.default)
        in_deltas = flatten_time_and_features(buffers.input_deltas.default)
        out_deltas = flatten_time_and_features(buffers.output_deltas.default)
        H = flatten_time(buffers.internals.H)
        dH = flatten_time(buffers.internals.dH)

        # calculate in_deltas and gradients
        _h.act_func_deriv[self.activation](H, outputs, out_deltas, dH)
        _h.dot_add_mm(dH, W, out=in_deltas)
        _h.dot_mm(dH, inputs, out=dW, transa=True)
        _h.sum_t(dH, axis=0, out=dbias)
