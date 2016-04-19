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
        return ConstructionWrapper.create(FullyConnectedLayerImpl, name=name,
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
        self.size = (self.size,) if isinstance(self.size, int) else self.size

        if not isinstance(self.size, (tuple, list)) or \
                not all(isinstance(item, int) for item in self.size):
            raise LayerValidationError('size must be int or tuple[int] but '
                                       'was {}'.format(self.size))
        in_size = in_shapes['default'].feature_size

        outputs = OrderedDict()
        outputs['default'] = BufferStructure('T', 'B', *self.size)
        out_size = outputs['default'].feature_size

        parameters = OrderedDict()
        parameters['W'] = BufferStructure(out_size, in_size)
        parameters['bias'] = BufferStructure(out_size)

        internals = OrderedDict()
        return outputs, parameters, internals

    def forward_pass(self, buffers, training_pass=True):
        # prepare
        _h = self.handler
        W, bias = buffers.parameters
        inputs = flatten_time_and_features(buffers.inputs.default)
        outputs = flatten_time_and_features(buffers.outputs.default)

        # calculate outputs
        _h.dot_mm(inputs, W, outputs, transb=True)
        _h.add_mv(outputs, bias.reshape((1, bias.shape[0])), outputs)
        _h.inplace_act_func[self.activation](outputs)

    def backward_pass(self, buffers):
        # prepare
        _h = self.handler
        W, bias = buffers.parameters
        dW, dbias = buffers.gradients
        inputs = flatten_time_and_features(buffers.inputs.default)
        outputs = flatten_time_and_features(buffers.outputs.default)
        in_deltas = flatten_time_and_features(buffers.input_deltas.default)
        out_deltas = flatten_time_and_features(buffers.output_deltas.default)

        # calculate in_deltas and gradients
        _h.inplace_act_func_deriv[self.activation](outputs, out_deltas)
        _h.dot_add_mm(out_deltas, W, out=in_deltas)
        _h.dot_mm(out_deltas, inputs, out=dW, transa=True)
        _h.sum_t(out_deltas, axis=0, out=dbias)
