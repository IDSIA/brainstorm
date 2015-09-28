#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from collections import OrderedDict
from brainstorm.structure.construction import ConstructionWrapper
from brainstorm.utils import LayerValidationError, flatten_time, \
    flatten_time_and_features
from brainstorm.layers.base_layer import BaseLayerImpl
from brainstorm.structure.shapes import StructureTemplate, BufferStructure


def FullyConnected(size, activation_function='rel', name=None):
    """Create a Fully Connected (inner product) layer."""
    return ConstructionWrapper.create('FullyConnected', size=size, name=name,
                                      activation_function=activation_function)


class FullyConnectedLayerImpl(BaseLayerImpl):

    expected_inputs = {'default': StructureTemplate('T', 'B', '...')}
    expected_kwargs = {'size', 'activation_function'}

    def set_handler(self, new_handler):
        super(FullyConnectedLayerImpl, self).set_handler(new_handler)

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

    def setup(self, kwargs, in_shapes):
        self.act_func = None
        self.act_func_deriv = None
        self.size = kwargs.get('size', self.in_shapes['default'].feature_size)
        if not isinstance(self.size, int):
            raise LayerValidationError('size must be int but was {}'.
                                       format(self.size))
        in_size = in_shapes['default'].feature_size

        outputs = OrderedDict()
        outputs['default'] = BufferStructure('T', 'B', self.size)

        parameters = OrderedDict()
        parameters['W'] = BufferStructure(self.size, in_size)
        parameters['bias'] = BufferStructure(self.size)

        internals = OrderedDict()
        internals['H'] = BufferStructure('T', 'B', self.size)
        internals['dH'] = BufferStructure('T', 'B', self.size,
                                          is_backward_only=True)
        return outputs, parameters, internals

    def forward_pass(self, buffers, training_pass=True):
        # prepare
        _h = self.handler
        W, bias = buffers.parameters
        inputs = buffers.inputs.default
        outputs = buffers.outputs.default
        H = buffers.internals.H

        # reshape
        flat_input = flatten_time_and_features(inputs)
        flat_H = flatten_time(H)

        # calculate outputs
        _h.dot_mm(flat_input, W, flat_H, transb=True)
        _h.add_mv(flat_H, bias.reshape((1, self.size)), flat_H)
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
        flat_input = flatten_time_and_features(inputs)
        flat_dH = flatten_time(dH)
        flat_in_delta_buffer = flatten_time_and_features(in_deltas)

        # calculate in_deltas and gradients
        self.act_func_deriv(H, outputs, out_deltas, dH)
        _h.dot_add_mm(flat_dH, W, out=flat_in_delta_buffer)
        _h.dot_mm(flat_dH, flat_input, out=dW, transa=True)
        _h.sum_t(flat_dH, axis=0, out=dbias)
