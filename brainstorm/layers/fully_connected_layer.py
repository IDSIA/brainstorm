#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from collections import OrderedDict
from brainstorm.utils import LayerValidationError
from brainstorm.layers.base_layer import LayerBaseImpl
from brainstorm.structure.shapes import ShapeTemplate


class FullyConnectedLayerImpl(LayerBaseImpl):
    expected_kwargs = {'size', 'activation_function'}

    def _setup_hyperparameters(self):
        self.act_func = None
        self.act_func_deriv = None
        self.size = self.kwargs.get('size',
                                    self.in_shapes['default'].feature_size)
        if not isinstance(self.size, int):
            raise LayerValidationError('size must be int but was {}'.
                                       format(self.size))

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
            self.kwargs.get('activation_function', 'linear')]

    def get_parameter_structure(self):
        in_size = self.in_shapes['default'].feature_size

        parameters = OrderedDict()
        parameters['W'] = ShapeTemplate(self.size, in_size)
        parameters['b'] = ShapeTemplate(self.size)
        return parameters

    def get_internal_structure(self):
        internals = OrderedDict()
        internals['Ha'] = ShapeTemplate('T', 'B', self.size)
        return internals

    def _get_output_shapes(self):
        return {'default': ShapeTemplate('T', 'B', self.size)}

    def forward_pass(self, forward_buffers, training_pass=True):
        # prepare
        _h = self.handler
        WX, W_bias = forward_buffers.parameters
        inputs = forward_buffers.inputs.default
        outputs = forward_buffers.outputs.default
        Ha = forward_buffers.internals.Ha

        # reshape
        t, b, f = inputs.shape
        flat_input = _h.reshape(inputs, (t * b, f))
        flat_Ha = _h.reshape(Ha, (t * b, self.out_shapes['default'][2]))

        # calculate outputs
        _h.dot_mm(flat_input, WX, flat_Ha, transb='T')
        _h.add_mv(flat_Ha, W_bias, flat_Ha)
        self.act_func(Ha, outputs)

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

        # reshape
        t, b, f = inputs.shape
        flat_input = _h.reshape(inputs, (t * b, f))
        flat_dHa = _h.reshape(dHa, (t * b, self.out_shapes['default'][2]))
        flat_in_delta_buffer = _h.reshape(in_deltas, (t * b, f))

        # calculate in_deltas and gradients
        self.act_func_deriv(Ha, outputs, out_deltas, dHa)
        _h.dot_add_mm(flat_dHa, WX, out=flat_in_delta_buffer)
        _h.dot_mm(flat_dHa, flat_input, dWX, transa='T')
        _h.sum_t(flat_dHa, axis=0, out=dW_bias)
