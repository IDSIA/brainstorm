#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from brainstorm.layers.base_layer import LayerBaseImpl


class FullyConnectedLayerImpl(LayerBaseImpl):
    expected_kwargs = {'shape', 'activation_function'}

    def __init__(self, name, in_shapes, incoming_connections,
                 outgoing_connections, **kwargs):
        super(FullyConnectedLayerImpl, self).__init__(
            name, in_shapes, incoming_connections, outgoing_connections,
            **kwargs)
        self.act_func = None
        self.act_func_deriv = None
        self.kwargs = kwargs

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
        return {
            'W': {
                '@shape': (self.in_shapes['default'][2],
                           self.out_shapes['default'][2]),
                '@index': 0},
            'b': {
                '@shape': (self.out_shapes['default'][2],),
                '@index': 1}
        }

    def get_internal_structure(self):
        return {
            'Ha': {
                '@shape': ('T', 'B', self.out_shapes['default'][2]),
                '@index': 0}
        }

    def forward_pass(self, forward_buffers, training_pass=True):
        # prepare
        _h = self.handler
        WX, W_bias = forward_buffers.parameters
        input = forward_buffers.inputs.default
        output = forward_buffers.outputs.default
        Ha = forward_buffers.internals.Ha

        # reshape
        t, b, f = input.shape
        flat_input = _h.reshape(input, (t * b, f))
        flat_Ha = _h.reshape(Ha, (t * b, self.out_shapes['default'][2]))

        # calculate outputs
        _h.dot_mm(flat_input, WX, flat_Ha)
        _h.add_mv(flat_Ha, W_bias, flat_Ha)
        self.act_func(Ha, output)

    def backward_pass(self, forward_buffers, backward_buffers):

        # prepare
        _h = self.handler
        WX, W_bias = forward_buffers.parameters
        dWX, dW_bias = backward_buffers.parameters
        input_buffer = forward_buffers.inputs.default
        output_buffer = forward_buffers.outputs.default
        in_delta_buffer = backward_buffers.inputs.default
        out_delta_buffer = backward_buffers.outputs.default
        Ha = forward_buffers.internals.Ha
        dHa = backward_buffers.internals.Ha

        # reshape
        t, b, f = input_buffer.shape
        flat_input = _h.reshape(input_buffer, (t * b, f))
        flat_dHa = _h.reshape(dHa, (t * b, self.out_shapes['default'][2]))
        flat_in_delta_buffer = _h.reshape(in_delta_buffer, (t * b, f))

        # calculate in_deltas and gradients
        self.act_func_deriv(Ha, output_buffer, out_delta_buffer, dHa)
        _h.dot_add_mm(flat_dHa, WX, out=flat_in_delta_buffer, transb='T')
        _h.dot_mm(flat_input, flat_dHa, dWX, transa='T')
        _h.sum_t(flat_dHa, axis=0, out=dW_bias)
