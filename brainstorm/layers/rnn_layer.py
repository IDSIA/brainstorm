#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from collections import OrderedDict
from brainstorm.utils import LayerValidationError
from brainstorm.layers.base_layer import LayerBaseImpl
from structure.shapes import ShapeTemplate


class RnnLayerImpl(LayerBaseImpl):
    expected_kwargs = {'size', 'activation_function'}

    def __init__(self, name, in_shapes, incoming_connections,
                 outgoing_connections, **kwargs):
        super(RnnLayerImpl, self).__init__(
            name, in_shapes, incoming_connections, outgoing_connections,
            **kwargs)
        self.act_func = None
        self.act_func_deriv = None
        self.kwargs = kwargs

    def set_handler(self, new_handler):
        super(RnnLayerImpl, self).set_handler(new_handler)

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
        out_size = self.out_shapes['default'].feature_size

        parameters = OrderedDict()
        parameters['W'] = ShapeTemplate(in_size, out_size)
        parameters['R'] = ShapeTemplate(out_size, out_size)
        parameters['bias'] = ShapeTemplate(out_size)
        return parameters

    def get_internal_structure(self):
        out_size = self.out_shapes['default'].feature_size

        internals = OrderedDict()
        internals['Ha'] = ShapeTemplate('T', 'B', out_size)
        return internals

    def _get_output_shapes(self):
        s = self.kwargs.get('size', self.in_shapes['default'].feature_size)
        if not isinstance(s, int):
            raise LayerValidationError('size must be int but was {}'.format(s))

        return {'default': ShapeTemplate('T', 'B', s, context_size=1)}

    def forward_pass(self, forward_buffers, training_pass=True):
        # prepare
        _h = self.handler
        W, R, bias = forward_buffers.parameters
        inputs = forward_buffers.inputs.default
        outputs = forward_buffers.outputs.default
        Ha = forward_buffers.internals.Ha

        for t in range(inputs.shape[0]):
            # calculate outputs
            _h.dot_mm(inputs[t], W, Ha[t])
            _h.dot_mm(outputs[t], R, Ha[t])  # outputs has a time offset of 1
            _h.add_mv(Ha[t], bias, Ha[t])
            self.act_func(Ha[t], outputs[t+1])

    def backward_pass(self, forward_buffers, backward_buffers):
        # TODO: IMPLEMENT
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
