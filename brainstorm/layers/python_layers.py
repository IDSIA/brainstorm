#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from brainstorm.layers.base_layer import LayerBase


class DataLayer(LayerBase):
    """
    Special input layer type, that provides access to external data.

    The ``data_name`` kwarg specifies which external data can be accessed
    through this layer. It defaults to 'input_data'
    """
    expected_kwargs = {'shape', 'data_name'}
    sink_names = []

    def __init__(self, in_shapes, incoming_connections, outgoing_connections,
                 **kwargs):
        super(DataLayer, self).__init__(in_shapes, incoming_connections,
                                        outgoing_connections, **kwargs)
        self.data_name = kwargs.get('data_name', 'input_data')


class NoOpLayer(LayerBase):
    """
    This layer just copies its input into its output.
    """

    def __init__(self, in_shapes, incoming_connections, outgoing_connections,
                 **kwargs):
        super(NoOpLayer, self).__init__(in_shapes, incoming_connections,
                                        outgoing_connections, **kwargs)

        if self.out_shapes != self.in_shapes:
            raise ValueError("For NoOpLayer in and out shapes must be equal, "
                             "but {} != {}".format(self.in_shapes['default'],
                                                   self.out_shapes['default']))

    def forward_pass(self, forward_buffers):
        self.handler.copy_to(forward_buffers.inputs.default,
                             forward_buffers.outputs.default)

    def backward_pass(self, forward_buffers, backward_buffers):
        # TODO: implement and use an add_into method instead
        self.handler.add(backward_buffers.outputs.default,
                         backward_buffers.inputs.default,
                         out=backward_buffers.inputs.default)


class FeedForwardLayer(LayerBase):
    expected_kwargs = {'shape', 'activation_function'}

    def __init__(self, in_shapes, incoming_connections, outgoing_connections,
                 **kwargs):
        super(FeedForwardLayer, self).__init__(in_shapes, incoming_connections,
                                               outgoing_connections, **kwargs)
        self.act_func = None
        self.act_func_deriv = None
        self.kwargs = kwargs

    def set_handler(self, new_handler):
        self.handler = new_handler

        # Assign act_func and act_dunc_derivs
        activation_functions = {
            'sigmoid': (self.handler.sigmoid, self.handler.sigmoid_deriv),
            'tanh': (self.handler.tanh, self.handler.tanh_deriv),
            'linear': (lambda x, y: self.handler.copy_to(y, x),
                       lambda x, y, dy, dx: self.handler.copy_to(dx, dy)),
            'rel': (self.handler.rel, self.handler.rel_deriv)
        }

        self.act_func, self.act_func_deriv = activation_functions[
            self.kwargs.get('activation_function', 'tanh')]

    def get_parameter_structure(self):
        return [
            ('W', (self.in_shapes['default'][0],
                   self.out_shapes['default'][0])),
            ('b', self.out_shapes['default'][0])
        ]

    def get_internal_structure(self):
        return [
            ('Ha', self.out_shapes['default'])
        ]

    def forward_pass(self, forward_buffers):
        # prepare
        H = self.handler
        WX, W_bias = forward_buffers.parameters
        input_buffer = forward_buffers.inputs.default
        output_buffer = forward_buffers.outputs.default
        Ha = forward_buffers.internal.Ha

        # reshape
        t, b, f = input_buffer.shape
        flat_input = H.reshape(input_buffer, (t * b, f))
        flat_output = H.reshape(output_buffer,
                                (t * b, self.out_shapes['default'][0]))

        # calculate outputs
        H.dot(flat_input, WX, flat_output)
        H.add_mv(flat_output, W_bias, flat_output)
        self.act_func(flat_output, flat_output)

    def backward_pass(self, forward_buffers, backward_buffers):

        # prepare
        H = self.handler
        WX, W_bias = forward_buffers.parameters
        dWX, dW_bias = backward_buffers.parameters
        input_buffer = forward_buffers.inputs.default
        output_buffer = forward_buffers.outputs.default
        in_delta_buffer = backward_buffers.inputs.default
        out_delta_buffer = backward_buffers.outputs.default
        dZ = H.zeros(output_buffer.shape)
        Ha = forward_buffers.internal.Ha
        dHa = backward_buffers.internal.Ha

        # reshape
        t, b, f = input_buffer.shape
        flat_input = H.reshape(input_buffer, (t * b, f))
        flat_dZ = H.reshape(dZ, (t * b, self.out_shapes['default'][0]))
        flat_in_delta_buffer = H.reshape(in_delta_buffer, (t * b, f))

        # calculate in deltas and gradients
        # TODO: Replace first argument in following call with the fwd state
        # since some activation functions might need it
        self.act_func_deriv(self.handler.EMPTY, output_buffer,
                            out_delta_buffer, dZ)
        H.dot_add(flat_dZ, WX, out=flat_in_delta_buffer, transb='T')
        H.dot(flat_input, flat_dZ, dWX, transa='T')
        H.sum(flat_dZ, axis=0, out=dW_bias)
