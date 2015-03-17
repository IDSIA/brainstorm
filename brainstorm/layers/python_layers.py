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

    def forward_pass(self, parameters, input_buffers, output_buffers):
        self.handler.copy_to(input_buffers.default, output_buffers.default)

    def backward_pass(self, parameters, input_buffers, output_buffers,
                      in_delta_buffers, out_delta_buffers, gradient_buffers):
        # TODO: implement and use an add_into method instead
        self.handler.add(out_delta_buffers.default, in_delta_buffers.default,
                         out=in_delta_buffers.default)


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
            'linear': (lambda x: x, 1),
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

    def forward_pass(self, parameters, input_buffers, output_buffers):
        # prepare
        H = self.handler
        WX, W_bias = parameters

        # reshape
        t, b, f = input_buffers.default.shape
        flat_input = H.reshape(input_buffers.default, (t * b, f))
        flat_output = H.reshape(output_buffers.default,
                                (t * b, self.out_shapes['default'][0]))

        # calculate outputs
        H.dot(flat_input, WX, flat_output)
        H.add_mv(flat_output, W_bias, flat_output)
        self.act_func(flat_output, flat_output)

    def backward_pass(self, parameters, input_buffers, output_buffers,
                      in_delta_buffers, out_delta_buffers, gradient_buffers):

        # prepare
        H = self.handler
        WX, W_bias = parameters['W'], parameters['b']
        dWX, dW_bias = gradient_buffers['W'], gradient_buffers['b']
        dZ = H.zeros(output_buffers.default.shape)

        # reshape
        t, b, f = input_buffers.default.shape
        flat_input = H.reshape(input_buffers.default, (t * b, f))
        flat_dZ = H.reshape(dZ, (t * b, self.out_shapes['default'][0]))
        flat_in_delta_buffer = H.reshape(in_delta_buffers.default, (t * b, f))

        # calculate in deltas and gradients
        # TODO: Replace first argument in following call with the fwd state
        # since some activation functions might need it
        self.act_func_deriv(self.handler.EMPTY, output_buffers.default,
                            out_delta_buffers.default, dZ)
        H.dot_add(flat_dZ, WX, out=flat_in_delta_buffer, transb='T')
        H.dot(flat_input, flat_dZ, dWX, transa='T')
        H.sum(flat_dZ, axis=0, out=dW_bias)
