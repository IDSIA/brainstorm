#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from brainstorm.layers.base_layer import LayerBase


class InputLayer(LayerBase):
    """
    Special input layer type, that provides access to external data.

    The 'out_shapes' kwarg is required and specifies the names and shapes of
    all external inputs.
    """
    expected_kwargs = {'out_shapes'}
    input_names = []

    def __init__(self, in_shapes, incoming_connections, outgoing_connections,
                 **kwargs):
        super(InputLayer, self).__init__(in_shapes, incoming_connections,
                                         outgoing_connections, **kwargs)
        self.data_name = kwargs.get('data_name', 'input_data')

    @classmethod
    def _get_output_shapes(cls, in_shapes, kwargs):
        assert 'out_shapes' in kwargs, "InputLayer requires 'out_shapes'"
        return kwargs['out_shapes']

    @classmethod
    def _validate_out_shapes(cls, out_shapes):
        for output_name, shape in out_shapes.items():
            if not isinstance(shape, tuple):
                raise ValueError('out_shape entry "{}" was not a shape'
                                 .format(shape))

    @classmethod
    def _validate_connections(cls, incoming_connections, outgoing_connections,
                              kwargs):
        if incoming_connections:
            raise ValueError('InputLayer cannot have any incoming connections!'
                             '(But: {})'.format(incoming_connections))

        for out_c in outgoing_connections:
            if out_c.output_name not in kwargs['out_shapes']:
                raise ValueError(
                    '{}: Invalid incoming connection ({}). Layer has no output'
                    ' named "{}".\nChoices are {}.'.format(
                        cls.__name__, out_c, out_c.output_name,
                        kwargs['out_shapes'].keys()))


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
        self.handler.add(backward_buffers.outputs.default,
                         backward_buffers.inputs.default,
                         out=backward_buffers.inputs.default)


class FullyConnectedLayer(LayerBase):
    expected_kwargs = {'shape', 'activation_function'}

    def __init__(self, in_shapes, incoming_connections, outgoing_connections,
                 **kwargs):
        super(FullyConnectedLayer, self).__init__(in_shapes, incoming_connections,
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
        return {
            'W': {
                'shape': (self.in_shapes['default'][0],
                          self.out_shapes['default'][0]),
                'index': 0},
            'b': {
                'shape': (self.out_shapes['default'][0],),
                'index': 1}
        }

    def get_internal_structure(self):
        return {
            'Ha': {
                'shape': self.out_shapes['default'],
                'index': 0}
        }

    def forward_pass(self, forward_buffers):
        # prepare
        _h = self.handler
        WX, W_bias = forward_buffers.parameters
        input = forward_buffers.inputs.default
        output = forward_buffers.outputs.default
        Ha = forward_buffers.internals.Ha

        # reshape
        t, n, f = input.shape
        flat_input = _h.reshape(input, (t * n, f))
        flat_Ha = _h.reshape(Ha, (t * n, self.out_shapes['default'][0]))

        # calculate outputs
        _h.dot(flat_input, WX, flat_Ha)
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
        flat_dHa = _h.reshape(dHa, (t * b, self.out_shapes['default'][0]))
        flat_in_delta_buffer = _h.reshape(in_delta_buffer, (t * b, f))

        # calculate in_deltas and gradients
        self.act_func_deriv(Ha, output_buffer, out_delta_buffer, dHa)
        _h.dot_add(flat_dHa, WX, out=flat_in_delta_buffer, transb='T')
        _h.dot(flat_input, flat_dHa, dWX, transa='T')
        _h.sum(flat_dHa, axis=0, out=dW_bias)
