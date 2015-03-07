#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from brainstorm.utils import get_inheritors


class LayerBase(object):
    """
    The base-class of all layer types defined in Python.
    """

    def __init__(self, size, in_size, sink_layers, source_layers, kwargs):
        self.in_size = in_size
        self.validate_kwargs(kwargs)
        self.kwargs = kwargs
        self.out_size = self._get_output_size(size, in_size, kwargs)
        self.sink_layers = sink_layers
        self.source_layers = source_layers
        self.handler = None

    def set_handler(self, new_handler):
        """
        A function that is called to set a new handler and then do some
        follow-up operations.
        For example, it may be used to reset activation functions.
        It may also be used to restrict the layer to certain handlers.
        """
        self.handler = new_handler

    @classmethod
    def validate_kwargs(cls, kwargs):
        assert not kwargs, "Unexpected kwargs: {}".format(list(kwargs.keys()))

    @classmethod
    def _get_output_size(cls, size, in_size, kwargs):
        return size if size is not None else in_size

    def get_parameter_structure(self):
        return []

    def forward_pass(self, parameters, input_buffer, output_buffer):
        pass

    def backward_pass(self, parameters, input_buffer, output_buffer,
                      in_delta_buffer, out_delta_buffer, gradient_buffer):
        pass


class InputLayer(LayerBase):
    """
    Special input layer type.
    """

    def __init__(self, size, in_size, sink_layers, source_layers, kwargs):
        super(InputLayer, self).__init__(size, in_size, sink_layers,
                                         source_layers, kwargs)
        assert not in_size, "InputLayer cannot have an in_size"


class NoOpLayer(LayerBase):
    """
    This layer just copies its input into its output.
    """

    def __init__(self, size, in_size, sink_layers, source_layers, kwargs):
        super(NoOpLayer, self).__init__(size, in_size, sink_layers,
                                        source_layers, kwargs)
        assert size == in_size, "For NoOpLayer in and out size must be equal"


class FeedForwardLayer(LayerBase):
    def __init__(self, size, in_size, sink_layers, source_layers, kwargs):
        super(FeedForwardLayer, self).__init__(size, in_size, sink_layers,
                                               source_layers, kwargs)
        self.act_func = None
        self.act_func_deriv = None
        self.kwargs = kwargs

    def set_handler(self, new_handler):
        self.handler = new_handler

        # Assign act_func and act_dunc_derivs
        activation_functions = {
            'sigmoid': (self.handler.sigmoid, self.handler.sigmoid_deriv),
            'tanh': (self.handler.tanh, self.handler.tanh_deriv),
            'linear': (lambda x: x, 1)
        }

        self.act_func, self.act_func_deriv = \
            activation_functions[self.kwargs.get('act_func', 'tanh')]

    @classmethod
    def validate_kwargs(cls, kwargs):
        for key in kwargs.keys():
            assert key in ['act_func'], "Unexpected kwarg: {} for " \
                                        "FeedForwardLayer".format(key)

    def get_parameter_structure(self):
        return [
            ('W', (self.in_size, self.out_size)),
            ('b', self.out_size)
        ]

    def forward_pass(self, parameters, input_buffer, output_buffer):
        W, b = parameters['W'], parameters['b']
        for t in range(input_buffer.shape[0]):
            output_buffer[t, :] = self.act_func(
                self.handler.add(self.handler.dot(input_buffer[t], W), b))

    def backward_pass(self, parameters, input_buffer, output_buffer,
                      in_delta_buffer, out_delta_buffer, gradient_buffer):
        W = parameters.W
        for t in range(input_buffer.shape[0]):
            d_z = self.handler.elem_mult(self.act_func_deriv(output_buffer[t]),
                                         out_delta_buffer[t])
            in_delta_buffer[t, :] = self.handler.dot(d_z, W.T)

        dW, db = gradient_buffer
        for t in range(input_buffer.shape[0]):
            dz = self.handler.elem_mult(self.act_func_deriv(output_buffer[t]),
                                        out_delta_buffer[t])
            dW += self.handler.dot(input_buffer[t].T, dz)
            db += self.handler.sum(dz, axis=0)


def get_layer_class_from_typename(typename):
    layer_classes = get_inheritors(LayerBase)
    for layer_class in layer_classes:
        if typename == layer_class.__name__:
            return layer_class
    else:
        raise TypeError("Layer-type '{}' unknown!".format(typename))
