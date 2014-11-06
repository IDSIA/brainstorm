#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np
from collections import namedtuple
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

    @classmethod
    def validate_kwargs(cls, kwargs):
        assert not kwargs, "Unexpected kwargs: {}".format(list(kwargs.keys()))

    @classmethod
    def _get_output_size(cls, size, in_size, kwargs):
        return size if size is not None else in_size

    def get_parameter_size(self):
        return 0

    def create_param_view(self, buffer):
        assert self.get_parameter_size() == buffer.size
        return None


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
    Parameters = namedtuple('FeedForwardLayer_Parameters', ['W', 'b'])

    def __init__(self, size, in_size, sink_layers, source_layers, kwargs):
        super(FeedForwardLayer, self).__init__(size, in_size, sink_layers,
                                               source_layers, kwargs)
        self.act_func = lambda x: 1. / (1. + np.exp(-x))

    def get_parameter_size(self):
        return self.in_size * self.out_size + self.out_size

    def create_param_view(self, buffer):
        assert self.get_parameter_size() == buffer.size
        W_size = self.in_size * self.out_size
        W = buffer[:W_size].reshape(self.in_size, self.out_size)
        b = buffer[W_size:]
        return FeedForwardLayer.Parameters(W, b)

    def forward_pass(self, parameters, input_buffer, output_buffer):
        W, b = parameters
        for t in range(input_buffer.shape[0]):
            output_buffer[t, :] = self.act_func(np.dot(input_buffer[t], W) + b)


def get_layer_class_from_typename(typename):
    layer_classes = get_inheritors(LayerBase)
    for layer_class in layer_classes:
        if typename == layer_class.__name__:
            return layer_class
    else:
        raise TypeError("Layer-type '{}' unknown!".format(typename))
