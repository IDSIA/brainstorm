#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
from brainstorm.utils import get_inheritors


class LayerBase(object):
    """
    The base-class of all layer types defined in Python.
    """
    def __init__(self, size, in_size, kwargs):
        self.in_size = in_size
        self.validate_kwargs(kwargs)
        self.kwargs = kwargs
        self.out_size = self._get_output_size(size, in_size, kwargs)

    @classmethod
    def validate_kwargs(cls, kwargs):
        assert not kwargs, "Unexpected kwargs: {}".format(list(kwargs.keys()))

    @classmethod
    def _get_output_size(cls, size, in_size, kwargs):
        return size if size is not None else in_size

    def get_parameter_size(self):
        return 0


class InputLayer(LayerBase):
    """
    Special input layer type.
    """
    def __init__(self, size, in_size, kwargs):
        super(InputLayer, self).__init__(size, in_size, kwargs)
        assert not in_size, "InputLayer cannot have an in_size"


class NoOpLayer(LayerBase):
    """
    This layer just copies its input into its output.
    """
    def __init__(self, size, in_size, kwargs):
        super(NoOpLayer, self).__init__(size, in_size, kwargs)
        assert size == in_size, "For NoOpLayer in and out size must be equal"


class FeedForwardLayer(LayerBase):
    def get_parameter_size(self):
        return self.in_size * self.out_size + self.out_size


def get_layer_class_from_typename(typename):
    layer_classes = get_inheritors(LayerBase)
    for layer_class in layer_classes:
        if typename == layer_class.__name__:
            return layer_class
    else:
        raise TypeError("Layer-type '{}' unknown!".format(typename))