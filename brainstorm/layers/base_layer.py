#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from brainstorm.utils import get_inheritors


def get_layer_class_from_typename(typename):
    layer_classes = get_inheritors(LayerBase)
    for layer_class in layer_classes:
        if typename == layer_class.__name__:
            return layer_class
    else:
        raise TypeError("Layer-type '{}' unknown!".format(typename))


class LayerBase(object):
    """
    The base-class of all layer types defined in Python.

    Each layer has a set of named sinks (inputs) and sources (outputs).
    """

    input_names = ['default']
    output_names = ['default']
    expected_kwargs = {'shape'}

    def __init__(self, in_shapes, incoming_connections, outgoing_connections,
                 **kwargs):
        """
        :param in_shapes: A dictionary of input shapes for all named sinks
        :type in_shapes: dict[str, tuple[int]]
        :param outgoing_connections: Set of all outgoing connections
        :type outgoing_connections: set[Connection]
        :param incoming_connections: Set of all incoming connections
        :type incoming_connections: set[Connection]
        :param kwargs: all further parameters for this layer
        """
        self._validate_kwargs(kwargs)
        self.kwargs = kwargs
        """ Additional parameters for this layer"""

        self._validate_in_shapes(in_shapes)
        self.in_shapes = in_shapes
        """ Dictionary of shape tuples for every sink (input). """

        self.out_shapes = self._get_output_shapes(in_shapes, kwargs)
        self._validate_out_shapes(self.out_shapes)
        """ Dictionary of shape tuples for every source (output). """

        self._validate_connections(incoming_connections, outgoing_connections,
                                   kwargs)
        self.incoming = incoming_connections
        """ List of incoming connections """

        self.outgoing = outgoing_connections
        """ List of outgoing connections """

        self.handler = None

    def set_handler(self, new_handler):
        """
        A function that is called to set a new handler and then do some
        follow-up operations.
        For example, it may be used to reset activation functions.
        It may also be used to restrict the layer to certain handlers.
        """
        self.handler = new_handler

    def get_parameter_structure(self):
        """
        :return: list of parameter buffers each with a name and a shape
        :rtype: list[dict]
        """
        return {}

    def get_internal_structure(self):
        """
        :return: list internal state buffers each with a name and respective
                 *feature* shape
        :rtype: list[dict]
        """
        return {}

    def forward_pass(self, forward_buffers):
        pass

    def backward_pass(self, forward_buffers, backward_buffers):
        pass

    @classmethod
    def _validate_kwargs(cls, kwargs):
        unexpected_kwargs = set(kwargs) - cls.expected_kwargs
        if unexpected_kwargs:
            raise ValueError("{}: Unexpected kwargs: {}".format(
                cls.__name__, unexpected_kwargs))

    @classmethod
    def _validate_in_shapes(cls, in_shapes):
        for input_name in in_shapes:
            if input_name not in cls.input_names:
                raise ValueError('{}: Invalid in_shapes.'
                                 'Layer has no input named "{}". Choices are:'
                                 ' {}'.format(cls.__name__, input_name,
                                              cls.input_names))

    @classmethod
    def _validate_out_shapes(cls, out_shapes):
        for output_name in out_shapes:
            if output_name not in cls.output_names:
                raise ValueError('{}: Invalid out_shapes.'
                                 'Layer has no output named "{}". Choices are:'
                                 ' {}'.format(cls.__name__, output_name,
                                              cls.input_names))

    @classmethod
    def _validate_connections(cls, incoming_connections, outgoing_connections,
                              kwargs):
        for in_c in incoming_connections:
            if in_c.input_name not in cls.input_names:
                raise ValueError(
                    '{}: Invalid incoming connection ({}). Layer has no sink '
                    'named "{}"'.format(cls.__name__, in_c, in_c.sink_name))

        for out_c in outgoing_connections:
            if out_c.output_name not in cls.output_names:
                raise ValueError(
                    '{}: Invalid incoming connection ({}). Layer has no output '
                    'named "{}"'.format(cls.__name__, out_c,
                                        out_c.output_name))

    @classmethod
    def _get_output_shapes(cls, in_shapes, kwargs):
        """ Determines the output-shape of this layer.

        Default behaviour is to look for 'shape' in kwargs. If that is not
        found try to use 'default' in_shape.

        Should be overridden by derived classes to customize this behaviour
        """
        s = kwargs.get('shape', in_shapes.get('default'))
        if s is None:
            return {'default': (0,)}
        return {'default': tuple(s) if isinstance(s, (tuple, list)) else (s,)}
