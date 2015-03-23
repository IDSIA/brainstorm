#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from brainstorm.utils import get_inheritors, LayerValidationError


def get_layer_class_from_typename(typename):
    layer_classes = get_inheritors(LayerBase)
    for layer_class in layer_classes:
        if typename == layer_class.__name__:
            return layer_class
    else:
        raise TypeError("Layer-type '{}' unknown!".format(typename))


def match_shape_template(shape, template):
    if len(shape) != len(template):
        return False
    for s, t in zip(shape, template):
        if s != t and t != 'F':
            return False
    return True


class LayerBase(object):
    """
    The base-class of all layer types defined in Python.

    Each layer has a set of named sinks (inputs) and sources (outputs).
    """

    inputs = {'default': ('T', 'B', 'F')}
    """Names and shape-templates for all inputs of this layer"""

    outputs = {'default': ('T', 'B', 'F')}
    """Names and shape-templates for all outputs of this layer"""

    expected_kwargs = {'shape'}
    """Set of all kwargs that this layer accepts"""

    def __init__(self, name, in_shapes, incoming_connections,
                 outgoing_connections, **kwargs):
        """
        :param in_shapes: A dictionary of input shapes for all named sinks
        :type in_shapes: dict[str, tuple[int]]
        :param outgoing_connections: Set of all outgoing connections
        :type outgoing_connections: set[Connection]
        :param incoming_connections: Set of all incoming connections
        :type incoming_connections: set[Connection]
        :param kwargs: all further parameters for this layer
        """
        self.name = name
        """ The name of this layer as specified in the architecture"""

        self.kwargs = kwargs
        """ Additional parameters for this layer"""

        self.in_shapes = in_shapes
        """ Dictionary of shape tuples for every sink (input). """

        self.incoming = incoming_connections
        """ List of incoming connections """

        self.outgoing = outgoing_connections
        """ List of outgoing connections """

        self.out_shapes = self._get_output_shapes()
        """ Dictionary of shape tuples for every source (output). """

        self.handler = None

        self._validate_kwargs()
        self._validate_in_shapes()
        self._validate_out_shapes()
        self._validate_connections()

    def set_handler(self, new_handler):
        """Set the handler of this layer to a new one.

        This function only sets the handler, but other Layers might extend it
        to perform some follow-up operations.
        For example, it may be used to reset activation functions.
        It may also be used to restrict the layer to certain handlers.
        """
        self.handler = new_handler

    def get_parameter_structure(self):
        """Return a dictionary mapping parameter names to shapes.

        :return: list of parameter buffers each with a name and a shape
        :rtype: dict[str, tuple[int]]
        """
        return {}

    def get_internal_structure(self):
        """Return a dictionary mapping internal-state names to shape templates.

        :return: list internal state buffers each with a name and respective
                 *feature* shape
        :rtype: dict[str, tuple]
        """
        return {}

    def forward_pass(self, forward_buffers):
        pass

    def backward_pass(self, forward_buffers, backward_buffers):
        pass

    def _validate_kwargs(self):
        """Ensure self.kwargs are all sound.

        Raise LayerValidationError otherwise."""
        unexpected_kwargs = set(self.kwargs) - self.expected_kwargs
        if unexpected_kwargs:
            raise LayerValidationError("{}: Unexpected kwargs: {}".format(
                self.name, unexpected_kwargs))

    def _validate_in_shapes(self):
        """Ensure self.in_shapes are all valid.

         Raise LayerValidationError otherwise."""
        for input_name, in_shape in self.in_shapes.items():
            if input_name not in self.inputs:
                raise LayerValidationError(
                    'Invalid in_shapes. {} has no input named "{}". '
                    'Choices are: {}'.format(self.name, input_name,
                                             self.inputs))

            if not match_shape_template(in_shape, self.inputs[input_name]):
                raise LayerValidationError(
                    "{}: in_shape ({}) for {} doesn't match shape-template {}"
                    .format(self.name, in_shape, input_name,
                            self.inputs[input_name])
                )

    def _validate_out_shapes(self):
        """Ensure self.out_shapes are all valid.

        Raise LayerValidationError otherwise."""
        for output_name, out_shape in self.out_shapes.items():
            if output_name not in self.outputs:
                raise LayerValidationError(
                    'Invalid out_shapes. {} has no output named "{}". '
                    'Choices are: {}'.format(self.name, output_name,
                                              self.inputs))

            if not match_shape_template(out_shape, self.outputs[output_name]):
                raise LayerValidationError(
                    "{}: out_shape ({}) for {} doesn't match shape-template {}"
                    .format(self.name, out_shape, output_name,
                            self.outputs[output_name])
                )

    def _validate_connections(self):
        """
        Ensure all connections from self.incoming and self.outgoing are valid.

        Raise LayerValidationError otherwise.
        """
        for in_c in self.incoming:
            if in_c.input_name not in self.inputs:
                raise LayerValidationError(
                    '{}: Invalid incoming connection ({}). Layer has no sink '
                    'named "{}"'.format(self.name, in_c, in_c.sink_name))

        for out_c in self.outgoing:
            if out_c.output_name not in self.outputs:
                raise LayerValidationError(
                    '{}: Invalid incoming connection ({}). Layer has no output'
                    ' named "{}"'.format(self.name, out_c,
                                         out_c.output_name))

    def _get_output_shapes(self):
        """ Determines the output-shape of this layer.

        Default behaviour is to look for 'shape' in kwargs. If that is not
        found try to use 'default' in_shape.

        Should be overridden by derived classes to customize this behaviour
        """
        s = self.kwargs.get('shape', self.in_shapes.get('default'))
        if s is None:
            return {'default': ('T', 'B', 0)}
        elif isinstance(s, (tuple, list)):
            return {'default': ('T', 'B') + tuple(s)}
        else:
            return {'default': ('T', 'B', s)}