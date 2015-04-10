#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from brainstorm.utils import get_inheritors, LayerValidationError, get_by_path
from brainstorm.structure.shapes import ShapeTemplate


def get_layer_class_from_typename(typename):
    layer_classes = get_inheritors(LayerBaseImpl)
    for layer_class in layer_classes:
        if typename == layer_class.__name__:
            return layer_class
    else:
        raise TypeError("Layer-type '{}' unknown!".format(typename))


class LayerBaseImpl(object):
    """
    The base-class of all layer types defined in Python.

    Each layer has a set of named sinks (inputs) and sources (outputs).
    """

    inputs = {'default': ShapeTemplate('T', 'B', 'F')}
    """Names and shape-templates for all inputs of this layer"""

    outputs = {'default': ShapeTemplate('T', 'B', 'F')}
    """Names and shape-templates for all outputs of this layer"""

    expected_kwargs = {}
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
        """Return a OrderedDict mapping parameter names to ShapeTemplates.

        :return: OrderedDict describing parameter buffers
        :rtype: OrderedDict[str, ShapeTemplate]
        """
        return {}

    def get_internal_structure(self):
        """Return a OrderedDict internal-state names to to ShapeTemplate.

        :return: OrderedDict describing internals
        :rtype: OrderedDict[str, ShapeTemplate]
        """
        return {}

    def forward_pass(self, forward_buffers, training_pass=True):
        pass

    def backward_pass(self, forward_buffers, backward_buffers):
        pass

    def get_shape(self, path):
        category, _, subpath = path.partition('.')
        categories = {'parameters', 'inputs', 'outputs', 'internals'}
        if category not in categories:
            raise ValueError("Category '{}' for path '{}' not found. Choices "
                             "are {}".format(category, path, categories))
        if category == 'parameters':
            parameters = self.get_parameter_structure()
            return get_by_path(parameters, subpath)['@shape']
        if category == 'internals':
            internals = self.get_internal_structure()
            return get_by_path(internals, subpath)['@shape']
        if category == 'inputs':
            return get_by_path(self.in_shapes, subpath)
        if category == 'outputs':
            return get_by_path(self.out_shapes, subpath)

    def _validate_kwargs(self):
        """Ensure self.kwargs are all sound.

        Raise LayerValidationError otherwise."""
        unexpected_kwargs = set(self.kwargs) - set(self.expected_kwargs)
        if unexpected_kwargs:
            raise LayerValidationError("{}: Unexpected kwargs: {}".format(
                self.name, unexpected_kwargs))

    def _validate_in_shapes(self):
        """Ensure self.in_shapes are all valid.

         Raise LayerValidationError otherwise."""
        in_shape_names = set(self.in_shapes.keys())
        input_names = set(self.inputs.keys())

        if not in_shape_names.issubset(input_names):
            raise LayerValidationError(
                'Invalid in_shapes. {} has no input(s) named "{}". Choices '
                'are: {}'.format(self.name, in_shape_names - input_names,
                                 input_names))

        if not input_names.issubset(in_shape_names):
            raise LayerValidationError(
                '{}: All inputs need to be connected. Missing {}.'
                .format(self.name, input_names - in_shape_names))

        for input_name, in_shape in self.in_shapes.items():
            if self.inputs[input_name].matches(in_shape):
                raise LayerValidationError(
                    "{}: in_shape ({}) for {} doesn't match shape-template {}"
                    .format(self.name, in_shape, input_name,
                            self.inputs[input_name]))

    def _validate_out_shapes(self):
        """Ensure self.out_shapes are all valid.

        Raise LayerValidationError otherwise."""
        out_shape_names = set(self.out_shapes.keys())
        output_names = set(self.outputs.keys())

        if not out_shape_names.issubset(output_names):
            raise LayerValidationError(
                'Invalid out_shapes. {} has no output(s) named "{}". Choices '
                'are: {}'.format(self.name, out_shape_names - output_names,
                                 output_names))

        for output_name, out_shape in self.out_shapes.items():
            if not self.outputs[output_name].matches(out_shape):
                raise LayerValidationError(
                    "{}: out_shape ({}) for {} doesn't match shape-template {}"
                    .format(self.name, out_shape, output_name,
                            self.outputs[output_name]))

    def _validate_connections(self):
        """
        Ensure all connections from self.incoming and self.outgoing are valid.

        Raise LayerValidationError otherwise.
        """
        for in_c in self.incoming:
            if in_c.input_name not in self.in_shapes:
                raise LayerValidationError(
                    '{}: Invalid incoming connection ({}). Layer has no input '
                    'named "{}"'.format(self.name, in_c, in_c.sink_name))

        for out_c in self.outgoing:
            if out_c.output_name.startswith('..'):
                category, _, substruct = out_c.output_name[2:].partition('.')
                if category not in {'parameters', 'internals'}:
                    raise LayerValidationError(
                        "{}: Invalid outgoing connection ({}). Category '{}' "
                        "is not allowed/does not exist. Choices are "
                        "['parameters', 'internals']".format(self.name, out_c,
                                                             category))
                if category == 'parameters':
                    parameters = self.get_parameter_structure()
                    if substruct not in parameters:
                        raise LayerValidationError(
                            "{}: Invalid outgoing connection ({}). Parameter"
                            " '{}' does not exist. Choices are {}".format(
                                self.name, out_c, list(parameters.keys())))

                if category == 'internals':
                    internals = self.get_internal_structure()
                    if substruct not in internals:
                        raise LayerValidationError(
                            "{}: Invalid outgoing connection ({}). Internal"
                            " '{}' does not exist. Choices are {}".format(
                                self.name, out_c, list(internals.keys())))

            elif out_c.output_name not in self.out_shapes:
                raise LayerValidationError(
                    '{}: Invalid outgoing connection ({}). Layer has no output'
                    ' named "{}"'.format(self.name, out_c,
                                         out_c.output_name))

    def _get_output_shapes(self):
        """ Determines the output-shape of this layer.

        Default behaviour is to look for 'shape' in kwargs. If that is not
        found try to use 'default' in_shape.

        Should be overridden by derived classes to customize this behaviour
        """
        raise NotImplementedError('LayerImplementations need to implement '
                                  'the _get_output_shapes method.')
