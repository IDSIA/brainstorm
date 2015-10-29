#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

from collections import OrderedDict

from brainstorm.utils import LayerValidationError, get_by_path, get_inheritors


def get_layer_class_from_typename(typename):
    layer_classes = get_inheritors(Layer)
    for layer_class in layer_classes:
        if typename == layer_class.__name__:
            return layer_class
    else:
        raise TypeError("Layer-type '{}' unknown!".format(typename))


class Layer(object):
    """
    The base class of all layer types defined in Python.

    Each layer has a set of named inputs and outputs.

    Args:
        in_shapes (dict[str, BufferStructure]):
             A dictionary of input buffer structures for all named inputs
        outgoing_connections (set[Connections]):
            Set of all outgoing connections
        incoming_connections (set[Connections]):
            Set of all incoming connections
        kwargs (dict):
            any further arguments passed to this layer

    Attributes:
        name (str):
            The name of this layer as specified in the architecture
        kwargs (dict):
            Additional options or hyperparameters for this layer
        in_shapes (OrderedDict[str, BufferStructure]):
            Dictionary of `BufferStructure`s for each input.
        out_shapes (OrderedDict[str, BufferStructure):
            Dictionary of `BufferStructure`s for each output.
        parameter_shapes (OrderedDict[str, BufferStructure):
            Dictionary of `BufferStructure`s for each parameter.
        internal_shapes (OrderedDict[str, BufferStructure):
            Dictionary of `BufferStructure`s for each internal buffer.
        incoming (list):
            List of incoming connections
        outgoing (list):
            List of outgoing connections
        handler (brainstorm.handlers.base_handler.Handler):
            The handler currently responsible for this layer
    """
    expected_kwargs = {}
    """Set of all kwargs that this layer accepts"""

    expected_inputs = {}
    """Names and shape-templates for all inputs of this layer"""

    computes_no_input_deltas_for = ()
    computes_no_gradients_for = ()
    takes_no_output_deltas_from = ()

    def __init__(self, name, in_shapes, incoming_connections,
                 outgoing_connections, **kwargs):
        self.name = name
        self.kwargs = kwargs
        self.in_shapes = OrderedDict()
        # make sure in_shapes are an ordered dict
        for key in sorted(in_shapes.keys()):
            self.in_shapes[key] = in_shapes[key]

        self.incoming = incoming_connections
        self.outgoing = outgoing_connections
        self.handler = None
        self._validate_kwargs()
        self._validate_in_shapes()
        out, param, intern = self.setup(self.kwargs, self.in_shapes)
        self.out_shapes = out
        self.parameter_shapes = param
        self.internal_shapes = intern
        self._validate_connections()

    def setup(self, kwargs, in_shapes):
        """
        Setup the layer and the buffer structures it uses.

        Each layer implementation needs to override this function.

        Args:
            kwargs (dict):
                any keyword arguments passed to this layer during construction
            in_shapes (dict[str, BufferStructure]):
                A dictionary of input buffer structures for all inputs

        Returns:
            (tuple): tuple containing:
                out_shapes (dict): A dictionary of buffer structures for all
                                   outputs of this layer
                parameter_shapes (dict): A dictionary of buffer structures for
                                         all parameters of this layer
                internal_shapes (dict): A dictionary of buffer structures for
                                        all internal buffers of this layer
        """
        raise NotImplementedError('LayerImplementations need to implement '
                                  'the setup() method.')

    def set_handler(self, new_handler):
        """Set the handler of this layer to a new one.

        This function only sets the handler, but other Layers might extend it
        to perform some follow-up operations.
        For example, it may be used to reset activation functions.
        It may also be used to restrict the layer to certain handlers.
        """
        self.handler = new_handler

    def forward_pass(self, buffers, training_pass=True):
        pass

    def backward_pass(self, buffers):
        pass

    def get_shape(self, path):
        category, _, subpath = path.partition('.')
        categories = {'parameters', 'inputs', 'outputs', 'internals'}
        if category not in categories:
            raise ValueError("Category '{}' for path '{}' not found. Choices "
                             "are {}".format(category, path, categories))
        category_shapes = {
            'parameters': self.parameter_shapes,
            'internals': self.internal_shapes,
            'inputs': self.in_shapes,
            'outputs': self.out_shapes
        }
        return get_by_path(category_shapes[category], subpath)

    def _validate_kwargs(self):
        """Ensure self.kwargs are all sound.

        Raises:
            LayerValidationError: if there are unexpected kwargs."""
        unexpected_kwargs = set(self.kwargs) - set(self.expected_kwargs)
        if unexpected_kwargs:
            raise LayerValidationError("{}: Unexpected kwargs: {}".format(
                self.name, unexpected_kwargs))

    def _validate_in_shapes(self):
        """Ensure all in_shapes are valid by comparing to `expected_inputs`.

        Raises:
            LayerValidationError: if there are unrecognized inputs, missing
                                  inputs or inputs that don't match the
                                  `StructureTemplate` from `expected_inputs`.
        """
        in_shape_names = set(self.in_shapes.keys())
        input_names = set(self.expected_inputs.keys())

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
            if not self.expected_inputs[input_name].matches(in_shape):
                raise LayerValidationError(
                    "{}: in_shape ({}) for {} doesn't match StructureTemplate "
                    "{}".format(self.name, in_shape, input_name,
                                self.expected_inputs[input_name]))

    def _validate_connections(self):
        """
        Ensure all incoming and outgoing connections are valid.

        Raises:
            LayerValidationError: if there is any:
                * incoming connection to a non-existent input
                * outgoing connection from a non-existent output, parameter
                  or internal buffer
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
                    parameters = self.parameter_shapes
                    if substruct not in parameters:
                        raise LayerValidationError(
                            "{}: Invalid outgoing connection ({}). Parameter"
                            " '{}' does not exist. Choices are {}".format(
                                self.name, out_c, substruct,
                                list(parameters.keys())))

                if category == 'internals':
                    internals = self.internal_shapes
                    if substruct not in internals:
                        raise LayerValidationError(
                            "{}: Invalid outgoing connection ({}). Internal"
                            " '{}' does not exist. Choices are {}".format(
                                self.name, out_c, substruct,
                                list(internals.keys())))

            elif out_c.output_name not in self.out_shapes:
                raise LayerValidationError(
                    '{}: Invalid outgoing connection ({}). Layer has no output'
                    ' named "{}". Choices are: {}'.format(
                        self.name, out_c, out_c.output_name,
                        list(self.out_shapes.keys())))
