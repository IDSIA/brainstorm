#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

import six
from brainstorm.utils import NetworkValidationError, is_valid_layer_name


class UniquelyNamed(object):
    """
    An object that maintains a scope of names and ensures that its name is
    unique within that scope by appending an appropriate index.

    If there are no collisions then its name is the same as the given basename.
    If there are multiple objects with the same name in the scope, then
    its name is the basename + _index where index is a number given according
    to the order in which the objects where created.
    """

    global_counter = 0

    def __init__(self, basename):
        self._basename = basename
        self.scope = {basename: [self]}
        self.creation_id = UniquelyNamed.global_counter
        UniquelyNamed.global_counter += 1

    def merge_scopes(self, other):
        new_scope = self.scope
        for name, scoped_names in other.scope.items():
            if name not in self.scope:
                new_scope[name] = []
            new_scope[name] = sorted(set(self.scope[name] + scoped_names),
                                     key=lambda x: x.creation_id)
        for n in new_scope:
            for sn in new_scope[n]:
                sn.scope = new_scope

    @property
    def name(self):
        if len(self.scope[self._basename]) == 1:
            return self._basename

        i = 1
        for un in self.scope[self._basename]:
            name = "{}_{}".format(self._basename, i)
            # see if this derived name is already taken
            # increase the index if need be
            while name in self.scope and len(self.scope[name]) == 1:
                i += 1
                name = "{}_{}".format(self._basename, i)
            if un is self:
                return name
            i += 1


class LayerDetails(UniquelyNamed):
    """
    Contains all details about a layer at construction time.

    This information is later used to generate an architecture, from which the
    actual layers are instantiated and combined into a network.
    """
    def __init__(self, layer_type, name=None, **kwargs):
        if not is_valid_layer_name(layer_type):
            raise NetworkValidationError(
                "Invalid layer_type: '{}'".format(layer_type))
        if not (name is None or is_valid_layer_name(name)):
            raise NetworkValidationError(
                "Invalid name for layer: '{}'".format(name))
        super(LayerDetails, self).__init__(name or layer_type)

        self.layer_type = layer_type
        """The type this layer should have when later being instantiated."""

        self.incoming = []
        """A list of all incoming connections, including input/output names.

        Each entry of the list has the form:
        (incoming_layer, output_name, input_name)
        and the type:
        tuple[LayerDetails, str, str]
        """

        self.outgoing = []
        """A list of all outgoing connections, including input/output names.

        Each entry of the list has the form:
        (output_name, input_name, outgoing_layer)
        and the type:
        tuple[str, str, LayerDetails]
        """

        self.layer_kwargs = kwargs
        """Dictionary of additional parameters for this layer"""

        self._traversing = False

    def collect_connected_layers(self):
        """Return a set of all layers that are somehow connected to this"""
        connectom = set()
        new_layers = {self}
        while new_layers:
            very_new_layers = set()
            for l in new_layers:
                very_new_layers |= {o[2] for o in l.outgoing}
                very_new_layers |= {i[0] for i in l.incoming}
            connectom |= new_layers
            new_layers = very_new_layers - connectom
        return connectom

    def __repr__(self):
        return "<Layer: {}>".format(self.name)


class ConstructionWrapper(object):
    """
    Class to realize the python interface for setting up architectures.

    Internally it keeps a LayerDetails object, which is updated with
    connections.

    It also implements the shift operation (>>) for wiring up layers, and the
    subtraction operation (-) for specifying named inputs or outputs.
    """

    @classmethod
    def create(cls, layer_type, name=None, **kwargs):
        if isinstance(layer_type, six.string_types):
            layer_type_name = layer_type
        else:
            layer_type_name = layer_type.__name__

        if not layer_type_name.endswith('LayerImpl'):
            raise NetworkValidationError("{} should end with 'LayerImpl'"
                                         .format(layer_type_name))
        layer_type_name = layer_type_name[:-9]

        details = LayerDetails(layer_type_name, name=name, **kwargs)
        return ConstructionWrapper(details)

    def __init__(self, layer_details, input_name='default',
                 output_name='default'):
        self.layer = layer_details
        self.input_name = input_name
        self.output_name = output_name

    def __rshift__(self, other):
        if not isinstance(other, ConstructionWrapper):
            return NotImplemented
        self.layer.outgoing.append((self.output_name, other.input_name,
                                    other.layer))
        other.layer.incoming.append((self.layer, self.output_name,
                                     other.input_name))
        self.layer.merge_scopes(other.layer)
        return other

    def __sub__(self, other):
        if not isinstance(other, six.string_types):
            return NotImplemented
        return ConstructionWrapper(self.layer, output_name=other,
                                   input_name=self.input_name)

    def __rsub__(self, other):
        if not isinstance(other, six.string_types):
            return NotImplemented
        return ConstructionWrapper(self.layer, input_name=other,
                                   output_name=self.output_name)

    def __repr__(self):
        return "<Layer: '{}' - {} - '{}'>".format(self.input_name,
                                                  self.layer.name,
                                                  self.output_name)
