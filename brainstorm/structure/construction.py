#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from brainstorm.uniquely_named import UniquelyNamed
from brainstorm.utils import (
    InvalidArchitectureError, is_valid_layer_name)


class ConstructionLayer(UniquelyNamed):
    """
    Class to realize the python interface for setting up architectures.

    It is implementing the shift operation (>>) for wiring up layers
    maintaining source_layers and sink_layers lists throughout the process.

    It also keeps track of layer_type, in- and out-sizes and kwargs for later
    instantiation of the actual layer object.
    """

    def __init__(self, layer_type, shape=None, name=None, **kwargs):
        if not is_valid_layer_name(layer_type):
            raise InvalidArchitectureError(
                "Invalid layer_type: '{}'".format(layer_type))
        if not (name is None or is_valid_layer_name(name)):
            raise InvalidArchitectureError(
                "Invalid name for layer: '{}'".format(name))

        super(ConstructionLayer, self).__init__(name or layer_type)
        self.layer_type = layer_type
        self.incoming = []
        self.outgoing = []
        self.traversing = False
        self.layer_kwargs = kwargs
        self.layer_kwargs['shape'] = shape

    def collect_connected_layers(self):
        """
        Return a set of all layers that are somehow connected to this one,
        including source layers.
        """
        connectom = set()
        new_layers = {self}
        while new_layers:
            very_new_layers = set()
            for l in new_layers:
                very_new_layers |= set(l.outgoing) | set(l.incoming)
            connectom |= new_layers
            new_layers = very_new_layers - connectom
        return connectom

    def __rshift__(self, other):
        if not isinstance(other, ConstructionLayer):
            return NotImplemented
        self.outgoing.append(other)
        other.incoming.append(self)
        self.merge_scopes(other)
        return other

    def __repr__(self):
        return "<ConstructionLayer: {}>".format(self.name)