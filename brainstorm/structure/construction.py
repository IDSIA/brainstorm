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

    def __init__(self, layer_type, size=None, name=None, **kwargs):
        if not is_valid_layer_name(layer_type):
            raise InvalidArchitectureError(
                "Invalid layer_type: '{}'".format(layer_type))
        if not (name is None or is_valid_layer_name(name)):
            raise InvalidArchitectureError(
                "Invalid name for layer: '{}'".format(name))

        super(ConstructionLayer, self).__init__(name or layer_type)
        self.layer_type = layer_type
        self.size = size
        self.source_layers = []
        self.sink_layers = []
        self.traversing = False
        self.layer_kwargs = kwargs
        self.injectors = []

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
                very_new_layers |= set(l.sink_layers) | set(l.source_layers)
            connectom |= new_layers
            new_layers = very_new_layers - connectom
        return connectom

    def __rshift__(self, other):
        if not isinstance(other, ConstructionLayer):
            return NotImplemented
        self.sink_layers.append(other)
        other.source_layers.append(self)
        self.merge_scopes(other)
        return other

    def __repr__(self):
        return "<ConstructionLayer: {}>".format(self.name)


class ConstructionInjector(UniquelyNamed):
    """
    Used to realize the python interface for setting up injectors.

    It is implementing the shift operations (>> and <<) for wiring up with
    layers and storing them in targets_from or outputs_from resp..

    It also keeps track of injector_type, name, and kwargs for later
    instantiation of the actual injector object.
    """

    def __init__(self, injector_type, target_from=None, name=None, **kwargs):
        if not is_valid_layer_name(injector_type):
            raise InvalidArchitectureError(
                "Invalid layer_type: '{}'".format(injector_type))
        if not (name is None or is_valid_layer_name(name)):
            raise InvalidArchitectureError(
                "Invalid name for layer: '{}'".format(name))
        super(ConstructionInjector, self).__init__(name or injector_type)
        self.injector_type = injector_type
        self.target_from = target_from
        self.layer = None
        self.injector_kwargs = kwargs

    def __rlshift__(self, other):
        if not isinstance(other, ConstructionLayer):
            return NotImplemented
        if not (self.layer is None or self.layer is other):
            raise InvalidArchitectureError(
                "Is already connected to layer: '%s'" % self.layer)
        self.layer = other
        other.injectors.append(self)
        return self

    def __rrshift__(self, other):
        if not isinstance(other, ConstructionLayer):
            return NotImplemented
        if not (self.target_from is None or self.target_from is other):
            raise InvalidArchitectureError(
                "Is already receiving targets from: '%s'" % self.target_from)
        self.target_from = other
        return self

    def __repr__(self):
        return "<ConstructionInjector: {}>".format(self.name)
