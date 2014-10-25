#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import re
from .uniquely_named import UniquelyNamed


PYTHON_IDENTIFIER = re.compile("^[_a-zA-Z][_a-zA-Z0-9]*$")


class InvalidArchitectureError(Exception):
    """
    Exception that is thrown if attempting to build an invalid architecture.
    (E.g. circle)
    """
    pass


class ConstructionLayer(UniquelyNamed):
    """
    Class to realize the python interface for setting up architectures.

    It is implementing the shift operation (>>) for wiring up layers
    maintaining source_layers and sink_layers lists throughout the process.

    It also keeps track of layer_type, in- and out-sizes and kwargs for later
    instantiation of the actual layer object.
    """

    def __init__(self, layer_type, size=None, name=None, **kwargs):
        assert PYTHON_IDENTIFIER.match(layer_type), \
            "Invalid layer_type: '{}'".format(layer_type)

        assert name is None or PYTHON_IDENTIFIER.match(name), \
            "Invalid layer name: '{}'".format(name)

        super(ConstructionLayer, self).__init__(name or layer_type)
        self.layer_type = layer_type
        self._size = size
        self.source_layers = []
        self.sink_layers = []
        self.traversing = False
        self.layer_kwargs = kwargs

    @property
    def size(self):
        if self._size is not None:
            return self._size

        if not self.source_layers:
            raise InvalidArchitectureError(
                "Could not determine size of {}".format(self))

        return sum(l.size for l in self.source_layers)

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

    def traverse_sink_layer_tree(self):
        """
        Recursively traverses all the sink layers of this layer.
        If there is a circle in the graph, it will raise an
        InvalidArchitectureError.
        """
        if self.traversing:
            raise InvalidArchitectureError(
                "Circle in Network at layer {}".format(self.name))
        self.traversing = True
        yield self
        for target in self.sink_layers:
            for t in target.traverse_sink_layer_tree():
                yield t
        self.traversing = False

    def assert_no_cycles(self):
        list(self.traverse_sink_layer_tree())  # raises if cycles are found

    def __rshift__(self, other):
        self.sink_layers.append(other)
        other.source_layers.append(self)
        self.assert_no_cycles()
        self.merge_scopes(other)
        return other

    def __repr__(self):
        return "<ConstructionLayer: {}>".format(self.name)
