#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from collections import OrderedDict
from copy import copy
from six import string_types
from brainstorm.utils import (InvalidArchitectureError,
                              is_valid_python_identifier)
from brainstorm.python_layers import get_layer_class_from_typename


def get_layer_description(layer):
    description = {
        '@type': layer.layer_type,
        'size': layer.size,
        'sink_layers': {l.name for l in layer.sink_layers}
    }
    if layer.layer_kwargs:
        description.update(layer.layer_kwargs)
    return description


def generate_architecture(some_layer):
    layers = some_layer.collect_connected_layers()
    arch = {layer.name: get_layer_description(layer) for layer in layers}
    return arch


def validate_architecture(architecture):
    try:
        # schema
        for name, layer in architecture.items():
            assert isinstance(name, string_types)
            assert '@type' in layer and isinstance(layer['@type'],
                                                   string_types)
            assert 'sink_layers' in layer and isinstance(layer['sink_layers'],
                                                         set)

        # no layer is called 'default'
        assert 'default' not in architecture, \
            "'default' is an invalid layer name"

        # layer naming
        for name in architecture:
            assert is_valid_python_identifier(name), \
                "Invalid layer name: '{}'".format(name)

        # all sink_layers are present
        for layer in architecture.values():
            for sink_name in layer['sink_layers']:
                assert sink_name in architecture, \
                    "Could not find sink layer '{}'".format(sink_name)

        # has InputLayer
        assert 'InputLayer' in architecture
        assert architecture['InputLayer']['@type'] == 'InputLayer'

        # has only one InputLayer
        inputs_by_type = [l for l in architecture.values()
                          if l['@type'] == 'InputLayer']
        assert len(inputs_by_type) == 1

        # no sources for InputLayer
        input_sources = [l for l in architecture.values()
                         if 'InputLayer' in l['sink_layers']]
        assert len(input_sources) == 0

        # only 1 output
        outputs = [l for l in architecture.values() if not l['sink_layers']]
        assert len(outputs) == 1
        # TODO: check if connected
        # TODO: check for cycles
    except AssertionError as e:
        raise InvalidArchitectureError(e)


def get_canonical_layer_order(architecture):
    """
    Takes a dictionary representation of an architecture and sorts it
    by (topological depth, name) and returns this canonical layer order as a
    list of names.
    """
    layer_order = []
    already_ordered_layers = set()
    while True:
        remaining_layers = [l for l in architecture.keys()
                            if l not in already_ordered_layers]
        new_layers = [
            n for n in remaining_layers
            if set(architecture[n]['sink_layers']) <= already_ordered_layers]

        if not new_layers:
            break
        layer_order += sorted(new_layers, reverse=True)
        already_ordered_layers = set(layer_order)

    remaining_layers = set(architecture.keys()) - already_ordered_layers

    assert not remaining_layers, "couldn't reach %s" % remaining_layers
    return list(reversed(layer_order))


def get_kwargs(layer):
    kwarg_ignore = {'@type', 'size', 'sink_layers', 'source_layers', 'kwargs'}
    return {k: copy(v) for k, v in layer.items() if k not in kwarg_ignore}


def get_source_layers(layer_name, architecture):
    return [n for n, l in architecture.items()
            if layer_name in l['sink_layers']]


def instantiate_layers_from_architecture(architecture):
    validate_architecture(architecture)
    layers = OrderedDict()
    for layer_name in get_canonical_layer_order(architecture):
        layer = architecture[layer_name]
        LayerClass = get_layer_class_from_typename(layer['@type'])

        size = layer.get('size')

        sources = get_source_layers(layer_name, architecture)

        in_size = sum(layers[l_name].out_size for l_name in sources)

        layers[layer_name] = LayerClass(size, in_size, layer['sink_layers'],
                                        sources, get_kwargs(layer))
    return layers
