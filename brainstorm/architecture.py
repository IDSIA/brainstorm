#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from six import string_types
from .construction import PYTHON_IDENTIFIER


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
    validate_architecture(arch)
    return arch


def validate_architecture(architecture):
    # schema
    for name, layer in architecture.items():
        assert isinstance(name, string_types)
        assert 'size' in layer and isinstance(layer['size'], int)
        assert '@type' in layer and isinstance(layer['@type'], string_types)
        assert 'sink_layers' in layer and isinstance(layer['sink_layers'], set)

    # no layer is called 'default'
    assert 'default' not in architecture, "'default' is an invalid layer name"

    # layer naming
    for name in architecture:
        assert PYTHON_IDENTIFIER.match(name), \
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
