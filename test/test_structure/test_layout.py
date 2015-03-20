#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

import pytest

from brainstorm.structure.layout import (
    create_param_layout, create_in_out_layout, ParameterLayoutEntry,
    create_layout_stub)
from brainstorm.structure.architecture import (
    instantiate_layers_from_architecture)
from brainstorm.utils import InvalidArchitectureError


@pytest.fixture
def impossible_layers():
    arch = {
        'InputLayer': {
            '@type': 'DataLayer',
            'shape': 2,
            '@outgoing_connections': {'A', 'B'}
        },
        'A': {
            '@type': 'FeedForwardLayer',
            'shape': 2,
            '@outgoing_connections': {'C', 'D'}
        },
        'B': {
            '@type': 'FeedForwardLayer',
            '@outgoing_connections': {'C', 'E'}
        },
        'C': {
            '@type': 'FeedForwardLayer',
            '@outgoing_connections': {'D', 'E'}
        },
        'D': {
            '@type': 'FeedForwardLayer',
            '@outgoing_connections': {'out'}
        },
        'E': {
            '@type': 'FeedForwardLayer',
            '@outgoing_connections': {'out'}
        },
        'out': {
            '@type': 'FeedForwardLayer',
            '@outgoing_connections': set()
        }
    }
    return instantiate_layers_from_architecture(arch)


def test_create_layout_stub(layers):
    layout = create_layout_stub(layers)
    import pprint
    pprint.pprint(layout)
    assert layout == {
        'InputLayer': {'index': 0, 'layout': {
            'inputs': {'index': 0, 'layout': {}},
            'outputs': {'index': 1, 'layout': {
                'default': {'index': 0, 'slice': (2, -1, -1), 'shape': (2,)},
            }},
            'parameters': {'index': 2, 'layout': {}},
            'internals': {'index': 3, 'layout': {}},
        }},
        'A': {'index': 1, 'layout': {
            'inputs': {'index': 0, 'layout': {
                'default': {'index': 0, 'slice': (2, -1, -1), 'shape': (2,)}
            }},
            'outputs': {'index': 1, 'layout': {
                'default': {'index': 0, 'slice': (2, -1, -1), 'shape': (3,)}
            }},
            'parameters': {'index': 2, 'layout': {
                'W': {'index': 0, 'slice': (0, -1, -1), 'shape': (2, 3)},
                'b': {'index': 1, 'slice': (0, -1, -1), 'shape': (3,)}
            }},
            'internals': {'index': 3, 'layout': {
                'Ha': {'index': 0, 'slice': (2, -1, -1), 'shape': (3,)}
            }},
        }},
        'B': {'index': 2, 'layout': {
            'inputs': {'index': 0, 'layout': {
                'default': {'index': 0, 'slice': (2, -1, -1), 'shape': (2,)}
            }},
            'outputs': {'index': 1, 'layout': {
                'default': {'index': 0, 'slice': (2, -1, -1), 'shape': (5,)}
            }},
            'parameters': {'index': 2, 'layout': {
                'W': {'index': 0, 'slice': (0, -1, -1), 'shape': (2, 5)},
                'b': {'index': 1, 'slice': (0, -1, -1), 'shape': (5,)}
            }},
            'internals': {'index': 3, 'layout': {
                'Ha': {'index': 0, 'slice': (2, -1, -1), 'shape': (5,)}
            }},
        }},
        'C': {'index': 3, 'layout': {
            'inputs': {'index': 0, 'layout': {
                'default': {'index': 0, 'slice': (2, -1, -1), 'shape': (8,)}
            }},
            'outputs': {'index': 1, 'layout': {
                'default': {'index': 0, 'slice': (2, -1, -1), 'shape': (7,)}
            }},
            'parameters': {'index': 2, 'layout': {
                'W': {'index': 0, 'slice': (0, -1, -1), 'shape': (8, 7)},
                'b': {'index': 1, 'slice': (0, -1, -1), 'shape': (7,)}
            }},
            'internals': {'index': 3, 'layout': {
                'Ha': {'index': 0, 'slice': (2, -1, -1), 'shape': (7,)}
            }},
        }},
        'D': {'index': 4, 'layout': {
            'inputs': {'index': 0, 'layout': {
                'default': {'index': 0, 'slice': (2, -1, -1), 'shape': (12,)}
            }},
            'outputs': {'index': 1, 'layout': {
                'default': {'index': 0, 'slice': (2, -1, -1), 'shape': (11,)}
            }},
            'parameters': {'index': 2, 'layout': {
                'W': {'index': 0, 'slice': (0, -1, -1), 'shape': (12, 11)},
                'b': {'index': 1, 'slice': (0, -1, -1), 'shape': (11,)}
            }},
            'internals': {'index': 3, 'layout': {
                'Ha': {'index': 0, 'slice': (2, -1, -1), 'shape': (11,)}
            }},
        }}
    }


def test_create_param_layout(layers):
    param_layout = create_param_layout(layers)
    assert param_layout.size == 230
    assert list(param_layout.layout) == ['InputLayer', 'A', 'B', 'C', 'D']
    structs = {name: layer.get_parameter_structure()
               for name, layer in layers.items()}
    assert param_layout.layout['InputLayer'] == ParameterLayoutEntry(0, 0, [])
    assert param_layout.layout['A'] == ParameterLayoutEntry(0, 9, structs['A'])
    assert param_layout.layout['B'] == ParameterLayoutEntry(9, 24,
                                                            structs['B'])
    assert param_layout.layout['C'] == ParameterLayoutEntry(24, 87,
                                                            structs['C'])
    assert param_layout.layout['D'] == ParameterLayoutEntry(87, 230,
                                                            structs['D'])


def test_create_in_out_layout(layers):
    hubs = create_in_out_layout(layers)
    assert len(hubs) == 3
    hub1, hub2, hub3 = hubs

    assert hub1.size == 2
    assert list(hub1.source_layout.keys()) == ['InputLayer']
    assert list(hub1.sink_layout.keys()) == ['A', 'B']
    assert hub1.source_layout['InputLayer'] == slice(0, 2)
    assert hub1.sink_layout['A'] == slice(0, 2)
    assert hub1.sink_layout['B'] == slice(0, 2)

    assert hub2.size == 15
    assert list(hub2.source_layout.keys()) == ['A', 'B', 'C']
    assert list(hub2.sink_layout.keys()) == ['C', 'D']
    assert hub2.source_layout['A'] == slice(0, 3)
    assert hub2.source_layout['B'] == slice(3, 8)
    assert hub2.source_layout['C'] == slice(8, 15)
    assert hub2.sink_layout['C'] == slice(0, 8)
    assert hub2.sink_layout['D'] == slice(3, 15)

    assert hub3.size == 11
    assert list(hub3.source_layout.keys()) == ['D']
    assert list(hub3.sink_layout.keys()) == []
    assert hub3.source_layout['D'] == slice(0, 11)


def test_raises_for_impossible_layers(impossible_layers):
    with pytest.raises(InvalidArchitectureError):
        hubs = create_in_out_layout(impossible_layers)
