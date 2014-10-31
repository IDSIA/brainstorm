#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import pytest
from brainstorm.buffers import create_param_layout, create_in_out_layout
from brainstorm.architecture import instantiate_layers_from_architecture
from brainstorm.utils import InvalidArchitectureError


@pytest.fixture
def layers():
    arch = {
        'InputLayer': {
            '@type': 'InputLayer',
            'size': 2,
            'sink_layers': {'A', 'B'}
        },
        'A': {
            '@type': 'FeedForwardLayer',
            'size': 3,
            'sink_layers': {'C'}
        },
        'B': {
            '@type': 'FeedForwardLayer',
            'size': 5,
            'sink_layers': {'C', 'D'}
        },
        'C': {
            '@type': 'FeedForwardLayer',
            'size': 7,
            'sink_layers': {'D'}
        },
        'D': {
            '@type': 'FeedForwardLayer',
            'size': 11,
            'sink_layers': set()
        }
    }
    return instantiate_layers_from_architecture(arch)


@pytest.fixture
def impossible_layers():
    arch = {
        'InputLayer': {
            '@type': 'InputLayer',
            'size': 2,
            'sink_layers': {'A', 'B'}
        },
        'A': {
            '@type': 'FeedForwardLayer',
            'sink_layers': {'C', 'D'}
        },
        'B': {
            '@type': 'FeedForwardLayer',
            'sink_layers': {'C', 'E'}
        },
        'C': {
            '@type': 'FeedForwardLayer',
            'sink_layers': {'D', 'E'}
        },
        'D': {
            '@type': 'FeedForwardLayer',
            'sink_layers': {'out'}
        },
        'E': {
            '@type': 'FeedForwardLayer',
            'sink_layers': {'out'}
        },
        'out': {
            '@type': 'FeedForwardLayer',
            'sink_layers': set()
        }
    }
    return instantiate_layers_from_architecture(arch)


def test_create_param_layout(layers):
    param_layout = create_param_layout(layers)
    assert param_layout.size == 230
    assert list(param_layout.layout) == ['InputLayer', 'A', 'B', 'C', 'D']
    assert param_layout.layout['InputLayer'] == slice(0, 0)
    assert param_layout.layout['A'] == slice(0, 9)
    assert param_layout.layout['B'] == slice(9, 24)
    assert param_layout.layout['C'] == slice(24, 87)
    assert param_layout.layout['D'] == slice(87, 230)


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
