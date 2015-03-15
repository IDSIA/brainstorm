#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from brainstorm.structure.construction import ConstructionLayer
from brainstorm.structure.architecture import (
    generate_architecture, get_layer_description, get_canonical_layer_order,
    instantiate_layers_from_architecture)
from brainstorm.layers.python_layers import DataLayer, NoOpLayer


def test_get_layer_description():
    l = ConstructionLayer('layertype', 10, name='foo')
    l2 = ConstructionLayer('layertype', 10, name='bar')
    l3 = ConstructionLayer('layertype', 10, name='baz')
    l >> l2
    l >> l3
    descr = get_layer_description(l)
    assert descr == {
        '@type': 'layertype',
        'shape': 10,
        '@outgoing_connections': {'bar', 'baz'}
    }


def test_layer_with_kwargs():
    l = ConstructionLayer('layertype', 10, name='foo', a=2, b=3)
    descr = get_layer_description(l)
    assert descr == {
        '@type': 'layertype',
        'shape': 10,
        '@outgoing_connections': set(),
        'a': 2,
        'b': 3
    }


def test_generate_architecture():
    l1 = ConstructionLayer('DataLayer', 10)
    l2 = ConstructionLayer('layertype', 20, name='bar')
    l3 = ConstructionLayer('layertype', 30, name='baz')
    l4 = ConstructionLayer('layertype', 40, name='out')
    l1 >> l2 >> l4
    l1 >> l3 >> l4

    arch1 = generate_architecture(l1)
    arch2 = generate_architecture(l2)
    arch3 = generate_architecture(l3)
    assert arch1 == arch2
    assert arch1 == arch3
    assert arch1 == {
        'DataLayer': {
            '@type': 'DataLayer',
            'shape': 10,
            '@outgoing_connections': {'bar', 'baz'}
        },
        'bar': {
            '@type': 'layertype',
            'shape': 20,
            '@outgoing_connections': {'out'}
        },
        'baz': {
            '@type': 'layertype',
            'shape': 30,
            '@outgoing_connections': {'out'}
        },
        'out': {
            '@type': 'layertype',
            'shape': 40,
            '@outgoing_connections': set()
        }
    }


def test_instantiate_layers_from_architecture():
    arch = {
        'InputLayer': {
            '@type': 'DataLayer',
            'shape': 10,
            '@outgoing_connections': {'A', 'B', 'C'}
        },
        'A': {
            '@type': 'NoOpLayer',
            'shape': 10,
            '@outgoing_connections': {'B'}
        },
        'B': {
            '@type': 'NoOpLayer',
            'shape': 20,
            '@outgoing_connections': {'D'}
        },
        'C': {
            '@type': 'NoOpLayer',
            'shape': 10,
            '@outgoing_connections': {'D'}
        },
        'D': {
            '@type': 'NoOpLayer',
            'shape': 30,
            '@outgoing_connections': set()
        }
    }
    layers = instantiate_layers_from_architecture(arch)
    assert set(arch.keys()) == set(layers.keys())

    assert type(layers['InputLayer']) == DataLayer
    assert type(layers['A']) == NoOpLayer
    assert type(layers['B']) == NoOpLayer
    assert type(layers['C']) == NoOpLayer
    assert type(layers['D']) == NoOpLayer

    assert layers['InputLayer'].in_shape == (0,)
    assert layers['InputLayer'].shape == (10,)
    assert layers['A'].in_shape == (10,)
    assert layers['A'].shape == (10,)
    assert layers['B'].in_shape == (20,)
    assert layers['B'].shape == (20,)
    assert layers['C'].in_shape == (10,)
    assert layers['C'].shape == (10,)
    assert layers['D'].in_shape == (30,)
    assert layers['D'].shape == (30,)
