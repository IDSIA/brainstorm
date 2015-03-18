#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from brainstorm.structure.construction import ConstructionWrapper
from brainstorm.structure.architecture import (
    generate_architecture, get_layer_description, get_canonical_layer_order,
    instantiate_layers_from_architecture)
from brainstorm.layers.python_layers import DataLayer, NoOpLayer


def test_get_layer_description():
    l = ConstructionWrapper.create('layertype', 10, name='foo')
    l2 = ConstructionWrapper.create('layertype', 10, name='bar')
    l3 = ConstructionWrapper.create('layertype', 10, name='baz')
    _ = l >> l2
    _ = l >> l3
    descr = get_layer_description(l.layer)
    assert descr == {
        '@type': 'layertype',
        '@outgoing_connections': {
            'default': {'bar.default', 'baz.default'}
        },
        'shape': 10
    }


def test_layer_with_kwargs():
    l = ConstructionWrapper.create('layertype', 10, name='foo', a=2, b=3)
    descr = get_layer_description(l.layer)
    assert descr == {
        '@type': 'layertype',
        '@outgoing_connections': {},
        'shape': 10,
        'a': 2,
        'b': 3
    }


def test_generate_architecture():
    l1 = ConstructionWrapper.create('DataLayer', 10)
    l2 = ConstructionWrapper.create('layertype', 20, name='bar')
    l3 = ConstructionWrapper.create('layertype', 30, name='baz')
    l4 = ConstructionWrapper.create('layertype', 40, name='out')
    _ = l1 >> l2 >> l4
    _ = l1 >> l3 >> l4

    arch1 = generate_architecture(l1.layer)
    arch2 = generate_architecture(l2.layer)
    arch3 = generate_architecture(l3.layer)
    assert arch1 == arch2
    assert arch1 == arch3
    assert arch1 == {
        'DataLayer': {
            '@type': 'DataLayer',
            'shape': 10,
            '@outgoing_connections': {
                'default': {'bar.default', 'baz.default'},
            }

        },
        'bar': {
            '@type': 'layertype',
            'shape': 20,
            '@outgoing_connections': {
                'default': {'out.default'},
            }
        },
        'baz': {
            '@type': 'layertype',
            'shape': 30,
            '@outgoing_connections': {
                'default': {'out.default'},
            }
        },
        'out': {
            '@type': 'layertype',
            'shape': 40,
            '@outgoing_connections': {}
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

    assert layers['InputLayer'].in_shapes == {}
    assert layers['InputLayer'].out_shapes == {'default': (10,)}
    assert layers['A'].in_shapes == {'default': (10,)}
    assert layers['A'].out_shapes == {'default': (10,)}
    assert layers['B'].in_shapes == {'default': (20,)}
    assert layers['B'].out_shapes == {'default': (20,)}
    assert layers['C'].in_shapes == {'default': (10,)}
    assert layers['C'].out_shapes == {'default': (10,)}
    assert layers['D'].in_shapes == {'default': (30,)}
    assert layers['D'].out_shapes == {'default': (30,)}
