#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from brainstorm.structure.construction import ConstructionLayer
from brainstorm.structure.architecture import (
    generate_architecture, get_layer_description, get_canonical_layer_order,
    instantiate_layers_from_architecture)
from brainstorm.layers.python_layers import InputLayer, NoOpLayer


def test_get_layer_description():
    l = ConstructionLayer('layertype', 10, name='foo')
    l2 = ConstructionLayer('layertype', 10, name='bar')
    l3 = ConstructionLayer('layertype', 10, name='baz')
    l >> l2
    l >> l3
    descr = get_layer_description(l)
    assert descr == {
        '@type': 'layertype',
        'size': 10,
        'sink_layers': {'bar', 'baz'}
    }


def test_layer_with_kwargs():
    l = ConstructionLayer('layertype', 10, name='foo', a=2, b=3)
    descr = get_layer_description(l)
    assert descr == {
        '@type': 'layertype',
        'size': 10,
        'sink_layers': set(),
        'a': 2,
        'b': 3
    }


def test_generate_architecture():
    l1 = ConstructionLayer('InputLayer', 10)
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
        'InputLayer': {
            '@type': 'InputLayer',
            'size': 10,
            'sink_layers': {'bar', 'baz'}
        },
        'bar': {
            '@type': 'layertype',
            'size': 20,
            'sink_layers': {'out'}
        },
        'baz': {
            '@type': 'layertype',
            'size': 30,
            'sink_layers': {'out'}
        },
        'out': {
            '@type': 'layertype',
            'size': 40,
            'sink_layers': set()
        }
    }


def test_get_canonical_architecture_order():
    arch = {
        'A': {
            '@type': 'InputLayer',
            'size': 10,
            'sink_layers': {'B1', 'C'}
        },
        'B1': {
            '@type': 'layertype',
            'size': 20,
            'sink_layers': {'B2'}
        },
        'B2': {
            '@type': 'layertype',
            'size': 20,
            'sink_layers': {'D'}
        },
        'C': {
            '@type': 'layertype',
            'size': 30,
            'sink_layers': {'D'}
        },
        'D': {
            '@type': 'layertype',
            'size': 40,
            'sink_layers': set()
        }
    }
    assert get_canonical_layer_order(arch) == ['A', 'B1', 'B2', 'C', 'D']


def test_instantiate_layers_from_architecture():
    arch = {
        'InputLayer': {
            '@type': 'InputLayer',
            'size': 10,
            'sink_layers': {'A', 'B', 'C'}
        },
        'A': {
            '@type': 'NoOpLayer',
            'size': 10,
            'sink_layers': {'B'}
        },
        'B': {
            '@type': 'NoOpLayer',
            'size': 20,
            'sink_layers': {'D'}
        },
        'C': {
            '@type': 'NoOpLayer',
            'size': 10,
            'sink_layers': {'D'}
        },
        'D': {
            '@type': 'NoOpLayer',
            'size': 30,
            'sink_layers': set()
        }
    }
    layers = instantiate_layers_from_architecture(arch)
    assert set(arch.keys()) == set(layers.keys())

    assert type(layers['InputLayer']) == InputLayer
    assert type(layers['A']) == NoOpLayer
    assert type(layers['B']) == NoOpLayer
    assert type(layers['C']) == NoOpLayer
    assert type(layers['D']) == NoOpLayer

    assert layers['InputLayer'].in_size == (0,)
    assert layers['InputLayer'].out_size == (10,)
    assert layers['A'].in_size == (10,)
    assert layers['A'].out_size == (10,)
    assert layers['B'].in_size == (20,)
    assert layers['B'].out_size == (20,)
    assert layers['C'].in_size == (10,)
    assert layers['C'].out_size == (10,)
    assert layers['D'].in_size == (30,)
    assert layers['D'].out_size == (30,)
