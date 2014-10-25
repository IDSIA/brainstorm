#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from brainstorm.construction import ConstructionLayer
from brainstorm.architecture import generate_architecture, get_layer_description


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