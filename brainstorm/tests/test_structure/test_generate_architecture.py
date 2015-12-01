#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

from brainstorm.layers.input_layer import InputLayerImpl
from brainstorm.layers.noop_layer import NoOpLayerImpl
from brainstorm.structure.architecture import (
    generate_architecture, get_layer_description,
    instantiate_layers_from_architecture)
from brainstorm.structure.buffer_structure import BufferStructure
from brainstorm.structure.construction import ConstructionWrapper


def test_get_layer_description():
    l = ConstructionWrapper.create('FooLayerImpl', name='foo')
    l2 = ConstructionWrapper.create('FooLayerImpl', name='bar')
    l3 = ConstructionWrapper.create('FooLayerImpl', name='baz')
    _ = l >> l2
    _ = l >> l3
    descr = get_layer_description(l.layer)
    assert descr == {
        '@type': 'Foo',
        '@outgoing_connections': {
            'default': ['bar.default', 'baz.default']
        }
    }


def test_get_layer_description_named_inputs_outputs():
    l = ConstructionWrapper.create('FooLayerImpl', name='foo')
    l2 = ConstructionWrapper.create('FooLayerImpl', name='bar')
    l3 = ConstructionWrapper.create('FooLayerImpl', name='baz')
    _ = l - 'out1' >> l2
    _ = l >> 'A' - l3
    descr = get_layer_description(l.layer)
    assert descr == {
        '@type': 'Foo',
        '@outgoing_connections': {
            'default': ['baz.A'],
            'out1': ['bar.default']
        }
    }


def test_layer_with_kwargs():
    l = ConstructionWrapper.create('FooLayerImpl', name='foo', a=2, b=3)
    descr = get_layer_description(l.layer)
    assert descr == {
        '@type': 'Foo',
        '@outgoing_connections': {},
        'a': 2,
        'b': 3
    }


def test_generate_architecture():
    l1 = ConstructionWrapper.create('InputLayerImpl')
    l2 = ConstructionWrapper.create('FooLayerImpl', name='bar')
    l3 = ConstructionWrapper.create('FooLayerImpl', name='baz')
    l4 = ConstructionWrapper.create('FooLayerImpl', name='out')
    _ = l1 - 'foo' >> l2 >> 'A' - l4
    _ = l1 - 'bar' >> l3 >> 'B' - l4

    arch1 = generate_architecture(l1.layer)
    arch2 = generate_architecture(l2.layer)
    arch3 = generate_architecture(l3.layer)
    assert arch1 == arch2
    assert arch1 == arch3
    assert arch1 == {
        'Input': {
            '@type': 'Input',
            '@outgoing_connections': {
                'foo': ['bar.default'],
                'bar': ['baz.default'],
            }

        },
        'bar': {
            '@type': 'Foo',
            '@outgoing_connections': {
                'default': ['out.A'],
            }
        },
        'baz': {
            '@type': 'Foo',
            '@outgoing_connections': {
                'default': ['out.B'],
            }
        },
        'out': {
            '@type': 'Foo',
            '@outgoing_connections': {}
        }
    }


def test_instantiate_layers_from_architecture():
    arch = {
        'Input': {
            '@type': 'Input',
            'out_shapes': {'default': ('T', 'B', 10,)},
            '@outgoing_connections': ['A', 'B', 'C']
        },
        'A': {
            '@type': 'NoOp',
            '@outgoing_connections': ['B']
        },
        'B': {
            '@type': 'NoOp',
            '@outgoing_connections': ['D']
        },
        'C': {
            '@type': 'NoOp',
            '@outgoing_connections': ['D']
        },
        'D': {
            '@type': 'NoOp',
            '@outgoing_connections': []
        }
    }
    layers = instantiate_layers_from_architecture(arch)
    assert set(arch.keys()) == set(layers.keys())

    assert isinstance(layers['Input'], InputLayerImpl)
    assert type(layers['A']) == NoOpLayerImpl
    assert type(layers['B']) == NoOpLayerImpl
    assert type(layers['C']) == NoOpLayerImpl
    assert type(layers['D']) == NoOpLayerImpl

    assert layers['Input'].in_shapes == {}
    assert layers['Input'].out_shapes == {'default':
                                          BufferStructure('T', 'B', 10)}
    assert layers['A'].in_shapes == {'default': BufferStructure('T', 'B', 10)}
    assert layers['A'].out_shapes == {'default': BufferStructure('T', 'B', 10)}
    assert layers['B'].in_shapes == {'default': BufferStructure('T', 'B', 20)}
    assert layers['B'].out_shapes == {'default': BufferStructure('T', 'B', 20)}
    assert layers['C'].in_shapes == {'default': BufferStructure('T', 'B', 10)}
    assert layers['C'].out_shapes == {'default': BufferStructure('T', 'B', 10)}
    assert layers['D'].in_shapes == {'default': BufferStructure('T', 'B', 30)}
    assert layers['D'].out_shapes == {'default': BufferStructure('T', 'B', 30)}
