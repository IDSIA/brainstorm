#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import pytest
from brainstorm.construction import ConstructionLayer, InvalidArchitectureError


@pytest.fixture
def layers():
    return [ConstructionLayer('dummy_type', i) for i in range(1, 6)]


def test_constructor():
    cl = ConstructionLayer('Foo', 7)
    assert cl.size == 7
    assert cl.layer_type == 'Foo'
    assert repr(cl) == "<ConstructionLayer: Foo>"


def test_raises_on_invalid_layer_type():
    with pytest.raises(InvalidArchitectureError):
        i = ConstructionLayer('not valid!')


def test_raises_on_invalid_layer_name():
    with pytest.raises(InvalidArchitectureError):
        i = ConstructionLayer('layertype', name='also invalid.')

def test_connecting_two_layers_sets_sinks_and_sources(layers):
    l1, l2, l3, l4, l5 = layers
    _ = l1 >> l2 >> l3
    assert l1 in l2.source_layers
    assert l2 in l1.sink_layers
    assert l3 in l2.sink_layers


def test_connect_multiple_targets(layers):
    l1, l2, l3, l4, l5 = layers
    _ = l1 >> l2
    _ = l1 >> l3
    _ = l1 >> l4
    assert l2 in l1.sink_layers
    assert l3 in l1.sink_layers
    assert l4 in l1.sink_layers


def test_connect_multiple_sources(layers):
    l1, l2, l3, l4, l5 = layers
    _ = l2 >> l1
    _ = l3 >> l1
    _ = l4 >> l1
    assert l2 in l1.source_layers
    assert l3 in l1.source_layers
    assert l4 in l1.source_layers


def test_collect_connected_layers(layers):
    l1, l2, l3, l4, l5 = layers
    _ = l1 >> l2 >> l3 >> l4 >> l5
    assert l1.collect_connected_layers() == set(layers)
    assert l5.collect_connected_layers() == set(layers)


def test_collect_connected_layers2(layers):
    l1, l2, l3, l4, l5 = layers
    _ = l1 >> l2 >> l3 >> l4
    _ = l1 >> l5 >> l4
    assert l1.collect_connected_layers() == set(layers)
    assert l4.collect_connected_layers() == set(layers)
    assert l5.collect_connected_layers() == set(layers)


def test_building_circle_raises_error(layers):
    l1, l2, l3, l4, l5 = layers
    with pytest.raises(InvalidArchitectureError):
        _ = l1 >> l1
    with pytest.raises(InvalidArchitectureError):
        _ = l1 >> l2 >> l1
    with pytest.raises(InvalidArchitectureError):
        _ = l1 >> l2 >> l3 >> l4 >> l5 >> l1


def test_name():
    l = ConstructionLayer('Foo', 7, name='bar')
    assert l.name == 'bar'


def test_default_name():
    l = ConstructionLayer('Foo', 7)
    assert l.name == 'Foo'


def test_name_unconnected():
    l1 = ConstructionLayer('Foo', 7, name='bar')
    l2 = ConstructionLayer('Foo', 7, name='bar')
    assert l1.name == 'bar'
    assert l2.name == 'bar'


def test_name_connected():
    l1 = ConstructionLayer('Foo', 7, name='bar')
    l2 = ConstructionLayer('Foo', 7, name='bar')
    _ = l1 >> l2
    assert l1.name == 'bar_1'
    assert l2.name == 'bar_2'


def test_name_connected_complex(layers):
    l1, l2, l3, l4, l5 = layers
    _ = l3 >> l4
    _ = l2 >> l1
    _ = l5 >> l2 >> l3
    assert l1.name == 'dummy_type_1'
    assert l2.name == 'dummy_type_2'
    assert l3.name == 'dummy_type_3'
    assert l4.name == 'dummy_type_4'
    assert l5.name == 'dummy_type_5'


def test_inherit_size():
    l7 = ConstructionLayer('foo', 7)
    l_ = ConstructionLayer('foo')

    l7 >> l_
    assert l_.size == 7


def test_inherit_size_summed():
    l7 = ConstructionLayer('foo', 7)
    l4 = ConstructionLayer('foo', 4)
    l_ = ConstructionLayer('foo')

    l7 >> l_
    l4 >> l_
    assert l_.size == 11


def test_inherit_size_tricky():
    l7 = ConstructionLayer('foo', 7)
    l4 = ConstructionLayer('foo', 4)
    l_1 = ConstructionLayer('foo')
    l_2 = ConstructionLayer('foo')

    l_1 >> l_2
    l7 >> l_1
    l4 >> l_2
    assert l_2.size == 11


def test_inherit_size_none():
    l_1 = ConstructionLayer('foo')
    l_2 = ConstructionLayer('foo')

    l_1 >> l_2
    with pytest.raises(InvalidArchitectureError):
        _ = l_2.size