#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

import pytest

from brainstorm.structure.construction import (
    ConstructionLayer, InvalidArchitectureError)


@pytest.fixture
def layers():
    return [ConstructionLayer('dummy_type', i) for i in range(1, 6)]


def test_constructor():
    cl = ConstructionLayer('Foo', 7)
    assert cl.layer_kwargs == {'shape': 7}
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
    assert (l1, 'default', 'default') in l2.incoming
    assert ('default', 'default', l2) in l1.outgoing
    assert ('default', 'default', l3) in l2.outgoing


def test_connect_multiple_targets(layers):
    l1, l2, l3, l4, l5 = layers
    _ = l1 >> l2
    _ = l1 >> l3
    _ = l1 >> l4
    assert ('default', 'default', l2) in l1.outgoing
    assert ('default', 'default', l3) in l1.outgoing
    assert ('default', 'default', l4) in l1.outgoing


def test_connect_multiple_sources(layers):
    l1, l2, l3, l4, l5 = layers
    _ = l2 >> l1
    _ = l3 >> l1
    _ = l4 >> l1
    assert (l2, 'default', 'default') in l1.incoming
    assert (l3, 'default', 'default') in l1.incoming
    assert (l4, 'default', 'default') in l1.incoming


def test_connect_named_output(layers):
    l1, l2, l3, l4, l5 = layers
    _ = l1 >> l2 - 'out1' >> l3

    assert ('default', 'default', l2) in l1.outgoing
    assert (l1, 'default', 'default') in l2.incoming
    assert (l2, 'out1', 'default') in l3.incoming
    assert ('out1', 'default', l3) in l2.outgoing


def test_connect_named_input(layers):
    l1, l2, l3, l4, l5 = layers
    _ = l1 >> "in1" - l2 >> l3

    assert (l1, 'default', 'in1') in l2.incoming
    assert ('default', 'in1', l2) in l1.outgoing
    assert ('default', 'default', l3) in l2.outgoing
    assert (l2, 'default', 'default') in l3.incoming


def test_connect_named_output_to_name_input(layers):
    l1, l2, l3, l4, l5 = layers
    _ = l1 >> l2 - "out1" >> "in1" - l3 >> l4

    assert ('default', 'default', l2) in l1.outgoing
    assert (l1, 'default', 'default') in l2.incoming

    assert ('out1', 'in1', l3) in l2.outgoing
    assert (l2, 'out1', 'in1') in l3.incoming

    assert ('default', 'default', l4) in l3.outgoing
    assert (l3, 'default', 'default') in l4.incoming


def test_connect_named_output_to_name_input_in_chain(layers):
    l1, l2, l3, l4, l5 = layers
    _ = l1 >> "in1" - l2 - "out1" >> l3

    assert ('default', 'in1', l2) in l1.outgoing
    assert (l1, 'default', 'in1') in l2.incoming
    assert ('out1', 'default', l3) in l2.outgoing
    assert (l2, 'out1', 'default') in l3.incoming


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
