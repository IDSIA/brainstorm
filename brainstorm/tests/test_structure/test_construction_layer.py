#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

import pytest

from brainstorm.structure.construction import (ConstructionWrapper,
                                               NetworkValidationError)


@pytest.fixture
def layers():
    return [ConstructionWrapper.create('DummyLayerImpl') for _ in range(5)]


def test_constructor():
    cl = ConstructionWrapper.create('FooLayerImpl')
    assert cl.layer.layer_type == 'Foo'
    assert repr(cl) == "<Layer: 'default' - Foo - 'default'>"


def test_raises_on_invalid_layer_type():
    with pytest.raises(NetworkValidationError):
        i = ConstructionWrapper.create('not valid!')


def test_raises_on_invalid_layer_name():
    with pytest.raises(NetworkValidationError):
        i = ConstructionWrapper.create('layertype', name='also invalid.')


def test_connecting_two_layers_sets_sinks_and_sources(layers):
    l1, l2, l3, l4, l5 = layers
    _ = l1 >> l2 >> l3
    assert (l1.layer, 'default', 'default') in l2.layer.incoming
    assert ('default', 'default', l2.layer) in l1.layer.outgoing
    assert ('default', 'default', l3.layer) in l2.layer.outgoing


def test_connect_multiple_targets(layers):
    l1, l2, l3, l4, l5 = layers
    _ = l1 >> l2
    _ = l1 >> l3
    _ = l1 >> l4
    assert ('default', 'default', l2.layer) in l1.layer.outgoing
    assert ('default', 'default', l3.layer) in l1.layer.outgoing
    assert ('default', 'default', l4.layer) in l1.layer.outgoing


def test_connect_multiple_sources(layers):
    l1, l2, l3, l4, l5 = layers
    _ = l2 >> l1
    _ = l3 >> l1
    _ = l4 >> l1
    assert (l2.layer, 'default', 'default') in l1.layer.incoming
    assert (l3.layer, 'default', 'default') in l1.layer.incoming
    assert (l4.layer, 'default', 'default') in l1.layer.incoming


def test_connect_named_output(layers):
    l1, l2, l3, l4, l5 = layers
    _ = l1 >> l2 - 'out1' >> l3

    assert ('default', 'default', l2.layer) in l1.layer.outgoing
    assert (l1.layer, 'default', 'default') in l2.layer.incoming
    assert (l2.layer, 'out1', 'default') in l3.layer.incoming
    assert ('out1', 'default', l3.layer) in l2.layer.outgoing


def test_connect_named_input(layers):
    l1, l2, l3, l4, l5 = layers
    _ = l1 >> "in1" - l2 >> l3

    assert (l1.layer, 'default', 'in1') in l2.layer.incoming
    assert ('default', 'in1', l2.layer) in l1.layer.outgoing
    assert ('default', 'default', l3.layer) in l2.layer.outgoing
    assert (l2.layer, 'default', 'default') in l3.layer.incoming


def test_connect_named_output_to_name_input(layers):
    l1, l2, l3, l4, l5 = layers
    _ = l1 >> l2 - "out1" >> "in1" - l3 >> l4

    assert ('default', 'default', l2.layer) in l1.layer.outgoing
    assert (l1.layer, 'default', 'default') in l2.layer.incoming

    assert ('out1', 'in1', l3.layer) in l2.layer.outgoing
    assert (l2.layer, 'out1', 'in1') in l3.layer.incoming

    assert ('default', 'default', l4.layer) in l3.layer.outgoing
    assert (l3.layer, 'default', 'default') in l4.layer.incoming


def test_connect_named_output_to_name_input_in_chain(layers):
    l1, l2, l3, l4, l5 = layers
    _ = l1 >> "in1" - l2 - "out1" >> l3

    assert ('default', 'in1', l2.layer) in l1.layer.outgoing
    assert (l1.layer, 'default', 'in1') in l2.layer.incoming
    assert ('out1', 'default', l3.layer) in l2.layer.outgoing
    assert (l2.layer, 'out1', 'default') in l3.layer.incoming


def test_collect_connected_layers(layers):
    l1, l2, l3, l4, l5 = layers
    _ = l1 >> l2 >> l3 >> l4 >> l5
    layer_set = {l.layer for l in layers}
    assert l1.layer.collect_connected_layers() == layer_set
    assert l5.layer.collect_connected_layers() == layer_set


def test_collect_connected_layers2(layers):
    l1, l2, l3, l4, l5 = layers
    _ = l1 >> l2 >> l3 >> l4
    _ = l1 >> l5 >> l4
    layer_set = {l.layer for l in layers}
    assert l1.layer.collect_connected_layers() == layer_set
    assert l4.layer.collect_connected_layers() == layer_set
    assert l5.layer.collect_connected_layers() == layer_set


def test_name():
    l = ConstructionWrapper.create('FooLayerImpl', name='bar')
    assert l.layer.name == 'bar'


def test_default_name():
    l = ConstructionWrapper.create('FooLayerImpl')
    assert l.layer.name == 'Foo'


def test_name_unconnected():
    l1 = ConstructionWrapper.create('FooLayerImpl', name='bar')
    l2 = ConstructionWrapper.create('FooLayerImpl', name='bar')
    assert l1.layer.name == 'bar'
    assert l2.layer.name == 'bar'


def test_name_connected():
    l1 = ConstructionWrapper.create('FooLayerImpl', name='bar')
    l2 = ConstructionWrapper.create('FooLayerImpl', name='bar')
    _ = l1 >> l2
    assert l1.layer.name == 'bar_1'
    assert l2.layer.name == 'bar_2'


def test_name_connected_complex(layers):
    l1, l2, l3, l4, l5 = layers
    _ = l3 >> l4
    _ = l2 >> l1
    _ = l5 >> l2 >> l3
    assert l1.layer.name == 'Dummy_1'
    assert l2.layer.name == 'Dummy_2'
    assert l3.layer.name == 'Dummy_3'
    assert l4.layer.name == 'Dummy_4'
    assert l5.layer.name == 'Dummy_5'
