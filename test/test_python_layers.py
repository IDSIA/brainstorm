#!/usr/bin/env python
# coding=utf-8

from __future__ import division, print_function, unicode_literals

import pytest

from brainstorm.layers.base_layer import get_layer_class_from_typename
from brainstorm.layers.input_layer import InputLayerImpl
from brainstorm.layers.noop_layer import NoOpLayerImpl
from brainstorm.layers.fully_connected_layer import FullyConnectedLayerImpl
from brainstorm.structure.architecture import Connection
from brainstorm.utils import LayerValidationError
from brainstorm.structure.shapes import ShapeTemplate


def test_get_layer_class_from_typename():
    assert get_layer_class_from_typename('InputLayerImpl') == InputLayerImpl
    assert get_layer_class_from_typename('NoOpLayerImpl') == NoOpLayerImpl


def test_get_layer_class_from_typename_raises_typeerror():
    with pytest.raises(TypeError):
        get_layer_class_from_typename('NonexistentLayer')


def test_layer_constructor():
    a = Connection('l', 'default', 'A', 'default')
    b = Connection('l', 'default', 'B', 'default')
    c = Connection('l', 'default', 'C', 'default')

    l = FullyConnectedLayerImpl('LayerName',
                                {'default': ShapeTemplate('T', 'B', 5)},
                                {c},
                                {a, b},
                                size=8)
    expected = {'default': ShapeTemplate('T', 'B', 8)}
    assert l.out_shapes == expected
    assert l.in_shapes == {'default': ShapeTemplate('T', 'B', 5)}
    assert l.incoming == {c}
    assert l.outgoing == {a, b}
    assert l.kwargs == {'size': 8}


def test_nooplayer_raises_on_size_mismatch():
    with pytest.raises(LayerValidationError):
        l = NoOpLayerImpl('LayerName', {'default': ('T', 'B', 5,)}, set(),
                          set(), size=8)


def test_inputlayer_raises_on_in_size():
    with pytest.raises(LayerValidationError):
        l = InputLayerImpl('LayerName', {'default': ('T', 'B', 5,)}, set(),
                           set(), out_shapes={'default': ('T', 'B', 5,)})


@pytest.mark.parametrize("LayerClass", [
    NoOpLayerImpl, FullyConnectedLayerImpl
])
def test_raises_on_unexpected_kwargs(LayerClass):
    with pytest.raises(LayerValidationError) as excinfo:
        l = LayerClass('LayerName', {'default': ShapeTemplate(5,)}, set(), set(),
                       some_foo=16)
    assert 'some_foo' in excinfo.value.args[0]
