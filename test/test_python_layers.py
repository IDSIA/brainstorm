#!/usr/bin/env python
# coding=utf-8

from __future__ import division, print_function, unicode_literals

import pytest

from brainstorm.layers.python_layers import (
    get_layer_class_from_typename, DataLayer, NoOpLayer, LayerBase)


def test_get_layer_class_from_typename():
    assert get_layer_class_from_typename('DataLayer') == DataLayer
    assert get_layer_class_from_typename('NoOpLayer') == NoOpLayer


def test_get_layer_class_from_typename_raises_typeerror():
    with pytest.raises(TypeError):
        get_layer_class_from_typename('NonexistentLayer')


def test_layer_constructor():
    l = LayerBase(5, 8, {'A', 'B'}, {'C'}, {})
    assert l.shape == 5
    assert l.in_shape == 8
    assert l.incoming == {'C'}
    assert l.outgoing == {'A', 'B'}
    assert l.kwargs == {}


def test_NoOp_raises_on_size_mismatch():
    with pytest.raises(AssertionError):
        l = NoOpLayer(5, 8, set(), set(), {})


def test_DataLayer_raises_on_in_size():
    with pytest.raises(AssertionError):
        l = DataLayer(5, 1, set(), set(), {})


@pytest.mark.parametrize("LayerClass", [
    LayerBase, DataLayer, NoOpLayer
])
def test_raises_on_unexpected_kwargs(LayerClass):
    with pytest.raises(AssertionError) as excinfo:
        l = LayerClass(0, 0, set(), set(), {'some_foo': 16})
    assert 'some_foo' in excinfo.value.msg
