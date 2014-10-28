#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import pytest

from brainstorm.python_layers import (
    get_layer_class_from_typename, InputLayer, NoOpLayer, LayerBase)


def test_get_layer_class_from_typename():
    assert get_layer_class_from_typename('InputLayer') == InputLayer
    assert get_layer_class_from_typename('NoOpLayer') == NoOpLayer


def test_get_layer_class_from_typename_raises_typeerror():
    with pytest.raises(TypeError):
        get_layer_class_from_typename('NonexistentLayer')


def test_layer_constructor():
    l = LayerBase(5, 8, {})
    assert l.out_size == 5
    assert l.in_size == 8
    assert l.kwargs == {}


def test_NoOp_raises_on_size_mismatch():
    with pytest.raises(AssertionError):
        l = NoOpLayer(5, 8, {})


def test_InputLayer_raises_on_in_size():
    with pytest.raises(AssertionError):
        l = InputLayer(5, 1, {})


@pytest.mark.parametrize("LayerClass", [
    LayerBase, InputLayer, NoOpLayer
])
def test_raises_on_unexpected_kwargs(LayerClass):
    with pytest.raises(AssertionError) as excinfo:
        l = LayerClass(0, 0, {'some_foo': 16})
    assert 'some_foo' in excinfo.value.msg
