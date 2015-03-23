#!/usr/bin/env python
# coding=utf-8

from __future__ import division, print_function, unicode_literals

import pytest

from brainstorm.layers.base_layer import get_layer_class_from_typename
from brainstorm.layers.python_layers import InputLayer, NoOpLayer, LayerBase
from brainstorm.structure.architecture import Connection


def test_get_layer_class_from_typename():
    assert get_layer_class_from_typename('InputLayer') == InputLayer
    assert get_layer_class_from_typename('NoOpLayer') == NoOpLayer


def test_get_layer_class_from_typename_raises_typeerror():
    with pytest.raises(TypeError):
        get_layer_class_from_typename('NonexistentLayer')


def test_layer_constructor():
    a = Connection('l', 'default', 'A', 'default')
    b = Connection('l', 'default', 'B', 'default')
    c = Connection('l', 'default', 'C', 'default')

    l = LayerBase({'default': (5,)}, {c}, {a, b}, shape=8)
    assert l.out_shapes == {'default': (8,)}
    assert l.in_shapes == {'default': (5,)}
    assert l.incoming == {c}
    assert l.outgoing == {a, b}
    assert l.kwargs == {'shape': 8}


def test_NoOp_raises_on_size_mismatch():
    with pytest.raises(ValueError):
        l = NoOpLayer({'default': (5,)}, set(), set(), shape=8)


def test_DataLayer_raises_on_in_size():
    with pytest.raises(ValueError):
        l = InputLayer({'default': (5,)}, set(), set())


@pytest.mark.parametrize("LayerClass", [
    LayerBase, InputLayer, NoOpLayer
])
def test_raises_on_unexpected_kwargs(LayerClass):
    with pytest.raises(ValueError) as excinfo:
        l = LayerClass({'default': (5,)}, set(), set(), some_foo=16)
    assert 'some_foo' in excinfo.value.args[0]
