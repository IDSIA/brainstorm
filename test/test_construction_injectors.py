#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import pytest
from brainstorm.construction import (ConstructionInjector, ConstructionLayer,
                                     InvalidArchitectureError)


def test_constructor():
    i = ConstructionInjector('bartype', target_from='from', name='somename',
                             kew='l')
    assert i.injector_type == 'bartype'
    assert i.target_from == 'from'
    assert i.name == 'somename'
    assert i.injector_kwargs == {'kew': 'l'}
    assert repr(i) == "<ConstructionInjector: somename>"


def test_raises_on_invalid_injector_type():
    with pytest.raises(InvalidArchitectureError):
        i = ConstructionInjector('not valid!')


def test_raises_on_invalid_layer_name():
    with pytest.raises(InvalidArchitectureError):
        i = ConstructionInjector('layertype', name='also invalid.')


def test_connect_injector_to_layer():
    l1 = ConstructionLayer('Foo')
    l2 = ConstructionLayer('Foo')
    i = ConstructionInjector('bar')

    _ = l1 >> l2 << i
    assert i.output_from == l2
    assert i in l2.injectors


def test_connect_layer_to_injector():
    l1 = ConstructionLayer('Foo')
    l2 = ConstructionLayer('Foo')
    i = ConstructionInjector('bar')

    _ = l1 >> l2 >> i
    assert i.target_from == l2


def test_cannot_connect_to_multiple_layers():
    l1 = ConstructionLayer('Foo')
    l2 = ConstructionLayer('Foo')
    i = ConstructionInjector('bar')
    _ = l1 << i
    with pytest.raises(InvalidArchitectureError):
        _ = l2 << i


def test_cannot_get_receive_multiple_targets1():
    l1 = ConstructionLayer('Foo')
    i = ConstructionInjector('bar', target_from='somewhere')
    with pytest.raises(InvalidArchitectureError):
        _ = l1 >> i


def test_cannot_get_receive_multiple_targets2():
    l1 = ConstructionLayer('Foo')
    l2 = ConstructionLayer('Foo')
    i = ConstructionInjector('bar')
    _ = l1 >> i
    with pytest.raises(InvalidArchitectureError):
        _ = l2 >> i


def test_shift_operation_only_works_with_layers():
    i = ConstructionInjector('bar')
    with pytest.raises(TypeError):
        2 << i
    with pytest.raises(TypeError):
        i << 2
    with pytest.raises(TypeError):
        2 >> i
    with pytest.raises(TypeError):
        i >> 2
