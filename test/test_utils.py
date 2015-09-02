#!/usr/bin/env python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
from brainstorm.handlers import NumpyHandler
from brainstorm.utils import get_inheritors, flatten, convert_to_nested_indices, \
    flatten_time, flatten_time_and_features
import numpy as np

def test_get_inheritors():
    class A(object):
        pass

    class B(A):
        pass

    class C(B):
        pass

    class D(A):
        pass

    class E(object):
        pass

    assert get_inheritors(A) == {B, C, D}


def test_flatten():
    assert list(flatten([0, (1, 2, 3), 4, [5, (6, 7), 8]])) == list(range(9))


def test_convert_to_nested_indices():
    assert list(convert_to_nested_indices(
        ['a', ('b', 'c', 'a'), 'b', ['c', ('c', 'c'), 'b']])) == \
        [0, [1, 2, 3], 4, [5, [6, 7], 8]]

def test_flatten_time():
    # Testing for NumpyHandler only
    _h = NumpyHandler(np.float64)
    shape = (2, 3, 2, 4)
    x = np.random.randn(*shape)
    y = flatten_time(x).copy()
    yp = x.reshape((6, 2, 4))
    assert np.allclose(y, yp)

def test_flatten_time_and_features():
    # Testing for NumpyHandler only
    _h = NumpyHandler(np.float64)
    shape = (2, 3, 2, 4)
    x = np.random.randn(*shape)
    y = flatten_time_and_features(x).copy()
    yp = x.reshape((6, 8))
    assert np.allclose(y, yp)
