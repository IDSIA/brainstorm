#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

import numpy as np
import pytest

from brainstorm.handlers.debug_handler import DebugArray


@pytest.fixture
def array():
    return DebugArray(np.zeros((11, 7, 5, 3, 2)))


def test_array_shape(array):
    assert isinstance(array.shape, tuple)
    assert array.shape == (11, 7, 5, 3, 2)


def test_simple_single_indexing(array):
    b = array[1]
    assert isinstance(b, type(array))
    assert b.shape == (7, 5, 3, 2)

    with pytest.raises(IndexError):
        _ = array[11]


def test_simple_double_indexing(array):
    b = array[1, 4]
    assert isinstance(b, type(array))
    assert b.shape == (5, 3, 2)

    with pytest.raises(IndexError):
        _ = array[1, 7]


def test_simple_triple_indexing(array):
    b = array[1, 4, 0]
    assert isinstance(b, type(array))
    assert b.shape == (3, 2)

    with pytest.raises(IndexError):
        _ = array[1, 2, 5]


def test_simple_quad_indexing(array):
    b = array[3, 4, 1, 2]
    assert isinstance(b, type(array))
    assert b.shape == (2, )

    with pytest.raises(IndexError):
        _ = array[1, 2, 4, 3]


def test_simple_quint_indexing(array):
    b = array[3, 4, 1, 2, 0]
    assert isinstance(b, type(array))
    assert b.shape == ()

    with pytest.raises(IndexError):
        _ = array[1, 2, 4, 2, 2]
