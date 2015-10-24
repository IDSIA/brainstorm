#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

import numpy as np
import pytest

from brainstorm.structure.buffers import BufferView


@pytest.fixture
def buffer_view():
    buffer_names = ['W', 'R', 'b']
    full_buffer = np.zeros(27)
    buffers = [full_buffer[:15].reshape((5, 3)),
               full_buffer[15:24].reshape((3, 3)),
               full_buffer[24:27].reshape((3,))]
    return BufferView(buffer_names, buffers, full_buffer)


def test_buffer_view_retains_full_buffer(buffer_view):
    assert buffer_view._full_buffer.size == 27
    assert isinstance(buffer_view._full_buffer, np.ndarray)


def test_buffer_view_buffer_slicing():
    buffer_names = ['W', 'R', 'b']
    full_buffer = np.arange(11)
    buffers = [full_buffer[:6].reshape((3, 2)),
               full_buffer[6:10].reshape((2, 2)),
               full_buffer[10:].reshape((1,))]
    bv = BufferView(buffer_names, buffers, full_buffer)

    assert bv._full_buffer is full_buffer
    assert np.all(bv[0] == np.array([0, 1, 2, 3, 4, 5]).reshape(3, 2))
    assert np.all(bv[1] == np.array([6, 7, 8, 9]).reshape(2, 2))
    assert np.all(bv[2] == np.array([10]))


def test_empty_list_param_view():
    buff = np.zeros(0)
    empty = BufferView([], [], buff)
    assert empty._full_buffer is buff
    assert empty._buffer_names == ()
    assert set(empty.__dict__.keys()) == {'_full_buffer', '_buffer_names',
                                          '_keys'}


def test_buffer_view_tuple_upacking(buffer_view):
    W, R, b = buffer_view
    assert isinstance(W, np.ndarray)
    assert W.shape == (5, 3)
    assert isinstance(R, np.ndarray)
    assert R.shape == (3, 3)
    assert isinstance(b, np.ndarray)
    assert b.shape == (3,)


def test_buffer_view_len(buffer_view):
    assert len(buffer_view) == 3


def test_buffer_view_tuple_getitem(buffer_view):
    W, R, b = buffer_view
    assert buffer_view[0] is W
    assert buffer_view[1] is R
    assert buffer_view[2] is b

    with pytest.raises(IndexError):
        _ = buffer_view[3]


def test_buffer_view_dict_getitem(buffer_view):
    W, R, b = buffer_view
    assert buffer_view['W'] is W
    assert buffer_view['R'] is R
    assert buffer_view['b'] is b

    with pytest.raises(KeyError):
        _ = buffer_view['nonexisting']


def test_buffer_view_contains(buffer_view):

    assert 'W' in buffer_view
    assert 'R' in buffer_view
    assert 'b' in buffer_view

    assert 'foo' not in buffer_view
    assert 'Q' not in buffer_view


def test_buffer_view_getattr(buffer_view):
    W, R, b = buffer_view
    assert buffer_view.W is W
    assert buffer_view.R is R
    assert buffer_view.b is b

    with pytest.raises(AttributeError):
        _ = buffer_view.nonexisting


def test_buffer_view_dict_for_autocompletion(buffer_view):
    assert set(buffer_view.__dict__.keys()) == {'W', 'R', 'b', '_buffer_names',
                                                '_full_buffer', '_keys'}


def test_buffer_view_as_dict(buffer_view):
    assert buffer_view._asdict() == {
        'W': buffer_view.W,
        'R': buffer_view.R,
        'b': buffer_view.b
    }


def test_buffer_view_dict_interface(buffer_view):
    assert list(buffer_view.keys()) == list(buffer_view._asdict().keys())
    assert list(buffer_view.values()) == list(buffer_view._asdict().values())
    assert list(buffer_view.items()) == list(buffer_view._asdict().items())


def test_deep_copying_of_buffer_view(buffer_view):
    foo_names = ['a', 'b']
    foo_buffers = [np.ones(2), np.zeros(3)]
    foo_view = BufferView(foo_names, foo_buffers)

    names = ['other', 'foo']
    buffers = [buffer_view, foo_view]
    my_buffer = BufferView(names, buffers)

    from copy import deepcopy

    my_buffer_copy = deepcopy(my_buffer)
    assert my_buffer_copy.foo is not foo_view
    assert my_buffer_copy.other is not buffer_view

    foo_view.a[:] = 7
    assert np.all(my_buffer_copy.foo.a == np.ones(2))
