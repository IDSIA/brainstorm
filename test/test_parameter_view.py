#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np
import pytest

from brainstorm.structure.buffers import ParameterView


@pytest.fixture
def param_view():
    structure = [
        ('W', (5, 3)),
        ('R', (3, 3)),
        ('b', (3,))
    ]
    buffer = np.zeros(27)
    return ParameterView(structure, buffer)


def test_param_view_retains_full_buffer(param_view):
    assert param_view._buffer.size == 27
    assert isinstance(param_view._buffer, np.ndarray)


def test_param_view_buffer_slicing():
    structure = [
        ('W', (3, 2)),
        ('R', (2, 2)),
        ('b', (1,))
    ]
    buffer = np.arange(11)
    pv = ParameterView(structure, buffer)
    assert pv._buffer is buffer
    assert np.all(pv[0] == np.array([0, 1, 2, 3, 4, 5]).reshape(3, 2))
    assert np.all(pv[1] == np.array([6, 7, 8, 9]).reshape(2, 2))
    assert np.all(pv[2] == np.array([10]))


def test_empty_list_param_view():
    buff = np.zeros(0)
    empty = ParameterView([], buff)
    assert empty._buffer is buff
    assert empty._names == ()
    assert set(empty.__dict__.keys()) == {'_buffer', '_names'}


def test_param_view_tuple_upacking(param_view):
    W, R, b = param_view
    assert isinstance(W, np.ndarray)
    assert W.shape == (5, 3)
    assert isinstance(R, np.ndarray)
    assert R.shape == (3, 3)
    assert isinstance(b, np.ndarray)
    assert b.shape == (3,)


def test_param_view_len(param_view):
    assert len(param_view) == 3


def test_param_view_tuple_getitem(param_view):
    W, R, b = param_view
    assert param_view[0] is W
    assert param_view[1] is R
    assert param_view[2] is b

    with pytest.raises(IndexError):
        _ = param_view[3]


def test_param_view_dict_getitem(param_view):
    W, R, b = param_view
    assert param_view['W'] is W
    assert param_view['R'] is R
    assert param_view['b'] is b

    with pytest.raises(KeyError):
        _ = param_view['nonexisting']


def test_param_view_getattr(param_view):
    W, R, b = param_view
    assert param_view.W is W
    assert param_view.R is R
    assert param_view.b is b

    with pytest.raises(AttributeError):
        _ = param_view.nonexisting


def test_param_view_dict_for_autocompletion(param_view):
    assert set(param_view.__dict__.keys()) == {'W', 'R', 'b', '_buffer',
                                               '_names'}


def test_param_view_as_dict(param_view):
    assert param_view._asdict() == {
        'W': param_view.W,
        'R': param_view.R,
        'b': param_view.b
    }


def test_param_view_dict_interface(param_view):
    assert list(param_view.keys()) == list(param_view._asdict().keys())
    assert list(param_view.values()) == list(param_view._asdict().values())
    assert list(param_view.items()) == list(param_view._asdict().items())