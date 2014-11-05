#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from collections import OrderedDict
import numpy as np
import pytest
from brainstorm.buffers import ParameterBuffer
from brainstorm.layout import ParameterLayout
from mock import MagicMock, call


####################### ParameterBuffer #######################################

@pytest.fixture
def param_layout():
    layout = OrderedDict()
    layout['A'] = slice(0, 5)
    layout['B'] = slice(5, 12)
    layout['C'] = slice(12, 23)
    return ParameterLayout(23, layout)


@pytest.fixture
def view_factories():
    return {
        'A': lambda x: 10,
        'B': lambda x: 100,
        'C': lambda x: 1000,
    }


def test_parameter_buffer_allocates_memory(param_layout, view_factories):
    pb = ParameterBuffer(param_layout, view_factories)
    assert isinstance(pb.memory, np.ndarray)
    assert pb.memory.size == param_layout.size


def test_parameter_buffer_wrong_memory_size_raises(param_layout,
                                                   view_factories):
    with pytest.raises(AssertionError):
        pb = ParameterBuffer(param_layout, view_factories, np.zeros(19))

    with pytest.raises(AssertionError):
        pb = ParameterBuffer(param_layout, view_factories, np.zeros(100))


def test_parameter_buffer_uses_passed_memory(param_layout, view_factories):
    mem = np.zeros(23)
    pb = ParameterBuffer(param_layout, view_factories, mem)
    assert pb.memory is mem


def test_parameter_buffer_memory_interface(param_layout, view_factories):
    mem = MagicMock(spec=['__getitem__'], size=param_layout.size)
    ParameterBuffer(param_layout, view_factories, mem)
    calls = [call(slice(0, 5)), call(slice(5, 12)), call(slice(12, 23))]
    mem.__getitem__.assert_has_calls(calls, any_order=True)


def test_parameter_buffer_dict_interface(param_layout, view_factories):
    pb = ParameterBuffer(param_layout, view_factories)
    assert set(pb.keys()) == {'A', 'B', 'C'}
    assert set(pb.values()) == {10, 100, 1000}
    assert set(pb.items()) == {('A', 10), ('B', 100), ('C', 1000)}
    assert 'A' in pb
    assert 'F' not in pb
    assert pb['A'] == 10
    assert pb['B'] == 100


def test_parameter_buffer_get_raw(param_layout, view_factories):
    mem = np.zeros(23)
    pb = ParameterBuffer(param_layout, view_factories, mem)
    assert pb.get_raw() is mem


def test_parameter_buffer_get_raw_for_layer(param_layout, view_factories):
    mem = np.zeros(23)
    pb = ParameterBuffer(param_layout, view_factories, mem)
    assert isinstance(pb.get_raw('A'), np.ndarray)
    assert isinstance(pb.get_raw('B'), np.ndarray)
    assert isinstance(pb.get_raw('C'), np.ndarray)
    assert pb.get_raw('A').shape == (5,)
    assert pb.get_raw('B').shape == (7,)
    assert pb.get_raw('C').shape == (11,)
    assert pb.get_raw('A').base is mem
    assert pb.get_raw('B').base is mem
    assert pb.get_raw('C').base is mem
