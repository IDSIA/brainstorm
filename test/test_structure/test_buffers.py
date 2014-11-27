#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from collections import OrderedDict

import numpy as np
import pytest
from mock import Mock, MagicMock, call

from brainstorm.structure.buffers import (ParameterBuffer, InOutBuffer,
                                          BufferManager)
from brainstorm.structure.layout import ParameterLayout


# ###################### Memory Mock ######################################

def get_item_mock(item):
    return memory_mock()


def reshape_mock(*args):
    return memory_mock(*args)


def memory_mock(*shape):
    mem = MagicMock(spec=['__getitem__', '__len__', 'reshape'], shape=shape)
    mem.__len__.return_value = shape[0] if shape else 0
    mem.__getitem__ = Mock(wraps=get_item_mock)
    mem.reshape = Mock(wraps=reshape_mock)
    return mem


# ###################### ParameterBuffer ######################################

@pytest.fixture
def param_buf():
    layout = OrderedDict()
    layout['A'] = slice(0, 5)
    layout['B'] = slice(5, 12)
    layout['C'] = slice(12, 23)
    param_layout = ParameterLayout(23, layout)
    view_factories = {
        'A': lambda x: 10,
        'B': lambda x: 100,
        'C': lambda x: 1000,
    }
    return ParameterBuffer(param_layout, view_factories)


def test_parameter_buffer_initializes_empty(param_buf):
    assert param_buf.memory is None
    assert not param_buf.keys()
    assert not param_buf.values()
    assert not param_buf.items()


def test_parameter_buffer_uses_passed_memory(param_buf):
    mem = memory_mock(23)
    param_buf.rearrange(mem)
    assert param_buf.memory is mem


def test_parameter_buffer_rearrange_too_small_memory_raises(param_buf):
    with pytest.raises(AssertionError):
        param_buf.rearrange(np.zeros(19))


def test_parameter_buffer_rearrange_too_big_memory_raises(param_buf):
    with pytest.raises(AssertionError):
        param_buf.rearrange(np.zeros(100))


def test_parameter_buffer_repeated_rearrange_is_ignored(param_buf):
    mem = memory_mock(23)
    param_buf.rearrange(mem)
    layouts = dict(param_buf)
    param_buf.rearrange(mem)
    assert dict(param_buf) == layouts


def test_parameter_buffer_repeated_rearrange_with_same_memory_is_ignored(
        param_buf):
    mem = memory_mock(23)
    param_buf.rearrange(mem)
    mem = param_buf.memory
    layouts = dict(param_buf)
    param_buf.rearrange(mem)
    assert dict(param_buf) == layouts


def test_parameter_buffer_memory_interface(param_buf):
    mem = memory_mock(23)
    param_buf.rearrange(mem)
    calls = [call(slice(0, 5)), call(slice(5, 12)), call(slice(12, 23))]
    mem.__getitem__.assert_has_calls(calls, any_order=True)


def test_parameter_buffer_dict_interface(param_buf):
    mem = memory_mock(23)
    param_buf.rearrange(mem)
    assert set(param_buf.keys()) == {'A', 'B', 'C'}
    assert set(param_buf.values()) == {10, 100, 1000}
    assert set(param_buf.items()) == {('A', 10), ('B', 100), ('C', 1000)}
    assert 'A' in param_buf
    assert 'F' not in param_buf
    assert param_buf['A'] == 10
    assert param_buf['B'] == 100


def test_parameter_buffer_get_raw(param_buf):
    mem = np.zeros(23)
    param_buf.rearrange(mem)
    assert param_buf.get_raw() is mem


def test_parameter_buffer_get_raw_for_layer(param_buf):
    mem = np.zeros(23)
    param_buf.rearrange(mem)
    assert isinstance(param_buf.get_raw('A'), np.ndarray)
    assert isinstance(param_buf.get_raw('B'), np.ndarray)
    assert isinstance(param_buf.get_raw('C'), np.ndarray)
    assert param_buf.get_raw('A').shape == (5,)
    assert param_buf.get_raw('B').shape == (7,)
    assert param_buf.get_raw('C').shape == (11,)
    assert param_buf.get_raw('A').base is mem
    assert param_buf.get_raw('B').base is mem
    assert param_buf.get_raw('C').base is mem


# ###################### InOutBuffer #######################################

@pytest.fixture
def inoutlayout():
    hub_sizes = [3, 12, 11]
    layout1 = OrderedDict([('A', slice(0, 3))])
    layout2 = OrderedDict([('B', slice(0, 5)), ('C', slice(5, 12))])
    layout3 = OrderedDict([('D', slice(0, 11))])
    source_layouts = [layout1, layout2, layout3]
    layout4 = OrderedDict([('B', slice(0, 3)), ('C', slice(0, 3))])
    layout5 = OrderedDict([('D', slice(0, 12))])
    layout6 = OrderedDict([])
    sink_layouts = [layout4, layout5, layout6]
    return hub_sizes, source_layouts, sink_layouts


@pytest.fixture
def inoutbuffer(inoutlayout):
    hub_sizes, source_layouts, sink_layouts = inoutlayout
    return InOutBuffer(hub_sizes, source_layouts)


def test_inoutbuffer_initializes_empty(inoutbuffer):
    assert inoutbuffer.size == 0
    assert inoutbuffer.memory is None
    assert inoutbuffer.shape is None
    assert not inoutbuffer.keys()
    assert not inoutbuffer.values()
    assert not inoutbuffer.items()


def test_inoutbuffer_without_memory_raises(inoutbuffer):
    with pytest.raises(AssertionError):
        inoutbuffer.rearrange((1, 1))


def test_inoutbuffer_rearrage_uses_passed_memory(inoutbuffer):
    mem = memory_mock(26)
    inoutbuffer.rearrange((1, 1), mem)
    assert inoutbuffer.memory is mem


def test_inoutbuffer_rearrage_raises_on_unsufficient_memory(inoutbuffer):
    mem = memory_mock(13)
    with pytest.raises(AssertionError):
        inoutbuffer.rearrange((1, 1), mem)


def test_inoutbuffer_rearrage_does_not_raise_on_too_much_memory(inoutbuffer):
    mem = memory_mock(100)
    inoutbuffer.rearrange((1, 1), mem)


def test_inoutbuffer_rearrage_memory_interface(inoutbuffer):
    mem = memory_mock(26)
    inoutbuffer.rearrange((1, 1), mem)

    calls = [call(slice(0, 3)), call(slice(3, 15)), call(slice(15, 26))]
    mem.__getitem__.assert_has_calls(calls, any_order=True)


def test_inoutbuffer_rearrange_ignore_high_shape_dims(inoutbuffer):
    mem = memory_mock(156)
    inoutbuffer.rearrange((2, 3, 5), mem)

    assert inoutbuffer.size == 3*2*3 + 12*2*3 + 11*2*3
    assert inoutbuffer.memory is mem
    assert inoutbuffer.shape == (2, 3)


def test_inoutbuffer_layout(inoutbuffer):
    mem = np.zeros(156)
    inoutbuffer.rearrange((2, 3), mem)

    assert set(inoutbuffer.keys()) == {'A', 'B', 'C', 'D'}
    assert inoutbuffer['A'].shape == (2, 3, 3)
    assert inoutbuffer['B'].shape == (2, 3, 5)
    assert inoutbuffer['C'].shape == (2, 3, 7)
    assert inoutbuffer['D'].shape == (2, 3, 11)


def test_inoutbuffer_rearrange_is_lazy(inoutbuffer):
    mem = memory_mock(156)
    inoutbuffer.rearrange((2, 3), mem)
    layouts = dict(inoutbuffer)
    inoutbuffer.rearrange((2, 3))
    assert dict(inoutbuffer) == layouts
    assert inoutbuffer.memory is mem


def test_inoutbuffer_rearrange_with_identical_memory_is_lazy(inoutbuffer):
    mem = memory_mock(26*2*3)
    inoutbuffer.rearrange((2, 3), mem)
    layouts = dict(inoutbuffer)
    m = inoutbuffer.memory
    inoutbuffer.rearrange((2, 3), mem)
    assert dict(inoutbuffer) == layouts
    assert inoutbuffer.memory is m


def test_inoutbuffer_rearrange_with_different_memory_updates(inoutbuffer):
    mem = memory_mock(26*2*3)
    inoutbuffer.rearrange((2, 3), mem)
    layouts = dict(inoutbuffer)
    mem2 = memory_mock(26*2*3)
    inoutbuffer.rearrange((2, 3), mem2)
    assert inoutbuffer.memory is mem2
    assert dict(inoutbuffer) != layouts


def test_inoutbuffer_rearrange_is_lazy_if_smaller(inoutbuffer):
    mem = memory_mock(156)
    inoutbuffer.rearrange((2, 3), mem)
    inoutbuffer.rearrange((1, 2))
    assert inoutbuffer.memory is mem


# ###################### BufferManager #######################################

@pytest.fixture
def buff_man():
    pb = Mock(size=10)
    sink_buf = Mock()
    source_buf = Mock()
    sink_buf.get_size = lambda s: 17
    source_buf.get_size = lambda s: 17
    return BufferManager(pb, sink_buf, source_buf)


def test_buffermanager_construction():
    pb = Mock()
    sink_buf = Mock()
    source_buf = Mock()
    bm = BufferManager(pb, sink_buf, source_buf)

    assert bm.parameters is pb
    assert bm.inputs is sink_buf
    assert bm.outputs is source_buf


def test_buffermanager_rearranges_in_out_buffers(buff_man):
    buff_man.rearrange_fwd((2, 3, 17))
    assert buff_man.inputs.rearrange.called
    assert buff_man.outputs.rearrange.called


def test_buffermanager_rearranges_lazily(buff_man):
    buff_man.rearrange_fwd((2, 3, 17))
    buff_man.parameters.rearrange.reset_mock()
    buff_man.inputs.rearrange.reset_mock()
    buff_man.outputs.rearrange.reset_mock()
    buff_man.rearrange_fwd((2, 3, 26))
    assert not buff_man.parameters.rearrange.called
    assert not buff_man.inputs.rearrange.called
    assert not buff_man.outputs.rearrange.called


def test_create_from_layers(layers):
    bm = BufferManager.create_from_layers(layers)
    bm.rearrange_parameters()
    bm.rearrange_fwd((1, 1))
    assert set(bm.parameters.keys()) == {'InputLayer', 'A', 'B', 'C', 'D'}
    assert set(bm.inputs.keys()) == {'A', 'B', 'C', 'D'}
    assert set(bm.outputs.keys()) == {'InputLayer', 'A', 'B', 'C', 'D'}

    assert bm.inputs.hub_sizes == bm.outputs.hub_sizes
