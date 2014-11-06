#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from collections import OrderedDict
import numpy as np
import pytest
from brainstorm.buffers import ParameterBuffer, InOutBuffer, BufferManager
from brainstorm.layout import ParameterLayout
from mock import Mock, MagicMock, call


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
    mem = memory_mock(param_layout.size)
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


def test_inoutbuffer_initializes_empty(inoutlayout):
    hub_sizes, source_layouts, sink_layouts = inoutlayout
    iob = InOutBuffer(hub_sizes, source_layouts)
    assert iob.size == 0
    assert iob.memory is None
    assert iob.shape is None
    assert not iob.keys()
    assert not iob.values()
    assert not iob.items()


def test_inoutbuffer_rearrange_default(inoutlayout):
    hub_sizes, source_layouts, sink_layouts = inoutlayout
    iob = InOutBuffer(hub_sizes, source_layouts)
    iob.rearrange((1, 1))

    assert iob.size == sum(hub_sizes)
    assert isinstance(iob.memory, np.ndarray)
    assert iob.memory.size == iob.size
    assert iob.shape == (1, 1)


def test_inoutbuffer_rearrage_uses_passed_memory(inoutlayout):
    hub_sizes, source_layouts, sink_layouts = inoutlayout
    iob = InOutBuffer(hub_sizes, source_layouts)
    mem = memory_mock(26)
    iob.rearrange((1, 1), mem)
    assert iob.memory is mem


def test_inoutbuffer_rearrage_raises_on_unsufficient_memory(inoutlayout):
    hub_sizes, source_layouts, sink_layouts = inoutlayout
    iob = InOutBuffer(hub_sizes, source_layouts)
    mem = memory_mock(13)
    with pytest.raises(AssertionError):
        iob.rearrange((1, 1), mem)


def test_inoutbuffer_rearrage_does_not_raise_on_too_much_memory(inoutlayout):
    hub_sizes, source_layouts, sink_layouts = inoutlayout
    iob = InOutBuffer(hub_sizes, source_layouts)
    mem = memory_mock(100)
    iob.rearrange((1, 1), mem)


def test_inoutbuffer_rearrage_memory_interface(inoutlayout):
    hub_sizes, source_layouts, sink_layouts = inoutlayout
    iob = InOutBuffer(hub_sizes, source_layouts)
    mem = memory_mock(26)
    iob.rearrange((1, 1), mem)

    calls = [call(slice(0, 3)), call(slice(3, 15)), call(slice(15, 26))]
    mem.__getitem__.assert_has_calls(calls, any_order=True)


def test_inoutbuffer_rearrange_ignore_high_shape_dims(inoutlayout):
    hub_sizes, source_layouts, sink_layouts = inoutlayout
    iob = InOutBuffer(hub_sizes, source_layouts)
    iob.rearrange((2, 3, 5))

    assert iob.size == 3*2*3 + 12*2*3 + 11*2*3
    assert isinstance(iob.memory, np.ndarray)
    assert iob.shape == (2, 3)


def test_inoutbuffer_layout(inoutlayout):
    hub_sizes, source_layouts, sink_layouts = inoutlayout
    iob = InOutBuffer(hub_sizes, source_layouts)
    iob.rearrange((2, 3))

    assert set(iob.keys()) == {'A', 'B', 'C', 'D'}
    assert iob['A'].shape == (2, 3, 3)
    assert iob['B'].shape == (2, 3, 5)
    assert iob['C'].shape == (2, 3, 7)
    assert iob['D'].shape == (2, 3, 11)


def test_inoutbuffer_rearrange_is_lazy(inoutlayout):
    hub_sizes, source_layouts, sink_layouts = inoutlayout
    iob = InOutBuffer(hub_sizes, source_layouts)
    iob.rearrange((2, 3))
    iobA = iob['A']
    iobB = iob['B']
    iobC = iob['C']
    iobD = iob['D']
    m = iob.memory
    iob.rearrange((2, 3))
    assert iob['A'] is iobA
    assert iob['B'] is iobB
    assert iob['C'] is iobC
    assert iob['D'] is iobD
    assert iob.memory is m


def test_inoutbuffer_rearrange_is_lazy_if_smaller(inoutlayout):
    hub_sizes, source_layouts, sink_layouts = inoutlayout
    iob = InOutBuffer(hub_sizes, source_layouts)
    iob.rearrange((2, 3))
    m = iob.memory
    iob.rearrange((1, 2))
    assert iob.memory is m


# ###################### BufferManager #######################################

@pytest.fixture
def buff_man():
    pb = Mock()
    sink_buf = Mock()
    source_buf = Mock()
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
    buff_man.rearrange((2, 3, 17))
    assert not buff_man.parameters.method_calls
    buff_man.inputs.rearrange.assert_called_once_with((2, 3))
    buff_man.outputs.rearrange.assert_called_once_with((2, 3),
                                                       buff_man.inputs.memory)


def test_buffermanager_rearranges_lazily(buff_man):
    buff_man.rearrange((2, 3, 17))
    buff_man.inputs.rearrange.reset_mock()
    buff_man.outputs.rearrange.reset_mock()
    buff_man.rearrange((2, 3, 26))
    assert not buff_man.inputs.rearrange.called
    assert not buff_man.outputs.rearrange.called


