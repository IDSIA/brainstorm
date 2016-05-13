#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

import numpy as np

from brainstorm.handlers.base_handler import Handler


# ############################## Debug Array ################################ #

class DebugArray(object):
    def __init__(self, arr):
        assert arr is not None
        self.shape = arr.shape
        self.array = arr
        self.size = self.array.size

    def __getitem__(self, item):
        if isinstance(item, (int, slice)):
            item = tuple([item])
        assert isinstance(item, tuple)
        for i in item:
            assert isinstance(i, (int, slice))
            if isinstance(i, slice):
                assert i.step is None
        return DebugArray(arr=self.array.__getitem__(item))

    def reshape(self, new_shape):
        if isinstance(new_shape, (tuple, list)):
            assert all([t >= 0 for t in tuple(new_shape)])
        else:
            assert isinstance(new_shape, int)
            assert new_shape >= 0
        return DebugArray(arr=self.array.reshape(new_shape))


def _check_for_inf(handler, arg, name):
    if isinstance(arg, (int, float)) and not np.isfinite(arg):
        raise ValueError('NaN or Inf encountered in "{}" argument'
                         .format(name))
    if isinstance(arg, DebugArray) and not handler.is_fully_finite(arg):
        raise ValueError('NaN or Inf encountered in "{}"'.format(name))


def check_for_inf_or_nan(f):
    def checked_f(*args, **kwargs):
        result = f(*args, **kwargs)
        handler = args[0]
        for i, arg in enumerate(args, start=1):
            _check_for_inf(handler, arg, '{}.'.format(i))
        for n, v in kwargs.items():
            _check_for_inf(handler, v, n)
        return result

    return checked_f


# ############################# Debug Handler ############################### #

class DebugHandler(Handler):
    __undescribed__ = {'EMPTY', 'array_type'}

    def __init__(self, handler):
        super(DebugHandler, self).__init__()
        self.handler = handler
        self.EMPTY = DebugArray(arr=handler.EMPTY)
        self.array_type = DebugArray

    def __init_from_description__(self, description):
        self.__init__(self.handler)

    # ------------------------- Allocate new memory ------------------------- #

    def allocate(self, shape):
        assert_is_shape(shape)
        return DebugArray(self.handler.allocate(shape))

    def ones(self, shape):
        assert_is_shape(shape)
        return DebugArray(self.handler.ones(shape))

    def zeros(self, shape):
        assert_is_shape(shape)
        return DebugArray(self.handler.zeros(shape))

    # ---------------------------- Copy and Fill ---------------------------- #

    @check_for_inf_or_nan
    def copy_to(self, src, dest):
        assert_debug_arrays(dest, src)
        assert_shapes_equal(dest, src)
        assert dest.size == src.size, "{} != {}".format(dest.size, src.size)
        self.handler.copy_to(src.array, dest.array)

    @check_for_inf_or_nan
    def copy_to_if(self, src, dest, cond):
        assert_debug_arrays(src, dest, cond)
        assert_shapes_equal(src, dest, cond)
        self.handler.copy_to_if(src.array, dest.array, cond.array)

    @check_for_inf_or_nan
    def create_from_numpy(self, arr):
        assert isinstance(arr, np.ndarray)
        return DebugArray(self.handler.create_from_numpy(arr))

    @check_for_inf_or_nan
    def fill(self, mem, val):
        assert_debug_arrays(mem)
        assert_is_scalar(val)
        self.handler.fill(mem.array, val)

    @check_for_inf_or_nan
    def fill_if(self, mem, val, cond):
        assert_is_scalar(val)
        assert_debug_arrays(mem, cond)
        assert_shapes_equal(mem, cond)
        self.handler.fill_if(mem.array, val, cond.array)

    @check_for_inf_or_nan
    def get_numpy_copy(self, mem):
        assert_debug_arrays(mem)
        return self.handler.get_numpy_copy(mem.array)

    @check_for_inf_or_nan
    def set_from_numpy(self, mem, arr):
        assert_debug_arrays(mem)
        assert isinstance(arr, np.ndarray)
        assert mem.shape == arr.shape, \
            "{} != {}".format(mem.shape, arr.shape)
        self.handler.set_from_numpy(mem.array, arr)

    # ---------------------------- Debug helpers ---------------------------- #

    def is_fully_finite(self, a):
        return self.handler.is_fully_finite(a.array)

    # ----------------------- Mathematical operations ----------------------- #

    @check_for_inf_or_nan
    def abs_t(self, a, out):
        assert_debug_arrays(a, out)
        assert_shapes_equal(a, out)
        self.handler.abs_t(a.array, out.array)

    @check_for_inf_or_nan
    def add_into_if(self, a, out, cond):
        assert_debug_arrays(a, out, cond)
        assert_shapes_equal(a, out, cond)
        self.handler.add_into_if(a.array, out.array, cond.array)

    @check_for_inf_or_nan
    def add_mv(self, m, v, out):
        assert_debug_arrays(m, v, out)
        assert_shapes_equal(m, out)
        assert len(m.shape) == 2, "len({}) != 2".format(m.shape)
        assert v.shape == (m.shape[0], 1) or v.shape == (1, m.shape[1]), \
            "invalid shape {}".format(v.shape)
        self.handler.add_mv(m.array, v.array, out.array)

    @check_for_inf_or_nan
    def add_st(self, s, t, out):
        assert_debug_arrays(t, out)
        assert_is_scalar(s)
        assert_shapes_equal(t, out)
        self.handler.add_st(s, t.array, out.array)

    @check_for_inf_or_nan
    def add_tt(self, a, b, out):
        assert_debug_arrays(a, b, out)
        assert_shapes_equal(a, b, out)
        self.handler.add_tt(a.array, b.array, out.array)

    @check_for_inf_or_nan
    def avgpool2d_backward_batch(self, inputs, window, outputs, padding,
                                 stride, in_deltas, out_deltas):
        assert_debug_arrays(inputs, outputs, in_deltas, out_deltas)
        assert_is_shape(window)
        assert len(window) == 2, "len({}) != 2".format(window)
        assert_is_shape(stride)
        assert len(stride) == 2, "len({}) != 2".format(stride)
        assert isinstance(padding, int) and 0 <= padding, \
            "invalid padding {}".format(padding)
        assert_shapes_equal(inputs, in_deltas)
        assert_shapes_equal(outputs, out_deltas)
        # TODO: check shapes of inputs, outputs
        self.handler.avgpool2d_backward_batch(inputs.array, window,
                                              outputs.array, padding, stride,
                                              in_deltas.array,
                                              out_deltas.array)

    @check_for_inf_or_nan
    def avgpool2d_forward_batch(self, inputs, window, outputs, padding,
                                stride):
        assert_debug_arrays(inputs, outputs)
        assert_is_shape(window)
        assert len(window) == 2, "len({}) != 2".format(window)
        assert_is_shape(stride)
        assert len(stride) == 2, "len({}) != 2".format(stride)
        assert isinstance(padding, int) and 0 <= padding, \
            "invalid padding {}".format(padding)
        # TODO: check shapes of inputs, outputs,
        self.handler.avgpool2d_forward_batch(inputs.array, window,
                                             outputs.array, padding, stride)

    @check_for_inf_or_nan
    def binarize_v(self, v, out):
        assert_debug_arrays(v, out)
        assert len(v.shape) == len(out.shape) == 2
        assert v.shape == (out.shape[0], 1)
        assert self.handler.get_numpy_copy(v.array).min() >= 0
        assert int(self.handler.get_numpy_copy(v.array).max()) < out.shape[1]
        self.handler.binarize_v(v.array, out.array)

    @check_for_inf_or_nan
    def broadcast_t(self, a, axis, out):
        assert_debug_arrays(a, out)
        assert (isinstance(axis, int) and 0 <= axis < len(out.shape)),\
            "invalid axis {}".format(axis)
        assert a.shape[axis] == 1
        assert a.shape == out.shape[:axis] + (1,) + out.shape[axis+1:]
        self.handler.broadcast_t(a.array, axis, out.array)

    @check_for_inf_or_nan
    def clip_t(self, a, a_min, a_max, out):
        assert_debug_arrays(a, out)
        assert_is_scalar(a_min)
        assert_is_scalar(a_max)
        assert_shapes_equal(a, out)
        assert a_min <= a_max, "not {} <= {}".format(a_min, a_max)
        self.handler.clip_t(a.array, a_min, a_max, out.array)

    @check_for_inf_or_nan
    def conv2d_backward_batch(self, inputs, weights, padding, stride,
                              in_deltas, out_deltas, weight_deltas,
                              bias_deltas):
        assert_debug_arrays(inputs, weights, in_deltas, out_deltas,
                            weight_deltas, bias_deltas)
        assert isinstance(padding, int) and 0 <= padding, \
            "invalid padding {}".format(padding)
        assert_is_shape(stride)
        assert len(stride) == 2, "len({}) != 2".format(stride)
        # TODO: check shapes of inputs, weights, in_deltas, out_deltas,
        # TODO: weight_deltas, bias_deltas
        self.handler.conv2d_backward_batch(inputs.array, weights.array,
                                           padding, stride, in_deltas.array,
                                           out_deltas.array,
                                           weight_deltas.array,
                                           bias_deltas.array)

    @check_for_inf_or_nan
    def conv2d_forward_batch(self, inputs, weights, bias, outputs,
                             padding, stride):
        assert_debug_arrays(inputs, weights, bias, outputs)
        assert isinstance(padding, int) and 0 <= padding, \
            "invalid padding {}".format(padding)
        assert_is_shape(stride)
        assert len(stride) == 2, "len({}) != 2".format(stride)
        # TODO check shapes of inputs, weights, bias, and outputs
        self.handler.conv2d_forward_batch(inputs.array, weights.array,
                                          bias.array, outputs.array,
                                          padding, stride)

    @check_for_inf_or_nan
    def dot_add_mm(self, a, b, out, transa=False, transb=False):
        assert_debug_arrays(a, b, out)
        assert len(a.shape) == 2, "len({}) != 2".format(a.shape)
        assert len(b.shape) == 2, "len({}) != 2".format(b.shape)
        assert len(out.shape) == 2, "len({}) != 2".format(out.shape)
        assert transa in [True, False]
        assert transb in [True, False]
        a1, a2 = a.shape
        b1, b2 = b.shape
        if transa:
            a1, a2 = a2, a1
        if transb:
            b1, b2 = b2, b1
        assert a2 == b1, "{} != {} ({}, {})".format(a2, b1, transa, transb)
        assert out.shape == (a1, b2), "{} != {}".format(out.shape, (a1, b2))
        self.handler.dot_add_mm(a.array, b.array, out.array, transa, transb)

    @check_for_inf_or_nan
    def dot_mm(self, a, b, out, transa=False, transb=False):
        assert_debug_arrays(a, b, out)
        assert len(a.shape) == 2, "len({}) != 2".format(a.shape)
        assert len(b.shape) == 2, "len({}) != 2".format(b.shape)
        assert len(out.shape) == 2, "len({}) != 2".format(out.shape)
        assert transa in [True, False]
        assert transb in [True, False]
        a1, a2 = a.shape
        b1, b2 = b.shape

        if transa:
            a1, a2 = a2, a1
        if transb:
            b1, b2 = b2, b1

        assert a2 == b1, "{} != {} ({}, {})".format(a2, b1, transa, transb)
        assert out.shape == (a1, b2), "{} != {}".format(out.shape, (a1, b2))

        self.handler.dot_mm(a.array, b.array, out.array, transa, transb)

    @check_for_inf_or_nan
    def divide_mv(self, m, v, out):
        assert_debug_arrays(m, v, out)
        assert_shapes_equal(m, out)
        assert len(m.shape) == 2, "len({}) != 2".format(m.shape)
        assert v.shape == (m.shape[0], 1) or v.shape == (1, m.shape[1]), \
            "invalid shape {}".format(v.shape)
        self.handler.divide_mv(m.array, v.array, out.array)

    @check_for_inf_or_nan
    def divide_tt(self, a, b, out):
        assert_debug_arrays(a, b, out)
        assert_shapes_equal(a, b, out)
        self.handler.divide_tt(a.array, b.array, out.array)

    @check_for_inf_or_nan
    def fill_gaussian(self, mean, std, out):
        assert_debug_arrays(out)
        assert std >= 0.0
        self.handler.fill_gaussian(mean, std, out.array)

    @check_for_inf_or_nan
    def generate_probability_mask(self, mask, probability):
        assert_debug_arrays(mask)
        assert_is_scalar(probability)
        assert 0.0 <= probability <= 1.0, "{}".format(probability)
        self.handler.generate_probability_mask(mask.array, probability)

    @check_for_inf_or_nan
    def index_m_by_v(self, m, v, out):
        assert_debug_arrays(m, v, out)
        assert_shapes_equal(v, out)
        assert len(m.shape) == len(v.shape) == 2
        assert v.shape == (m.shape[0], 1)
        self.handler.index_m_by_v(m.array, v.array, out.array)

    @check_for_inf_or_nan
    def log_t(self, a, out):
        assert_debug_arrays(a, out)
        assert_shapes_equal(a, out)
        self.handler.log_t(a.array, out.array)

    @check_for_inf_or_nan
    def maxpool2d_backward_batch(self, inputs, window, outputs, padding,
                                 stride, argmax, in_deltas, out_deltas):
        assert_debug_arrays(inputs, outputs, argmax, in_deltas, out_deltas)
        assert_is_shape(window)
        assert len(window) == 2, "len({}) != 2".format(window)
        assert_is_shape(stride)
        assert len(stride) == 2, "len({}) != 2".format(stride)
        assert isinstance(padding, int) and 0 <= padding, \
            "invalid padding {}".format(padding)
        assert_shapes_equal(inputs, in_deltas)
        assert_shapes_equal(outputs, out_deltas)
        # TODO: check shapes of inputs, outputs, argmax
        self.handler.maxpool2d_backward_batch(inputs.array, window,
                                              outputs.array,
                                              padding, stride, argmax.array,
                                              in_deltas.array,
                                              out_deltas.array)

    @check_for_inf_or_nan
    def maxpool2d_forward_batch(self, inputs, window, outputs, padding,
                                stride, argmax):
        assert_debug_arrays(inputs, outputs, argmax)
        assert_is_shape(window)
        assert len(window) == 2, "len({}) != 2".format(window)
        assert_is_shape(stride)
        assert len(stride) == 2, "len({}) != 2".format(stride)
        assert isinstance(padding, int) and 0 <= padding, \
            "invalid padding {}".format(padding)
        # TODO: check shapes of inputs, outputs, argmax
        self.handler.maxpool2d_forward_batch(inputs.array, window,
                                             outputs.array, padding, stride,
                                             argmax.array)

    @check_for_inf_or_nan
    def merge_tt(self, a, b, out):
        assert(a.shape[-1] + b.shape[-1] == out.shape[-1])
        assert_debug_arrays(a, b, out)
        self.handler.merge_tt(a.array, b.array, out.array)

    @check_for_inf_or_nan
    def modulo_tt(self, a, b, out):
        assert_debug_arrays(a, b, out)
        assert_shapes_equal(a, b, out)
        self.handler.modulo_tt(a.array, b.array, out.array)

    @check_for_inf_or_nan
    def mult_add_st(self, s, t, out):
        assert_debug_arrays(t, out)
        assert_is_scalar(s)
        assert_shapes_equal(t, out)
        self.handler.mult_add_st(s, t.array, out.array)

    @check_for_inf_or_nan
    def mult_add_tt(self, a, b, out):
        assert_debug_arrays(a, b, out)
        assert_shapes_equal(a, b, out)
        self.handler.mult_add_tt(a.array, b.array, out.array)

    @check_for_inf_or_nan
    def mult_mv(self, m, v, out):
        assert_debug_arrays(m, v, out)
        assert_shapes_equal(m, out)
        assert len(m.shape) == 2, "len({}) != 2".format(m.shape)
        assert v.shape in [(m.shape[0], 1), (1, m.shape[1]), m.shape],\
            "invalid shape {} (m.shape = {})".format(v.shape, m.shape)
        self.handler.mult_mv(m.array, v.array, out.array)

    @check_for_inf_or_nan
    def mult_add_mv(self, m, v, out):
        assert_debug_arrays(m, v, out)
        assert_shapes_equal(m, out)
        assert len(m.shape) == 2, "len({}) != 2".format(m.shape)
        assert v.shape == (m.shape[0], 1) or v.shape == (1, m.shape[1]), \
            "invalid shape {}".format(v.shape)
        self.handler.mult_add_mv(m.array, v.array, out.array)

    @check_for_inf_or_nan
    def mult_st(self, s, t, out):
        assert_debug_arrays(t, out)
        assert_is_scalar(s)
        assert_shapes_equal(t, out)
        self.handler.mult_st(s, t.array, out.array)

    @check_for_inf_or_nan
    def mult_tt(self, a, b, out):
        assert_debug_arrays(a, b, out)
        assert_shapes_equal(a, b, out)
        self.handler.mult_tt(a.array, b.array, out.array)

    @check_for_inf_or_nan
    def sign_t(self, a, out):
        assert_debug_arrays(a, out)
        assert_shapes_equal(a, out)
        self.handler.sign_t(a.array, out.array)

    @check_for_inf_or_nan
    def split_add_tt(self, x, out_a, out_b):
        assert(out_a.shape[-1] + out_b.shape[-1] == x.shape[-1])
        assert_debug_arrays(out_a, out_b, x)
        self.handler.split_add_tt(x.array, out_a.array, out_b.array)

    @check_for_inf_or_nan
    def sqrt_t(self, a, out):
        assert_debug_arrays(a, out)
        assert_shapes_equal(a, out)
        self.handler.sqrt_t(a.array, out.array)

    @check_for_inf_or_nan
    def subtract_mv(self, m, v, out):
        assert_debug_arrays(m, v, out)
        assert_shapes_equal(m, out)
        assert len(m.shape) == 2, "len({}) != 2".format(m.shape)
        assert v.shape == (m.shape[0], 1) or v.shape == (1, m.shape[1]), \
            "invalid shape {}".format(v.shape)
        self.handler.subtract_mv(m.array, v.array, out.array)

    @check_for_inf_or_nan
    def subtract_tt(self, a, b, out):
        assert_debug_arrays(a, b, out)
        assert_shapes_equal(a, b, out)
        self.handler.subtract_tt(a.array, b.array, out.array)

    @check_for_inf_or_nan
    def sum_t(self, a, axis, out):
        assert_debug_arrays(a, out)
        dims = len(a.shape)
        assert axis is None or (isinstance(axis, int) and 0 <= axis < dims),\
            "invalid axis {}".format(axis)
        # TODO check shapes of a and out
        self.handler.sum_t(a.array, axis, out.array)

    # ------------------------ Activation functions ------------------------- #

    @check_for_inf_or_nan
    def sigmoid(self, x, y):
        assert_debug_arrays(x, y)
        assert_shapes_equal(x, y)
        self.handler.sigmoid(x.array, y.array)

    @check_for_inf_or_nan
    def sigmoid_deriv(self, x, y, dy, dx):
        assert_debug_arrays(y, dy, dx)
        assert_shapes_equal(y, dy, dx)
        if x is not None:
            assert_debug_arrays(x)
            assert_shapes_equal(x, y)
            x = x.array
        self.handler.sigmoid_deriv(x, y.array, dy.array, dx.array)

    @check_for_inf_or_nan
    def tanh(self, x, y):
        assert_debug_arrays(x, y)
        assert_shapes_equal(x, y)
        self.handler.tanh(x.array, y.array)

    @check_for_inf_or_nan
    def tanh_deriv(self, x, y, dy, dx):
        assert_debug_arrays(y, dy, dx)
        assert_shapes_equal(y, dy, dx)
        if x is not None:
            assert_debug_arrays(x)
            assert_shapes_equal(x, y)
            x = x.array
        self.handler.tanh_deriv(x, y.array, dy.array, dx.array)

    @check_for_inf_or_nan
    def rel(self, x, y):
        assert_debug_arrays(x, y)
        assert_shapes_equal(x, y)
        self.handler.rel(x.array, y.array)

    @check_for_inf_or_nan
    def rel_deriv(self, x, y, dy, dx):
        assert_debug_arrays(y, dy, dx)
        assert_shapes_equal(y, dy, dx)
        if x is not None:
            assert_debug_arrays(x)
            assert_shapes_equal(x, y)
            x = x.array
        self.handler.rel_deriv(x, y.array, dy.array, dx.array)

    @check_for_inf_or_nan
    def guided_rel_deriv(self, x, y, dy, dx):
        assert_debug_arrays(y, dy, dx)
        assert_shapes_equal(y, dy, dx)
        if x is not None:
            assert_debug_arrays(x)
            assert_shapes_equal(x, y)
            x = x.array
        self.handler.guided_rel_deriv(x, y.array, dy.array, dx.array)

    @check_for_inf_or_nan
    def el(self, x, y):
        assert_debug_arrays(x, y)
        assert_shapes_equal(x, y)
        self.handler.el(x.array, y.array)

    @check_for_inf_or_nan
    def el_deriv(self, x, y, dy, dx):
        assert_debug_arrays(y, dy, dx)
        assert_shapes_equal(y, dy, dx)
        if x is not None:
            assert_debug_arrays(x)
            assert_shapes_equal(x, y)
            x = x.array
        self.handler.el_deriv(x, y.array, dy.array, dx.array)

    @check_for_inf_or_nan
    def softplus(self, x, y):
        assert_debug_arrays(x, y)
        assert_shapes_equal(x, y)
        self.handler.rel(x.array, y.array)

    @check_for_inf_or_nan
    def softplus_deriv(self, x, y, dy, dx):
        assert_debug_arrays(y, dy, dx)
        assert_shapes_equal(y, dy, dx)
        if x is not None:
            assert_debug_arrays(x)
            assert_shapes_equal(x, y)
            x = x.array
        self.handler.rel_deriv(x, y.array, dy.array, dx.array)

    @check_for_inf_or_nan
    def softmax_m(self, m, out):
        assert_debug_arrays(m, out)
        assert_shapes_equal(m, out)
        assert len(m.shape) == 2, "len({}) != 2".format(m.shape)
        self.handler.softmax_m(m.array, out.array)


# ############################ Helper Methods ############################### #


def assert_is_shape(shape):
    assert isinstance(shape, tuple), type(shape)
    for i in shape:
        assert isinstance(i, int), "{} was {}".format(i, type(i))
        assert 0 <= i, "{} < 0".format(i)


def assert_debug_arrays(*arrays):
    for i, arr in enumerate(arrays):
        assert isinstance(arr, DebugArray), \
            "{}. is no DebugArray but a {}".format(i, type(arr))


def assert_shapes_equal(ref_shape, *shapes):
    if isinstance(ref_shape, DebugArray):
        ref_shape = ref_shape.shape
    assert_is_shape(ref_shape)
    for i, shape in enumerate(shapes, start=1):
        if isinstance(shape, DebugArray):
            shape = shape.shape
        assert_is_shape(shape)
        assert shape == ref_shape, \
            "Shape mismatch: {}[arg_nr={}] != {}[arg_nr=0]".format(shape, i,
                                                                   ref_shape)


def assert_is_scalar(s):
    assert isinstance(s, (int, float)), \
        "{} is not a scalar but a {}".format(s, type(s))
