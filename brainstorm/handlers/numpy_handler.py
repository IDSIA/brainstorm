#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np


class NumpyHandler(object):
    def __init__(self, dtype):
        self.array_type = np.ndarray
        self.dtype = dtype
        self.size = lambda x: x.size
        self.shape = lambda x: x.shape
        self.reshape = lambda x, s: x.reshape(s)
        self.slice = lambda x, s: x[s]
        self.context = 'numpy'
        self.EMPTY = np.zeros(0)

    def allocate(self, size):
        return np.zeros(size, dtype=self.dtype)

    @staticmethod
    def fill(mem, val):
        mem.fill(val)

    def set_from_numpy(self, mem, arr):
        mem[:] = arr.astype(self.dtype)

    def get_numpy_copy(self, mem):
        assert type(mem) == self.array_type
        return mem.copy()

    @staticmethod
    def copy_to(dest, src):
        # FIXME: change casting to 'no'
        np.copyto(dest, src, casting='same_kind')

    def zeros(self, shape):
        return np.zeros(shape=shape, dtype=self.dtype)

    def ones(self, shape):
        return np.ones(shape=shape, dtype=self.dtype)

    # ---------------- Mathematical Operations ---------------- #

    def sum_t(self, a, axis, out):
        if len(out.shape) == len(a.shape):
            keepdims = True
        else:
            keepdims = False
        np.sum(a, axis=axis, dtype=self.dtype, out=out, keepdims=keepdims)

    @staticmethod
    def dot_mm(a, b, out, transa='N', transb='N'):
        x = a.T if (transa == 'T') else a
        y = b.T if (transb == 'T') else b
        # np.dot(x, y, out)  # FIXME: doesn't work with strided out
        out[:] = np.dot(x, y)

    @staticmethod
    def dot_add_mm(a, b, out, transa='N', transb='N'):
        x = a.T if (transa == 'T') else a
        y = b.T if (transb == 'T') else b
        out[:] += np.dot(x, y)

    @staticmethod
    def elem_mult_tt(a, b, out):
        np.multiply(a, b, out)

    @staticmethod
    def elem_mult_st(a, b, out):
        np.multiply(a, b, out)

    @staticmethod
    def add_tt(a, b, out):
        assert a.shape == b.shape == out.shape
        out[:] = a + b

    @staticmethod
    def subtract_tt(a, b, out):
        assert a.shape == b.shape == out.shape
        out[:] = a - b

    @staticmethod
    def add_mv(a, b, out):
        # TODO: Generalize to support broadcast along both dimensions
        assert len(a.shape) == 2
        assert len(b.shape) == 1
        out[:] = a + b

    @staticmethod
    def broadcast_features_t(a, out):
        assert len(a.shape) == 3
        assert a.shape[2] == 1
        assert len(out.shape) > 2
        num_extra_dims = len(out.shape) - 3
        shape_to_add = tuple([1] * num_extra_dims)
        b = np.reshape(a, a.shape + shape_to_add)

        shape_to_tile = (1, 1) + out.shape[2:]
        out[:] = np.tile(b, shape_to_tile)

    @staticmethod
    def clip_t(a, a_min, a_max, out):
        np.clip(a, a_min, a_max, out)

    @staticmethod
    def log_t(a, out):
        np.log(a, out)

    @staticmethod
    def divide_tt(a, b, out):
        out[:] = a / b

    # Activation functions

    @staticmethod
    def sigmoid(x, y):
        y[:] = 1. / (1. + np.exp(-x))

    @staticmethod
    def sigmoid_deriv(x, y, dy, dx):
        dx[:] = dy * y * (1. - y)

    @staticmethod
    def tanh(x, y):
        np.tanh(x, y)

    @staticmethod
    def tanh_deriv(x, y, dy, dx):
        dx[:] = dy * (1. - y * y)

    @staticmethod
    def rel(x, y):
        y[:] = x * (x > 0)

    @staticmethod
    def rel_deriv(x, y, dy, dx):
        dx[:] = dy * (x > 0)

