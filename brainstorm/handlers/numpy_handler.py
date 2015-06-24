#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np
from brainstorm.handlers.base_handler import Handler


# noinspection PyMethodOverriding
class NumpyHandler(Handler):

    def __init__(self, dtype):
        self.array_type = np.ndarray
        self.dtype = dtype
        self.size = lambda x: x.size
        self.shape = lambda x: x.shape
        self.reshape = lambda x, s: x.reshape(s)
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

    @staticmethod
    def sum_t(a, axis, out):
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
    def mult_tt(a, b, out):
        np.multiply(a, b, out)

    @staticmethod
    def mult_add_tt(a, b, out):
        out[:] += a * b

    @staticmethod
    def mult_st(a, b, out):
        np.multiply(a, b, out)

    @staticmethod
    def add_tt(a, b, out):
        assert a.shape == b.shape == out.shape
        out[:] = a + b

    @staticmethod
    def add_st(s, t, out):
        out[:] = t + s

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

    @staticmethod
    def divide_mv(m, v, out):
        """
        Divide (M, N) matrix elementwise by a (1, N) vector using broadcasting.
        """
        out[:] = m / v

    @staticmethod
    def mult_mv(m, v, out):
        """
        Multiply (M, N) matrix elementwise by a (1, N) vector using
        broadcasting.
        """
        out[:] = m * v

    @staticmethod
    def binarize_v(v, out):
        out[:] = 0.
        for i in range(v.shape[0]):
            out[i, int(v[i])] = 1.0

    @staticmethod
    def index_m_by_v(m, v, out):
        for i in range(m.shape[0]):
            out[i] = m[i, int(v[i])]

    @staticmethod
    def get_im2col_map(num_input_maps, input_rows, input_cols,
                       kernel_size, stride):
        # im2col built upon http://stackoverflow.com/a/30110497
        # Returns a 2D map which performs im2col on a 3D array
        # Apply map to a 3D array using numpy.take(array, map)

        # Parameters
        col_extent = input_cols - kernel_size[1] + 1
        row_extent = input_rows - kernel_size[0] + 1

        # Get Starting block indices
        start_idx = np.arange(kernel_size[0])[:, None] * input_cols + \
            np.arange(kernel_size[1])

        # Get offsetted indices across the height and width of input array
        offset_idx = np.arange(row_extent)[:, None] * input_cols + \
            np.arange(col_extent)

        indices = start_idx.ravel()[:, None] + \
            offset_idx[::stride[0], ::stride[1]].ravel()
        adder = (np.arange(num_input_maps) * input_rows * input_cols)\
            .reshape((num_input_maps, 1, 1))

        # Extend to multiple input maps
        im2col_map = indices + adder

        # Reshape to stack input maps
        im2col_map = im2col_map.reshape((kernel_size[0] * kernel_size[1] *
                                         num_input_maps, -1))

        return im2col_map

    @classmethod
    def conv2d_forward_batch(cls, inputs, weights, bias, outputs, pad, stride):

        num_filters = weights.shape[0]
        num_images, num_input_maps, input_rows, input_cols = inputs.shape
        kernel_size = (weights.shape[2], weights.shape[3])

        im2col_map = cls.get_im2col_map(num_input_maps, input_rows + 2 * pad,
                                        input_cols + 2 * pad, kernel_size,
                                        stride)

        # reshape
        for i in range(inputs.shape[0]):
            # pad
            if pad == 0:
                im = inputs[i]
            else:
                im = np.zeros((inputs.shape[1], inputs.shape[2] + 2 * pad,
                               inputs.shape[3] + 2 * pad))
                im[:, pad: -pad, pad: -pad] = inputs[i]

            # Get all actual indices & index into input array for final output
            col = np.take(im, im2col_map)

            # multiply
            reshaped_weights = weights.reshape(num_filters, kernel_size[0] *
                                               kernel_size[1] * num_input_maps)
            outputs[i] = np.dot(reshaped_weights, col).reshape(outputs[i].shape)

        outputs += bias.reshape((1, num_filters, 1, 1))

    @staticmethod
    def conv2d_backward_batch(out_deltas, inputs, in_deltas, weights, bias,
                              weight_deltas, bias_deltas, pad, stride):
        pass

    # ---------------- Activation functions -----------------------------------

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

    @staticmethod
    def softmax_m(m, out):
        """Applies softmax to matrix over last dimension"""
        maxes = np.amax(m, axis=1, keepdims=True)
        e = np.exp(m - maxes)
        out[:] = e / np.sum(e, axis=1, keepdims=True)
