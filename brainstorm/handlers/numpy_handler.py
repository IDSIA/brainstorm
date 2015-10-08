#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

import numpy as np

from brainstorm.handlers import _cpuop
from brainstorm.handlers.base_handler import Handler
from brainstorm.randomness import global_rnd


# noinspection PyMethodMayBeStatic
class NumpyHandler(Handler):
    __undescribed__ = {'context', 'EMPTY', 'rnd'}

    def __init__(self, dtype, seed=None):
        self.dtype = dtype
        self.context = 'numpy'
        self.EMPTY = np.zeros(0)
        self.rnd = global_rnd.create_random_state(seed)

    array_type = np.ndarray

    def __describe__(self):
        return {
            '@type': self.__class__.__name__,
            'dtype': str(np.dtype(self.dtype))
        }

    def __init_from_description__(self, description):
        self.__init__(np.dtype(description['dtype']))

    # ------------------------- Allocate new memory ------------------------- #

    def allocate(self, size):
        return np.zeros(size, dtype=self.dtype)

    def ones(self, shape):
        return np.ones(shape=shape, dtype=self.dtype)

    def zeros(self, shape):
        return np.zeros(shape=shape, dtype=self.dtype)

    # ---------------------------- Copy and Fill ---------------------------- #

    def copy_to(self, dest, src):
        # FIXME: change casting to 'no'
        np.copyto(dest, src, casting='same_kind')

    def create_from_numpy(self, arr):
        return arr.copy()

    def fill(self, mem, val):
        mem.fill(val)

    def get_numpy_copy(self, arr):
        assert type(arr) == self.array_type
        return arr.copy()

    def set_from_numpy(self, mem, arr):
        mem[:] = arr.astype(self.dtype)

    # ---------------------------- Debug helpers ---------------------------- #

    def is_fully_finite(self, a):
        return np.all(np.isfinite(a))

    # ----------------------- Mathematical operations ----------------------- #

    def abs_t(self, a, out):
        np.abs(a, out=out)

    def add_mv(self, m, v, out):
        out[:] = m + v

    def add_st(self, s, t, out):
        out[:] = t + s

    def add_tt(self, a, b, out):
        assert a.shape == b.shape == out.shape
        out[:] = a + b

    def avgpool2d_backward_batch(self, inputs, window, outputs, padding,
                                 stride, in_deltas, out_deltas):
        _cpuop.avgpool_backward(inputs, window, outputs, padding, stride,
                                in_deltas, out_deltas)

    def avgpool2d_forward_batch(self, inputs, window, outputs, padding,
                                stride):
        _cpuop.avgpool_forward(inputs, window, outputs, padding, stride)

    def binarize_v(self, v, out):
        out[:] = 0.
        for i in range(v.shape[0]):
            out[i, int(v[i])] = 1.0

    def broadcast_features_t(self, a, out):
        num_extra_dims = len(out.shape) - 3
        shape_to_add = tuple([1] * num_extra_dims)
        b = np.reshape(a, a.shape + shape_to_add)
        shape_to_tile = (1, 1) + out.shape[2:]
        out[:] = np.tile(b, shape_to_tile)

    def clip_t(self, a, a_min, a_max, out):
        np.clip(a, a_min, a_max, out)

    def conv2d_backward_batch(self, inputs, weights, padding, stride,
                              in_deltas, out_deltas, weight_deltas,
                              bias_deltas):
        if stride != (1, 1):
            raise NotImplementedError("Strides > 1 for ConvolutionLayer2D are "
                                      "not supported yet.")
        num_filters = weights.shape[0]
        num_images, num_input_maps, input_rows, input_cols = inputs.shape
        _, num_output_maps, output_rows, output_cols = out_deltas.shape
        kernel_size = (weights.shape[2], weights.shape[3])

        im2col_map = self.get_im2col_map(num_input_maps,
                                         input_rows + 2 * padding,
                                         input_cols + 2 * padding,
                                         kernel_size, stride)

        dpadh = ((input_rows + 2 * padding - 1) * stride[0] + kernel_size[0] -
                 output_rows) // 2
        dpadw = ((input_cols + 2 * padding - 1) * stride[1] + kernel_size[1] -
                 output_cols) // 2
        col2im_map = self.get_im2col_map(num_output_maps,
                                         output_rows + 2 * dpadh,
                                         output_cols + 2 * dpadw,
                                         kernel_size, (1, 1))
        weight_deltas.fill(0.0)
        bias_deltas.fill(0.0)
        for i in range(num_images):
            # pad
            if padding == 0:
                im = inputs[i]
            else:
                im = np.zeros((num_input_maps, input_rows + 2 * padding,
                               input_cols + 2 * padding))
                im[:, padding: -padding, padding: -padding] = inputs[i]

            # Get all actual indices & index into input array for final output
            col = np.take(im, im2col_map)

            # Compute gradients
            reshaped_dweights = weight_deltas.reshape(num_filters,
                                                      kernel_size[0] *
                                                      kernel_size[1] *
                                                      num_input_maps)
            reshaped_out_deltas = out_deltas[i].reshape((num_filters, -1))
            self.dot_add_mm(reshaped_out_deltas, col, out=reshaped_dweights,
                            transb=True)
            bias_deltas += np.sum(reshaped_out_deltas, axis=1)

            # Compute in_deltas

            # But first some reshaping magic to rotate all kernels twice by 90
            prod_k = kernel_size[0] * kernel_size[1]
            _weights = np.fliplr(weights.reshape(-1, prod_k)).reshape(
                weights.shape)
            reshaped_weights = _weights.swapaxes(0, 1).reshape(num_input_maps,
                                                               prod_k *
                                                               num_filters)

            im = np.zeros((num_filters, output_rows + 2 * dpadh,
                           output_cols + 2 * dpadw))
            im[:, dpadh: -dpadh, dpadw: -dpadw] = out_deltas[i]

            col = np.take(im, col2im_map)

            # temp contains deltas WRT padded inputs
            new_shape = (num_input_maps,
                         input_rows + 2 * padding,
                         input_cols + 2 * padding)
            temp = np.dot(reshaped_weights, col).reshape(new_shape)
            # Remove padding
            if padding == 0:
                in_deltas[i] += temp
            else:
                in_deltas[i] += temp[:, padding: -padding, padding: -padding]

    def conv2d_forward_batch(self, inputs, weights, bias, outputs,
                             padding, stride):

        num_filters = weights.shape[0]
        num_images, num_input_maps, input_rows, input_cols = inputs.shape
        kernel_size = (weights.shape[2], weights.shape[3])
        out_shape = outputs.shape[1:]

        im2col_map = self.get_im2col_map(num_input_maps,
                                         input_rows + 2 * padding,
                                         input_cols + 2 * padding,
                                         kernel_size, stride)

        # reshape
        for i in range(num_images):
            # pad
            if padding == 0:
                im = inputs[i]
            else:
                im = np.zeros((inputs.shape[1], inputs.shape[2] + 2 * padding,
                               inputs.shape[3] + 2 * padding))
                im[:, padding: -padding, padding: -padding] = inputs[i]

            # Get all actual indices & index into input array for output
            col = np.take(im, im2col_map)

            # multiply
            reshaped_weights = weights.reshape(num_filters,
                                               kernel_size[0] *
                                               kernel_size[1] *
                                               num_input_maps)
            outputs[i] = np.dot(reshaped_weights, col).reshape(out_shape)

        outputs += bias.reshape((1, num_filters, 1, 1))

    def dot_add_mm(self, a, b, out, transa=False, transb=False):
        x = a.T if transa else a
        y = b.T if transb else b
        out[:] += np.dot(x, y)

    def dot_mm(self, a, b, out, transa=False, transb=False):
        x = a.T if transa else a
        y = b.T if transb else b
        # np.dot(x, y, out)  # FIXME: doesn't work with strided out
        out[:] = np.dot(x, y)

    def divide_mv(self, m, v, out):
        out[:] = m / v

    def divide_tt(self, a, b, out):
        out[:] = a / b

    def fill_gaussian(self, mean, std, out):
        out[:] = std * self.rnd.standard_normal(out.shape) + mean

    def generate_probability_mask(self, mask, probability):
        mask[:] = self.rnd.uniform(size=mask.shape) < probability

    def get_im2col_map(self, num_input_maps, input_rows, input_cols,
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

        # Get offset-ed indices across the height and width of input array
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

    def index_m_by_v(self, m, v, out):
        for i in range(m.shape[0]):
            out[i] = m[i, int(v[i])]

    def log_t(self, a, out):
        np.log(a, out)

    def maxpool2d_backward_batch(self, inputs, window, outputs, padding,
                                 stride, argmax, in_deltas, out_deltas):
        _cpuop.maxpool_backward(inputs, window, outputs, padding, stride,
                                argmax, in_deltas, out_deltas)

    def maxpool2d_forward_batch(self, inputs, window, outputs, padding,
                                stride, argmax):
        _cpuop.maxpool_forward(inputs, window, outputs, padding,
                               stride, argmax)

    def mult_add_st(self, s, t, out):
        out[:] += s * t

    def mult_add_tt(self, a, b, out):
        out[:] += a * b

    def mult_mv(self, m, v, out):
        out[:] = m * v

    def mult_st(self, s, t, out):
        np.multiply(s, t, out)

    def mult_tt(self, a, b, out):
        np.multiply(a, b, out)

    def sign_t(self, a, out):
        np.sign(a, out=out)

    def sqrt_t(self, a, out):
        np.sqrt(a, out)
    def sigmoid(self, x, y):
        indices = x >= 0
        y[indices] = 1. / (1. + np.exp(-x[indices]))
        indices = x < 0
        y[indices] = np.exp(x[indices]) / (1. + np.exp(x[indices]))

    def subtract_mv(self, m, v, out):
        out[:] = m - v

    def subtract_tt(self, a, b, out):
        assert a.shape == b.shape == out.shape
        out[:] = a - b

    def sum_t(self, a, axis, out):
        if axis is not None and len(out.shape) == len(a.shape):
            keepdims = True
        else:
            keepdims = False
        np.sum(a, axis=axis, out=out, keepdims=keepdims)

    # ------------------------ Activation functions ------------------------- #

    def rel(self, x, y):
        y[:] = x * (x > 0)

    def rel_deriv(self, x, y, dy, dx):
        dx[:] = dy * (y > 0)

    def sigmoid(self, x, y):
        indices = x >= 0
        y[indices] = 1. / (1. + np.exp(-x[indices]))
        indices = x < 0
        y[indices] = np.exp(x[indices]) / (1. + np.exp(x[indices]))

    def sigmoid_deriv(self, x, y, dy, dx):
        dx[:] = dy * y * (1. - y)

    def softmax_m(self, m, out):
        maxes = np.amax(m, axis=1, keepdims=True)
        e = np.exp(m - maxes)
        out[:] = e / np.sum(e, axis=1, keepdims=True)

    def tanh(self, x, y):
        np.tanh(x, y)

    def tanh_deriv(self, x, y, dy, dx):
        dx[:] = dy * (1. - y * y)
