#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

import numpy as np

import brainstorm.handlers._cpuop
from brainstorm.handlers.base_handler import Handler
from brainstorm.randomness import global_rnd


# noinspection PyMethodMayBeStatic
class NumpyHandler(Handler):
    __undescribed__ = {'context', 'EMPTY', 'rnd'}

    def __init__(self, dtype, seed=None):
        super(NumpyHandler, self).__init__()
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

    def copy_to(self, src, dest):
        # FIXME: change casting to 'no'
        # Can't be done right now, because it will lead to problems with
        # provide_external_data trying to call this function on ndarrays with
        # the wrong dtype.
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

    def add_into_if(self, a, out, cond):
        out[cond != 0] += a[cond != 0]

    def add_mv(self, m, v, out):
        out[:] = m + v

    def add_st(self, s, t, out):
        out[:] = t + s

    def add_tt(self, a, b, out):
        out[:] = a + b

    def avgpool2d_backward_batch(self, inputs, window, outputs, padding,
                                 stride, in_deltas, out_deltas):
        brainstorm.handlers._cpuop.avgpool_backward(
            inputs, window, outputs, padding, stride, in_deltas, out_deltas)

    def avgpool2d_forward_batch(self, inputs, window, outputs, padding,
                                stride):
        brainstorm.handlers._cpuop.avgpool_forward(inputs, window, outputs,
                                                   padding, stride)

    def binarize_v(self, v, out):
        eye = np.eye(out.shape[1], dtype=self.dtype)
        out[:] = eye[v.astype(np.int32)].reshape(out.shape)

    def broadcast_t(self, a, axis, out):
        assert (out.shape[:axis] + (1,) + out.shape[axis+1:]) == a.shape
        out[:] = a[:]  # automatically does the right thing via broadcasting

    def clip_t(self, a, a_min, a_max, out):
        np.clip(a, a_min, a_max, out)

    def conv2d_backward_batch(self, inputs, params, padding, stride,
                              in_deltas, out_deltas, dparams, dbias):
        num_filters = params.shape[0]
        num_images, input_rows, input_cols, num_input_maps = inputs.shape
        kernel_shape = params.shape[1:]
        num_output_pixels = out_deltas.shape[1] * out_deltas.shape[2]
        num_kernel_params = np.prod(kernel_shape)

        dparams.fill(0.0)
        dbias.fill(0.0)
        col = np.zeros((num_output_pixels, num_kernel_params),
                       dtype=self.dtype)

        for i in range(num_images):
            brainstorm.handlers._cpuop.im2col(
                inputs[i].reshape(inputs[i].size),
                input_rows, input_cols, num_input_maps,
                kernel_shape[0], kernel_shape[1],
                padding, padding, padding, padding,
                stride[0], stride[1], col.reshape(col.size))

            # Compute gradients
            reshaped_dparams = dparams.reshape(num_filters, num_kernel_params)
            reshaped_out_deltas = out_deltas[i].reshape((num_output_pixels,
                                                         num_filters))
            self.dot_add_mm(reshaped_out_deltas, col, out=reshaped_dparams,
                            transa=True)
            dbias += np.sum(reshaped_out_deltas, axis=0)

            # Compute in_deltas
            reshaped_params = params.reshape((num_filters, num_kernel_params))
            np.dot(reshaped_out_deltas, reshaped_params, out=col)
            brainstorm.handlers._cpuop.col2im(
                col.reshape(col.size),
                input_rows, input_cols, num_input_maps,
                kernel_shape[0], kernel_shape[1],
                padding, padding, padding, padding,
                stride[0], stride[1], in_deltas[i].reshape(in_deltas[i].size))

    def conv2d_forward_batch(self, inputs, weights, bias, outputs,
                             padding, stride):
        num_filters = weights.shape[0]
        num_images, input_rows, input_cols, num_input_maps = inputs.shape
        kernel_shape = weights.shape[1:]
        num_output_pixels = outputs.shape[1] * outputs.shape[2]
        num_kernel_params = np.prod(kernel_shape)
        out_shape = (num_output_pixels, num_filters)

        for i in range(num_images):
            col = np.zeros((num_output_pixels, num_kernel_params),
                           dtype=self.dtype)
            brainstorm.handlers._cpuop.im2col(
                inputs[i].reshape(inputs[i].size),
                input_rows, input_cols, num_input_maps,
                kernel_shape[0], kernel_shape[1],
                padding, padding, padding, padding, stride[0],
                stride[1], col.reshape(col.size))

            reshaped_params = weights.reshape(num_filters, num_kernel_params)
            np.dot(col, reshaped_params.T, out=outputs[i].reshape(out_shape))

        outputs += bias.reshape((1, 1, 1, num_filters))

    def copy_to_if(self, src, dest, cond):
        dest[cond != 0] = src[cond != 0]

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

    def fill_if(self, mem, val, cond):
        mem[cond != 0] = val

    def generate_probability_mask(self, mask, probability):
        mask[:] = self.rnd.uniform(size=mask.shape) < probability

    def index_m_by_v(self, m, v, out):
        out[:, 0] = m[np.arange(m.shape[0]), v.squeeze().astype(np.int32)]
        # for i in range(m.shape[0]):
        #     out[i] = m[i, int(v[i])]

    def log_t(self, a, out):
        np.log(a, out)

    def maxpool2d_backward_batch(self, inputs, window, outputs, padding,
                                 stride, argmax, in_deltas, out_deltas):
        brainstorm.handlers._cpuop.maxpool_backward(inputs, window, outputs,
                                                    padding, stride, argmax,
                                                    in_deltas, out_deltas)

    def maxpool2d_forward_batch(self, inputs, window, outputs, padding,
                                stride, argmax):
        brainstorm.handlers._cpuop.maxpool_forward(inputs, window, outputs,
                                                   padding, stride, argmax)

    def merge_tt(self, a, b, out):
        out_flat = out.reshape(-1, out.shape[-1])
        sa = a.shape[-1]
        out_flat[:, :sa] = a.reshape(-1, a.shape[-1])
        out_flat[:, sa:] = b.reshape(-1, b.shape[-1])

    def modulo_tt(self, a, b, out):
        np.fmod(a, b, out)

    def mult_add_st(self, s, t, out):
        out[:] += s * t

    def mult_add_tt(self, a, b, out):
        out[:] += a * b

    def mult_mv(self, m, v, out):
        out[:] = m * v

    def mult_add_mv(self, m, v, out):
        out[:] += m * v

    def mult_st(self, s, t, out):
        np.multiply(s, t, out)

    def mult_tt(self, a, b, out):
        np.multiply(a, b, out)

    def sign_t(self, a, out):
        np.sign(a, out=out)

    def split_add_tt(self, x, out_a, out_b):
        oa = out_a.reshape(-1, out_a.shape[-1])
        ob = out_b.reshape(-1, out_b.shape[-1])
        x_flat = x.reshape(-1, x.shape[-1])
        oa[:] += x_flat[:, :oa.shape[-1]]
        ob[:] += x_flat[:, oa.shape[-1]:]

    def sqrt_t(self, a, out):
        np.sqrt(a, out)

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

    def guided_rel_deriv(self, x, y, dy, dx):
        dx[:] = dy * (y > 0) * (dy > 0)

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

    def el(self, x, y):
        # TODO: Optimize this implementation
        y[:] = x * (x >= 0.) + (np.exp(x) - 1.) * (x < 0.)

    def el_deriv(self, x, y, dy, dx):
        # TODO: Optimize this implementation
        dx[:] = dy * (y >= 0.) + dy * (y + 1.) * (y < 0.)

    def softplus(self, x, y):
        y[:] = np.log(1 + np.exp(x))

    def softplus_deriv(self, x, y, dy, dx):
        dx[:] = dy * (1 - np.exp(y))
