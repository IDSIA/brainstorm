#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function
import numpy as np
from neon.backends.nervanagpu import NervanaGPU, GPUTensor
from pycuda import gpuarray
import pycuda
from pycuda.elementwise import ElementwiseKernel
from brainstorm.handlers.base_handler import Handler
from brainstorm.randomness import global_rnd

import skcuda.linalg as culinalg
import skcuda.misc as cumisc
culinalg.init()

class NervanaGPUHandler(Handler):
    __undescribed__ = {'context', 'dtype', 'EMPTY', 'rnd'}

    def __init__(self, seed=None):
        self.dtype = np.float32
        if seed is None:
            seed = global_rnd.generate_seed()
        self.context = NervanaGPU(rng_seed=seed, default_dtype=np.float32,
                                  stochastic_round=False)
        self.EMPTY = self.context.empty((), dtype=self.dtype)
        self.rnd = None

    array_type = GPUTensor

    def __init_from_description__(self, description):
        self.__init__()

    # ------------------------- Allocate new memory ------------------------- #

    def allocate(self, size):
        return self.context.empty(size, dtype=self.dtype)

    def ones(self, shape):
        return self.context.ones(shape=shape, dtype=self.dtype)

    def zeros(self, shape):
        return self.context.zeros(shape=shape, dtype=self.dtype)

    # ---------------------------- Copy and Fill ---------------------------- #

    def copy_to(self, dest, src):
        dest.fill(src)

    def copy_to_if(self, src, dest, cond):
        pass

    def create_from_numpy(self, arr):
        return self.context.array(arr, dtype=self.dtype)

    def fill(self, mem, val):
        mem.fill(val)

    def fill_if(self, mem, val, cond):
        pass

    def get_numpy_copy(self, mem):
        assert type(mem) == self.array_type
        return mem.get()

    def set_from_numpy(self, mem, arr):
        if len(arr.shape) == 1 and mem.shape == (arr.shape[0], 1):
            mem.set(arr.reshape((arr.shape[0], 1)).astype(self.dtype))
            return
        assert mem.shape == arr.shape, "Shape of destination ({}) != Shape " \
                                       "of source ({})".format(mem.shape,
                                                               arr.shape)
        mem.set(arr.astype(self.dtype))

    def as_gpuarray(self, mem):
        """Get a GPUArray reference to memory."""
        return gpuarray.GPUArray(shape=mem.shape, dtype=mem.dtype,
                                 gpudata=mem.gpudata)

    # ---------------------------- Debug helpers ---------------------------- #

    def is_fully_finite(self, a):
        return np.all(self.context.finite(a).get())

    # ----------------------- Mathematical operations ----------------------- #

    def abs_t(self, a, out):
        pass

    def add_into_if(self, a, out, cond):
        add_into_if_kernel(a, out, cond)

    def add_mv(self, m, v, out):
        out[:] = m + v

    def add_st(self, s, t, out):
        out[:] = t + s

    def add_tt(self, a, b, out):
        out[:] = a + b

    def avgpool2d_backward_batch(self, inputs, window, outputs, padding,
                                 stride, in_deltas, out_deltas):
        raise NotImplementedError

    def avgpool2d_forward_batch(self, inputs, window, outputs, padding,
                                stride):
        raise NotImplementedError

    def binarize_v(self, v, out):
        tmp = self.context.zeros((v.size, 1), dtype=np.int32)
        tmp[:] = v
        self.context.onehot(tmp, axis=1, out=out)

    def broadcast_t(self, a, out):
        assert len(a.shape) == 3
        assert a.shape[2] == 1
        assert len(out.shape) > 2
        a_flat = self.as_gpuarray(a.reshape(a.size))
        out_flat = self.as_gpuarray(out.reshape(out.size))
        broadcast_features_kernel(out_flat, a_flat, np.prod(out.shape[2:]))

    def clip_t(self, a, a_min, a_max, out):
        self.context.clip(a, a_min, a_max, out)

    def conv2d_backward_batch(self, inputs, weights, padding, stride,
                              in_deltas, out_deltas, weight_deltas,
                              bias_deltas):
        raise NotImplementedError

    def conv2d_forward_batch(self, inputs, weights, bias, outputs,
                             padding, stride):
        raise NotImplementedError

    def dot_add_mm(self, a, b, out, transa=False, transb=False):
        x = a.T if transa else a
        y = b.T if transb else b
        self.context.compound_dot(x, y, out, beta=1.0)

    def dot_mm(self, a, b, out, transa=False, transb=False):
        # x = a.T if transa else a
        # y = b.T if transb else b
        # self.context.compound_dot(x, y, out)
        transa = 'T' if transa else 'N'
        transb = 'T' if transb else 'N'
        culinalg.dot(self.as_gpuarray(a), self.as_gpuarray(b), transa=transa,
                     transb=transb,
                     out=self.as_gpuarray(out))

    def divide_mv(self, m, v, out):
        out[:] = m / v

    def divide_tt(self, a, b, out):
        out[:] = a / b

    def fill_gaussian(self, mean, std, out):
        raise NotImplementedError

    def generate_probability_mask(self, mask, probability):
        raise NotImplementedError

    def index_m_by_v(self, m, v, out):
        index_m_by_v_kernel(self.as_gpuarray(out),
                            self.as_gpuarray(v), self.as_gpuarray(m),
                            m.shape[0], m.shape[1])

    def log_t(self, a, out):
        self.context.log(a, out)

    def maxpool2d_backward_batch(self, inputs, window, outputs, padding,
                                 stride, argmax, in_deltas, out_deltas):
        raise NotImplementedError

    def maxpool2d_forward_batch(self, inputs, window, outputs, padding,
                                stride, argmax):
        raise NotImplementedError

    def merge_tt(self, a, b, out):
        pass

    def modulo_tt(self, a, b, out):
        pass

    def mult_add_st(self, a, b, out):
        out[:] += a * b

    def mult_add_tt(self, a, b, out):
        out[:] += a * b

    def mult_mv(self, m, v, out):
        out[:] = m * v

    def mult_st(self, a, b, out):
        self.context.multiply(a, b, out)

    def mult_tt(self, a, b, out):
        self.context.multiply(a, b, out)

    def sign_t(self, a, out):
        self.context.sgn(a, out)

    def split_add_tt(self, x, out_a, out_b):
        pass

    def sqrt_t(self, a, out):
        self.context.sqrt(a, out)

    def subtract_mv(self, m, v, out):
        out[:] = m - v

    def subtract_tt(self, a, b, out):
        out[:] = a - b

    def sum_t(self, a, axis, out):
        # assert len(a.shape) == 2
        # if axis is None:
        #     tmp = out.reshape((1, 1))
        #     self.context.sum(a.reshape((a.size, 1)), axis=0, out=tmp)
        # else:
        #     self.context.sum(a, axis=axis, out=out)
        a = self.as_gpuarray(a)
        out = self.as_gpuarray(out)
        if len(a.shape) < 3 and (axis == 0 or axis == 1):
            cumisc.sum(a, axis, out)
        elif axis is None:
            pycuda.driver.memcpy_dtod(out.gpudata, cumisc.sum(a).gpudata, out.nbytes)
        else:
            raise NotImplementedError

    def _pool2d_forward_batch(self, inputs, window, outputs, padding,
                              stride, argmax, pooling_mode):
        raise NotImplementedError

    def _pool2d_backward_batch(self, inputs, window, outputs, padding, stride,
                               argmax, in_deltas, out_deltas, pooling_mode):
        raise NotImplementedError

    # ------------------------ Activation functions ------------------------- #

    def rel(self, x, y):
        self.context.maximum(x, 0, y)

    def rel_deriv(self, x, y, dy, dx):
        dx[:] = dy * self.context.greater(y, 0)

    def sigmoid(self, x, y):
        self.context.sig(x, y)

    def sigmoid_deriv(self, x, y, dy, dx):
        dx[:] = dy * y * (1. - y)

    def softmax_m(self, m, out):
        out[:] = (self.context.reciprocal(self.context.sum(
                  self.context.exp(m - self.context.max(m, axis=0)), axis=0)) *
                  self.context.exp(m - self.context.max(m, axis=0)))

    def tanh(self, x, y):
        self.context.tanh(x, y)

    def tanh_deriv(self, x, y, dy, dx):
        dx[:] = dy * (1. - y * y)

# -------------------------------- Kernels ---------------------------------- #

broadcast_features_kernel = ElementwiseKernel(
    "float* out, float* a, unsigned int broadcast_size",
    "out[i] = a[i / broadcast_size]",
    "bc_features_kernel"
)

index_m_by_v_kernel = ElementwiseKernel(
    "float* out, float* v, float* m, int nrows, int ncols",
    "out[i] = m[i * ncols + int(v[i])]",
    "index_m_by_v_kernel"
)
