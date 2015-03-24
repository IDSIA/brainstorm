#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np
from pycuda import gpuarray, cumath
import pycuda.driver as drv
import pycuda.autoinit
from pycuda.elementwise import ElementwiseKernel
import scikits.cuda.linalg as culinalg
import scikits.cuda.misc as cumisc
culinalg.init()


class PyCudaHandler(object):
    def __init__(self):
        self.array_type = pycuda.gpuarray.GPUArray
        self.dtype = np.float32
        self.size = lambda x: x.size
        self.shape = lambda x: x.shape
        self.reshape = lambda x, s: x.reshape(s)
        self.slice = lambda x, s: x[s]
        self.context = cumisc._global_cublas_handle
        self.EMPTY = gpuarray.zeros((), dtype=self.dtype)

    def allocate(self, size):
        return gpuarray.zeros(size, dtype=self.dtype)

    @staticmethod
    def fill(mem, val):
        mem.fill(val)

    def set_from_numpy(self, mem, arr):
        assert mem.shape == arr.shape, "{} != {}".format(mem.shape, arr.shape)
        mem.set(arr.astype(self.dtype))

    def get_numpy_copy(self, mem):
        assert type(mem) == self.array_type
        return mem.get()

    @staticmethod
    def copy_to(dest, src):
        # Copy data from src to dest (both must be GPUArrays)
        drv.memcpy_dtod(dest.gpudata, src.gpudata, dest.nbytes)

    def zeros(self, shape):
        return gpuarray.zeros(shape=shape, dtype=self.dtype)

    # ---------------- Mathematical Operations ---------------- #

    @staticmethod
    def sum(a, axis, out):
        cumisc.sum(a, axis, out)

    @staticmethod
    def dot(a, b, out, transa='N', transb='N'):
        culinalg.dot(a, b, transa=transa, transb=transb, out=out)

    @staticmethod
    def dot_add(a, b, out, transa='N', transb='N'):
        culinalg.add_dot(a, b, out, transa, transb)

    @staticmethod
    def elem_mult(a, b, out):
        elem_mult_kernel(a, b, out)

    @staticmethod
    def elem_mult_st(a, b, out):
        elem_mult_st_kernel(a, b, out)

    @staticmethod
    def add_mm(a, b, out):
        add_mm_kernel(a, b, out)

    @staticmethod
    def subtract_mm(a, b, out):
        subtract_mm_kernel(a, b, out)

    @staticmethod
    def add_mv(a, b, out):
        cumisc.add_matvec(a, b, out=out)

    # Activation functions

    @staticmethod
    def sigmoid(x, y):
        sigmoid_kernel(x, y)

    @staticmethod
    def sigmoid_deriv(x, y, dy, dx):
        sigmoid_deriv_kernel(x, y, dy, dx)

    @staticmethod
    def tanh(x, y):
        cumath.tanh(x, out=y)

    @staticmethod
    def tanh_deriv(x, y, dy, dx):
        tanh_deriv_kernel(x, y, dy, dx)

    @staticmethod
    def rel(x, y):
        rel_kernel(x, y)

    @staticmethod
    def rel_deriv(x, y, dy, dx):
        rel_deriv_kernel(x, y, dy, dx)


elem_mult_kernel = ElementwiseKernel(
    b"float* x, float* y, float *out",
    b"out[i] = x[i] * y[i]",
    b"elem_mult_kernel"
)

elem_mult_st_kernel = ElementwiseKernel(
    b"float x, float* y, float *out",
    b"out[i] = x * y[i]",
    b"elem_mult_kernel"
)

add_mm_kernel = ElementwiseKernel(
    b"float* x, float* y, float *out",
    b"out[i] = x[i] + y[i]",
    b"add_mm_kernel"
)

subtract_mm_kernel = ElementwiseKernel(
    b"float* x, float* y, float *out",
    b"out[i] = x[i] - y[i]",
    b"subtract_mm_kernel"
)

sigmoid_kernel = ElementwiseKernel(
    b"float* x, float* y",
    b"y[i] = 1.0/(1.0 + exp(-1*x[i])",
    b"sigmoid_kernel"
)

sigmoid_deriv_kernel = ElementwiseKernel(
    b"float* x, float* y, float* dy, float* dx",
    b"dx[i] = dy[i] * y[i] * (1.0 - y[i])",
    b"sigmoid_deriv_kernel"
)

tanh_deriv_kernel = ElementwiseKernel(
    b"float* x, float* y, float* dy, float* dx",
    b"dx[i] = dy[i] * (1.0 - y[i] * y[i])",
    b"tanh_deriv_kernel"
)

rel_kernel = ElementwiseKernel(
    b"float* x, float* y",
    b"if (x[i]>0) y[i] = x[i]; else y[i]=0.0;",
    b"rel_kernel"
)

rel_deriv_kernel = ElementwiseKernel(
    b"float* x, float* y, float* dy, float* dx",
    b"if (x[i]>0) dx[i] = dy[i]; else dx[i]=0.0;",
    b"rel_deriv_kernel"
)
