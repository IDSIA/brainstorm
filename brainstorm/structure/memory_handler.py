#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np


class NumpyHandler(object):
    def __init__(self, dtype):
        self.dtype = dtype
        self.size = lambda x: x.size
        self.shape = lambda x: x.shape
        self.reshape = lambda x, s: x.reshape(s)
        self.slice = lambda x, s: x[s]
        self.get = lambda x: x
        self.context = 'numpy'
        self.empty = np.zeros(0)

    def allocate(self, size):
        return np.zeros(size, dtype=self.dtype)

    def fill(self, mem, val):
        mem.fill(val)

    def set(self, mem, arr):
        mem[:] = arr.astype(self.dtype)


try:
    from pycuda import gpuarray
    import scikits.cuda.linalg as culinalg
    import scikits.cuda.misc as cumisc
    culinalg.init()

    class PyCudaHandle(object):
        def __init__(self):
            self.dtype = np.float32
            self.context = cumisc._global_cublas_handle
            self.empty = gpuarray.zeros(0, dtype=self.dtype)

        @staticmethod
        def size(mem):
            return mem.size

        def allocate(self, size):
            return gpuarray.zeros(size, dtype=self.dtype)

        @staticmethod
        def fill(mem, val):
            mem.fill(val)

        def set(self, mem, arr):
            assert mem.shape == arr.shape
            mem.set(arr.astype(self.dtype))

        @staticmethod
        def get(mem):
            return mem.get()

        @staticmethod
        def shape(mem):
            return mem.shape

        @staticmethod
        def reshape(mem, shape):
            return mem.reshape(shape)

        @staticmethod
        def slice(mem, item):
            return mem[item]

except ImportError:
    pass


default_handler = NumpyHandler(np.float64)