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
        self.EMPTY = np.zeros(0)

        ############# Mathematical Operations ################
        # TODO: Make a list of allowed mathematical operations

        # Add 2 matrices of same shape or a matrix and a scalar
        self.add = lambda x, y: x + y

        # Multiply 2 matrices, or inner product of (row vector, column vector) or (column vector, row vector)
        self.dot = lambda x, y: np.dot(x, y)

        # Elementwise multiply 2 matrices, or a matrix with a scalar
        self.elem_mult = lambda x, y: x * y

        # Sum elements along a given axis
        self.sum = np.sum

        # Activation functions
        self.sigmoid = lambda x: 1. / (1. + np.exp(-x))
        self.sigmoid_deriv = lambda y: y * (1 - y)
        self.tanh = np.tanh
        self.tanh_deriv = lambda y: 1 - y * y

    def allocate(self, size):
        return np.zeros(size, dtype=self.dtype)

    @staticmethod
    def fill(mem, val):
        mem.fill(val)

    def set(self, mem, arr):
        mem[:] = arr.astype(self.dtype)