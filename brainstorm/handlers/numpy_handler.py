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

    @staticmethod
    def fill(self, mem, val):
        mem.fill(val)

    def set(self, mem, arr):
        mem[:] = arr.astype(self.dtype)