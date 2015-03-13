#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np


class DebugArray(object):
    def __init__(self, arr):
        assert arr is not None
        self.shape = arr.shape
        self._array = arr
        self.size = self._array.size

    def __getitem__(self, item):
        if isinstance(item, (int, slice)):
            item = tuple([item])
        assert isinstance(item, tuple)
        for i in item:
            assert isinstance(i, (int, slice))
            if isinstance(i, slice):
                assert i.step is None
        return DebugArray(arr=self._array.__getitem__(item))

    def reshape(self, new_shape):
        if isinstance(new_shape, (tuple, list)):
            assert all([t >= 0 for t in tuple(new_shape)])
        else:
            assert isinstance(new_shape, int)
            assert new_shape >= 0
        return DebugArray(arr=self._array.reshape(new_shape))


class DebugHandler(object):
    def __init__(self, handler):
        self.handler = handler
        self.EMPTY = DebugArray(arr=handler.EMPTY)

    def allocate(self, size):
        return DebugArray(self.handler.allocate(size))

    def set(self, a, b):
        assert isinstance(a, DebugArray)
        assert a.size == b.size
        self.handler.set_from_numpy(a._array, b)

    def dot(self, a, b):

        return DebugArray(self.handler.dot(a._array, b._array))