#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np


class DebugArray(object):
    def __init__(self, shape=None, arr=None):
        if shape is None:
            assert arr is not None
            self.shape = arr.shape
            self._array = arr
        else:
            assert shape is not None
            self.shape = shape
            self._array = np.zeros(shape, dtype=np.float32)

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
    def __init__(self):
        self.EMPTY = DebugArray(arr=np.zeros(0, np.float32))

    @staticmethod
    def allocate(size):
        return DebugArray(size)

    def set(self, a, b):
        assert isinstance(a, DebugArray)
        assert a.size == b.size
        a._array[:] = b
