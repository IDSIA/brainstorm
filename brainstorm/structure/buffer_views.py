#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals


class BufferView(tuple):
    def __new__(cls, buffer_names, buffers, full_buffer=None):
        instance = tuple.__new__(cls, buffers)
        return instance

    def __init__(self, buffer_names, buffers, full_buffer=None):
        super(BufferView, self).__init__()
        if not len(buffers) == len(buffer_names):
            raise ValueError("Length mismatch between buffers and names ({} !="
                             " {})".format(len(buffers), len(buffer_names)))
        self._full_buffer = full_buffer
        self._buffer_names = tuple(buffer_names)
        for i, n in enumerate(buffer_names):
            self.__dict__[n] = self[i]

    def _asdict(self):
        return dict(zip(self._buffer_names, self))

    def items(self):
        return self._asdict().items()

    def keys(self):
        return self._asdict().keys()

    def values(self):
        return self._asdict().values()

    def __getitem__(self, item):
        if isinstance(item, int):
            return super(BufferView, self).__getitem__(item)
        return self.__dict__[item]

    def __contains__(self, item):
        return item in self._buffer_names
