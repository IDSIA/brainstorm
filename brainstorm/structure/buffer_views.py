#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals


class BufferView(tuple):
    def __new__(cls, name, buffer_names, buffers, timed, full_buffer=None):
        instance = tuple.__new__(cls, buffers)
        return instance

    def __init__(self, name, buffer_names, buffers, timed, full_buffer=None):
        super(BufferView, self).__init__()
        if not len(buffers) == len(buffer_names):
            raise ValueError("Length mismatch between buffers and names ({} !="
                             " {})".format(len(buffers), len(buffer_names)))
        self._name = name
        self._full_buffer = full_buffer
        self._buffer_names = tuple(buffer_names)
        self._timed = timed
        for i, name in enumerate(buffer_names):
            self.__dict__[name] = self[i]

    def _asdict(self):
        return dict(zip(self._buffer_names, self))

    def items(self):
        return self._asdict().items()

    def keys(self):
        return self._asdict().keys()

    def values(self):
        return self._asdict().values()

    def get_time_slice(self, item):
        new_buffers = []
        for own_buffer, timed in zip(self, self._timed):
            new_buffers.append(own_buffer[item] if timed else own_buffer)
        return BufferView(self._name, self._buffer_names, new_buffers,
                          self._timed, self._full_buffer)

    def set_time_slice(self, item, bview):
        for own_buffer, other_buffer, timed in zip(self, bview, self._timed):
            if timed:
                own_buffer[item] = other_buffer
            else:
                own_buffer[:] = other_buffer

    def __getitem__(self, item):
        if isinstance(item, int):
            return super(BufferView, self).__getitem__(item)
        return self.__dict__[item]

    def __unicode__(self):
        return "<{}>".format(self._name)
