#!/usr/bin/env python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import numpy as np
from brainstorm.handlers import default_handler
from brainstorm.structure.buffer_views import BufferView
from brainstorm.structure.layout import validate_shape_template
from brainstorm.utils import sort_by_index_key


def create_buffer_views_from_layout(layout, buffers, time_offset):
    if '@slice' in layout:
        start, stop = layout['@slice']
        shape = layout['@shape']
        t_start = time_offset - layout.get('@time_offset', 0)
        buffer_type = validate_shape_template(shape)
        if buffer_type == 0:
            full_buffer = buffers[buffer_type][start:stop]
            full_buffer = full_buffer.reshape(shape[buffer_type:])
        elif buffer_type == 1:
            full_buffer = buffers[buffer_type][:, start:stop]
            full_buffer = full_buffer.reshape(full_buffer.shape[:1] +
                                              shape[buffer_type:])
        else:  # buffer_type == 2
            full_buffer = buffers[buffer_type][t_start:, :, start:stop]
            full_buffer = full_buffer.reshape(
                (full_buffer.shape[0] - t_start,
                 full_buffer.shape[1]) +
                shape[buffer_type:])
    else:
        full_buffer = None

    if layout['@type'] == 'BufferView':
        children = [(n, create_buffer_views_from_layout(sub_node, buffers,
                                                        time_offset))
                    for n, sub_node in sorted(layout.items(),
                                              key=sort_by_index_key)
                    if not n.startswith('@')]
        if children:
            names, child_buffers = zip(*children)
        else:
            names, child_buffers = [], []
        return BufferView(names, child_buffers, full_buffer)
    else:  # layout['@type'] == 'array':
        assert full_buffer is not None, layout
        return full_buffer


class BufferManager(object):
    def __init__(self, layout, sizes, max_time_offset,
                 handler=default_handler):
        self.feature_sizes = sizes
        self.handler = handler
        self.layout = layout
        self.max_time_offset = max_time_offset
        self.time_size = -1
        self.batch_size = -1
        self.size = -1
        self.full_buffer = None
        self.forward = None
        self.backward = None
        self.resize(0, 0)
        self.forward_buffers = []
        self.backward_buffers = []

    def get_total_size_slices_and_shapes(self):
        shapes = [
            (self.feature_sizes[0],),
            (self.batch_size, self.feature_sizes[1]),
            (self.time_size + self.max_time_offset, self.batch_size,
             self.feature_sizes[2]),
        ]
        totals = np.cumsum([0] + [int(np.prod(s)) for s in shapes] * 2)
        size = int(totals[-1])
        slices = [slice(int(i), int(j)) for i, j in zip(totals[:-1],
                                                        totals[1:])]
        return size, slices, shapes

    def resize(self, time_size, batch_size):
        if time_size == self.time_size and batch_size == self.batch_size:
            return  # lazy

        self.time_size = time_size
        self.batch_size = batch_size
        total_size, slices, shapes = self.get_total_size_slices_and_shapes()

        if total_size > self.size:
            self.full_buffer = self.handler.allocate(total_size)
            self.size = total_size

        self.forward_buffers = [
            self.full_buffer[slices[0]].reshape(shapes[0]),
            self.full_buffer[slices[1]].reshape(shapes[1]),
            self.full_buffer[slices[2]].reshape(shapes[2])
        ]

        parameters = None
        if self.forward is not None:
            # copy the parameters
            parameters = self.handler.get_numpy_copy(self.forward.parameters)

        self.forward = create_buffer_views_from_layout(
            self.layout, self.forward_buffers, self.max_time_offset)

        if parameters is not None:
            self.handler.set_from_numpy(self.forward.parameters, parameters)

        # TODO optimization: allocate the backward pass only if needed
        self.backward_buffers = [
            self.full_buffer[slices[3]].reshape(shapes[0]),
            self.full_buffer[slices[4]].reshape(shapes[1]),
            self.full_buffer[slices[5]].reshape(shapes[2])
        ]

        self.backward = create_buffer_views_from_layout(
            self.layout, self.backward_buffers, self.max_time_offset)

    def set_memory_handler(self, new_handler):
        self.full_buffer = None
        self.size = -1
        self.time_size = -1
        self.batch_size = -1
        parameters = None
        if self.forward is not None:
            parameters = self.handler.get_numpy_copy(self.forward.parameters)
        self.forward = None
        self.handler = new_handler
        self.resize(0, 0)
        if parameters is not None:
            self.handler.set_from_numpy(self.forward.parameters, parameters)

    def get_context(self):
        if self.forward_buffers is None:
            return None
        timed_buffer = self.forward_buffers[2]
        t, b, f = timed_buffer.shape
        context = self.handler.zeros((self.max_time_offset, b, f))
        self.handler.copy_to(context, timed_buffer[t - self.max_time_offset:t])
        return context

    def apply_context(self, context):
        timed_buffer = self.forward_buffers[2]
        context_slice = timed_buffer[:self.max_time_offset]
        self.handler.copy_to(context_slice, context)

    def clear_context(self):
        timed_buffer = self.forward_buffers[2]
        context_slice = timed_buffer[:self.max_time_offset]
        self.handler.fill(context_slice, 0.)

    def clear_backward_buffers(self):
        for b in self.backward_buffers:
            self.handler.fill(b, 0.)
