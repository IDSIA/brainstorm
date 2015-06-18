#!/usr/bin/env python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
from itertools import repeat
import numpy as np
from brainstorm.handlers import default_handler
from brainstorm.structure.buffer_views import BufferView
from brainstorm.structure.layout import validate_shape_template
from brainstorm.utils import sort_by_index_key


def create_buffer_views_from_layout(layout, buffers, total_context_size):
    if '@slice' in layout:
        buffer_nr = layout['@hub']
        start, stop = layout['@slice']
        shape = layout['@shape']

        cutoff = total_context_size - layout.get('@context_size', 0)
        t_slice = slice(0, -cutoff if cutoff else None)
        buffer_type = validate_shape_template(shape)
        if buffer_type == 0:
            full_buffer = buffers[buffer_nr][start:stop]
            full_buffer = full_buffer.reshape(shape[buffer_type:])
        elif buffer_type == 1:
            full_buffer = buffers[buffer_nr][:, start:stop]
            full_buffer = full_buffer.reshape(full_buffer.shape[:1] +
                                              shape[buffer_type:])
        else:  # buffer_type == 2
            full_buffer = buffers[buffer_nr][t_slice, :, start:stop]
            full_buffer = full_buffer.reshape(
                (full_buffer.shape[0] - cutoff,
                 full_buffer.shape[1]) +
                shape[buffer_type:])
    else:
        full_buffer = None

    if layout['@type'] == 'BufferView':
        children = [(n, create_buffer_views_from_layout(sub_node, buffers,
                                                        total_context_size))
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
    def __init__(self, layout, sizes, max_context_size,
                 handler=default_handler):
        self.hubs = list(zip(sizes[0], repeat(0))) + \
                    list(zip(sizes[1], repeat(1))) + \
                    list(zip(sizes[2], repeat(2)))

        self.feature_sizes = [sum(s) for s in sizes]
        self.handler = handler
        self.layout = layout
        self.max_context_size = max_context_size
        self.time_size = -1
        self.batch_size = -1
        self.size = -1
        self.full_buffer = None
        self.forward = None
        self.backward = None
        self.resize(0, 0)
        self.forward_buffers = []
        self.backward_buffers = []

    def get_hub_shape(self, size, btype):
        return (self.time_size, self.batch_size, size)[2 - btype:]

    def get_total_size_slices_and_shapes(self):
        shapes = [self.get_hub_shape(s, t) for s, t in self.hubs] * 2
        totals = np.cumsum([0] + [int(np.prod(s)) for s in shapes])
        size = int(totals[-1])
        slices = [slice(int(i), int(j)) for i, j in zip(totals[:-1],
                                                        totals[1:])]
        return size, slices, shapes

    def resize(self, time_size, batch_size):
        if time_size == self.time_size and batch_size == self.batch_size:
            return  # lazy

        N = len(self.hubs)

        self.time_size = time_size
        self.batch_size = batch_size
        total_size, slices, shapes = self.get_total_size_slices_and_shapes()

        if total_size > self.size:
            self.full_buffer = self.handler.allocate(total_size)
            self.size = total_size

        self.forward_buffers = [self.full_buffer[slices[i]].reshape(shapes[i])
                                for i in range(N)]

        parameters = None
        if self.forward is not None:
            # copy the parameters
            parameters = self.handler.get_numpy_copy(self.forward.parameters)

        self.forward = create_buffer_views_from_layout(
            self.layout, self.forward_buffers, self.max_context_size)

        if parameters is not None:
            self.handler.set_from_numpy(self.forward.parameters, parameters)

        # TODO optimization: allocate the backward pass only if needed
        self.backward_buffers = [self.full_buffer[slices[i]].reshape(shapes[i])
                                 for i in range(N, 2 * N)]

        self.backward = create_buffer_views_from_layout(
            self.layout, self.backward_buffers, self.max_context_size)

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

        c_start_idx = self.time_size - self.max_context_size
        context = []
        for (s, t), buf in zip(self.hubs, self.forward_buffers):
            if t != 2:
                context.append(None)
            else:
                c = self.handler.zeros(
                    (self.max_context_size, self.batch_size, s))
                self.handler.copy_to(
                    c, buf[c_start_idx - self.max_context_size:c_start_idx])
                context.append(c)

        return context

    def apply_context(self, context):
        c_start_idx = self.time_size - self.max_context_size
        for c, buf in zip(context, self.forward_buffers):
            if c is None:
                continue
            self.handler.copy_to(buf[c_start_idx:], c)

    def clear_context(self):
        if self.forward_buffers is None:
            return None
        c_start_idx = self.time_size - self.max_context_size
        for (s, t), buf in zip(self.hubs, self.forward_buffers):
            if t != 2:
                continue
            self.handler.fill(
                buf[c_start_idx - self.max_context_size:c_start_idx], 0.)

    def clear_backward_buffers(self):
        for b in self.backward_buffers:
            self.handler.fill(b, 0.)
