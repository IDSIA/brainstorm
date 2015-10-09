#!/usr/bin/env python
# coding=utf-8

from __future__ import division, print_function, unicode_literals

import numpy as np

from brainstorm.handlers import default_handler
from brainstorm.structure.buffer_structure import BufferStructure
from brainstorm.structure.buffer_views import BufferView
from brainstorm.utils import sort_by_index_key


def create_buffer_views_from_layout(layout, buffers, hubs, existing_view=None):
    if '@slice' in layout:
        buffer_nr = layout['@hub']
        feature_slice = slice(*layout['@slice'])
        structure = BufferStructure.from_layout(layout)
        full_buffer = structure.create_from_buffer_hub(
            buffers[buffer_nr], hubs[buffer_nr], feature_slice)
    else:
        full_buffer = None

    if layout['@type'] == 'BufferView':
        names, child_buffers = [], []
        for n, sub_node in sorted(layout.items(), key=sort_by_index_key):
            if n.startswith('@'):
                continue
            if existing_view:
                assert n in existing_view
                c = create_buffer_views_from_layout(
                    sub_node, buffers, hubs, existing_view=existing_view[n])
            else:
                c = create_buffer_views_from_layout(sub_node, buffers, hubs)
            names.append(n)
            child_buffers.append(c)

        if existing_view:
            return existing_view.adjust(names, child_buffers, full_buffer)
        else:
            return BufferView(names, child_buffers, full_buffer)
    else:  # layout['@type'] == 'array':
        assert full_buffer is not None, layout
        return full_buffer


def get_total_size_slices_and_shapes(hubs, time_size, batch_size):
        shapes = [h.get_shape(time_size, batch_size) for h in hubs]
        totals = np.cumsum([0] + [int(np.prod(s)) for s in shapes])
        size = int(totals[-1])
        slices = [slice(int(i), int(j))
                  for i, j in zip(totals[:-1], totals[1:])]
        return size, slices, shapes


class BufferManager(object):
    def __init__(self, layout, hubs, handler=default_handler):
        self.hubs = hubs
        self.handler = handler
        self.layout = layout
        self.time_size = -1
        self.batch_size = -1
        self.size = -1
        self.full_buffer = None
        self.buffers = []
        self.views = None
        self.resize(0, 0)

    def resize(self, time_size, batch_size):
        if time_size == self.time_size and batch_size == self.batch_size:
            return self.views  # lazy

        self.time_size = time_size
        self.batch_size = batch_size
        total_size, slices, shapes = get_total_size_slices_and_shapes(
            self.hubs, time_size, batch_size)

        if total_size > self.size:
            self.full_buffer = self.handler.allocate((total_size,))
            self.size = total_size

        self.buffers = [self.full_buffer[slices[i]].reshape(shapes[i])
                        for i in range(len(self.hubs))]

        parameters = None
        if self.views is not None:
            # copy the parameters
            parameters = self.handler.get_numpy_copy(self.views.parameters)

        self.views = create_buffer_views_from_layout(
            self.layout, self.buffers, self.hubs, existing_view=self.views)

        if parameters is not None:
            self.handler.set_from_numpy(self.views.parameters, parameters)

        return self.views

    def set_handler(self, new_handler):
        self.full_buffer = None
        self.size = -1
        self.time_size = -1
        self.batch_size = -1
        parameters = None
        if self.views is not None:
            parameters = self.handler.get_numpy_copy(self.views.parameters)
        self.views = None
        self.handler = new_handler
        self.resize(0, 0)
        if parameters is not None:
            self.handler.set_from_numpy(self.views.parameters, parameters)

    def get_context(self):
        if self.buffers is None:
            return None
        context = []
        for hub, buf in zip(self.hubs, self.buffers):
            if hub.btype != 2 or hub.context_size == 0:
                context.append(None)
            else:
                c = self.handler.zeros(
                    (hub.context_size, self.batch_size, hub.size))

                context_start_idx = self.time_size - hub.context_size
                context_stop_idx = self.time_size

                self.handler.copy_to(buf[context_start_idx:context_stop_idx],
                                     c)
                context.append(c)

        return context

    def apply_context(self, context):
        for c, buf in zip(context, self.buffers):
            if c is None:
                continue
            self.handler.copy_to(c, buf[self.time_size:])

    def clear_context(self):
        if self.buffers is None:
            return None
        for hub, buf in zip(self.hubs, self.buffers):
            if hub.btype != 2 or not hub.context_size:
                continue
            self.handler.fill(
                buf[self.time_size - hub.context_size:], 0.)

    def clear_backward_buffers(self):
        for h, b in zip(self.hubs, self.buffers):
            if h.is_backward_only:
                self.handler.fill(b, 0.)
