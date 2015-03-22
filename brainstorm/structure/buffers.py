#!/usr/bin/env python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
from copy import copy
import numpy as np
from brainstorm.handlers import default_handler
from brainstorm.structure.buffer_views import BufferView


def create_buffer_views_from_layout(layout, buffers):
    if 'slice' in layout:
        buffer_type, start, stop = layout['slice']
        shape = layout.get('shape', (stop - start,))
        full_buffer = buffers[buffer_type][start:stop].reshape(shape)
    else:
        full_buffer = None

    if 'layout' not in layout:
        assert full_buffer
        return full_buffer
    else:
        children = [(n, create_buffer_views_from_layout(sub_node, buffers))
                    for n, sub_node in sorted(layout['layout'].items(),
                                              key=lambda x: x[1]['index'])]
        names, child_buffers = zip(*children)
        return BufferView(names, child_buffers, full_buffer)


class BufferManager(object):
    def __init__(self, layout, sizes, handler=default_handler):
        self.feature_sizes = sizes
        self.handler = handler
        self.layout = layout
        self.time_size = -1
        self.batch_size = -1
        self.full_forward_buffers = None
        self.full_backward_buffers = None
        self.forward = None
        self.backward = None
        self.resize_memory(0, 0)

    def get_total_sizes(self):
        return (self.feature_sizes[0],
                self.feature_sizes[1] * self.batch_size,
                self.feature_sizes[2] * self.batch_size * self.time_size)

    def resize_memory(self, time_size, batch_size):
        self.time_size = time_size
        self.batch_size = batch_size
        total_sizes = self.get_total_sizes()
        memory = self.handler.allocate(sum(total_sizes))
        self.full_forward_buffers = [
            self.handler.zeros((self.feature_sizes[0],)),
            self.handler.zeros((self.batch_size, self.feature_sizes[1])),
            self.handler.zeros((self.time_size, self.batch_size, self.feature_sizes[2]))]
        self.full_backward_buffers = [
            self.handler.zeros((self.feature_sizes[0],)),
            self.handler.zeros((self.batch_size, self.feature_sizes[1])),
            self.handler.zeros((self.time_size, self.batch_size, self.feature_sizes[2]))]
        self.forward = create_buffer_views_from_layout(
            self.layout, self.full_forward_buffers)
        self.backward = create_buffer_views_from_layout(
            self.layout, self.full_backward_buffers)

    def reset(self):
        self.fwd_shape = None
        self.bwd_shape = None
        self.param_memory = None
        self.grad_memory = None
        self.fwd_memory = self.handler.EMPTY
        self.bwd_memory = self.handler.EMPTY

    def set_memory_handler(self, handler):
        # remember the parameters
        params = None
        if self.param_memory is not None:
            params = self.handler.get(self.param_memory)
        self.reset()
        # set all handlers
        self.handler = handler
        self.parameters.handler = handler
        self.gradient.handler = handler
        self.inputs.handler = handler
        self.outputs.handler = handler
        self.in_deltas.handler = handler
        self.out_deltas.handler = handler
        # restore the parameters
        if params is not None:
            self.handler.set_from_numpy(self.param_memory, params)
            self.parameters.rearrange(self.param_memory)

    def rearrange_parameters(self):
        if self.param_memory is None:
            self.param_memory = self.handler.allocate(self.parameters.size)
            self.parameters.rearrange(self.param_memory)

    def rearrange_fwd(self, shape):
        """
        Resize the buffers needed for a foward pass and prepare them.
        :param shape: Tuple specifying the dimensions. Only the first two are
            used. They should be (nr_timesteps, nr_sequences).
        :type shape: tuple[int]
        """
        if self.fwd_shape == shape[:2]:
            return
        self.fwd_shape = shape[:2]

        in_size = self.inputs.get_size(self.fwd_shape)

        if self.fwd_memory.size < in_size:
            self.fwd_memory = self.handler.allocate(in_size)
            self.inputs.rearrange(self.fwd_shape, self.fwd_memory)
            self.outputs.rearrange(self.fwd_shape, self.fwd_memory)
        else:
            self.inputs.rearrange(self.fwd_shape)
            self.outputs.rearrange(self.fwd_shape)

    def rearrange_bwd(self):
        """
        Resize the buffers needed for a backward pass and prepare them.
        Reuses the same shape as for the forward pass.
        """
        if self.bwd_shape == self.fwd_shape:
            return
        self.bwd_shape = self.fwd_shape

        if self.grad_memory is None:
            self.grad_memory = self.handler.allocate(self.gradient.size)
            self.gradient.rearrange(self.grad_memory)

        deltas_size = self.in_deltas.get_size(self.bwd_shape)

        if self.handler.size(self.bwd_memory) < deltas_size:
            self.bwd_memory = self.handler.allocate(deltas_size)

            self.in_deltas.rearrange(self.bwd_shape, self.bwd_memory)
            self.out_deltas.rearrange(self.bwd_shape, self.bwd_memory)
        else:
            self.in_deltas.rearrange(self.bwd_shape)
            self.out_deltas.rearrange(self.bwd_shape)

    @classmethod
    def create_from_layers(cls, layers):
        #param_layout = create_param_layout(layers)
        #param_buffer = ParameterBuffer(param_layout)

        #buffer_hub_layouts = create_in_out_layout(layers)
        #hub_sizes, source_hubs, sink_hubs = zip(*buffer_hub_layouts)
        #out_buffer = InOutBuffer(hub_sizes, source_hubs)
        #in_buffer = InOutBuffer(hub_sizes, sink_hubs)
        #return cls(param_buffer, in_buffer, out_buffer)
        pass
