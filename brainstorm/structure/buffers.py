#!/usr/bin/env python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
from copy import copy
from brainstorm.structure.layout import (create_param_layout,
                                         create_in_out_layout)
from brainstorm.structure.memory_handler import default_handler


class ParameterBuffer(dict):
    """
    Handles the parameters of the network.
    The buffer is allocated at initialization, and the views for all the
    layers are created.
    """
    def __init__(self, param_layout, view_factories, handler=default_handler):
        super(ParameterBuffer, self).__init__()
        self.size, self.layout = param_layout
        self.view_factories = view_factories
        self.memory = None
        self.handler = handler

    def rearrange(self, memory):
        relocated = self._relocate_internal_memory(memory)
        if relocated:
            self._lay_out()

    def _relocate_internal_memory(self, memory):
        assert memory is not None, "No memory given to ParameterBuffer"

        if memory is self.memory:
            return False

        mem_size = self.handler.size(memory)
        assert mem_size == self.size, \
            "Given memory is wrong size: {} != {}".format(mem_size, self.size)
        self.memory = memory
        return True

    def _lay_out(self):
        for layer_name in self.layout:
            view = self.view_factories[layer_name](self.get_raw(layer_name))
            self[layer_name] = view

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self.handler.slice(self.memory, item)
        else:
            return dict.__getitem__(self, item)

    def get_raw(self, layer_name=None):
        """
        Get the part of the memory that corresponds to the given layer, or the
        the whole buffer if none is specified.
        """
        if layer_name is None:
            return self.memory
        else:
            return self.handler.slice(self.memory, self.layout[layer_name])


class InOutBuffer(dict):
    """
    Handles input or output buffers. The memory is allocated on demand.
    There should always be one of this object for the inputs and one for the
    outputs with corresponding layouts that share the same memory region.
    """
    def __init__(self, hub_sizes, layouts, handler=default_handler):
        super(InOutBuffer, self).__init__()
        self.hub_sizes = hub_sizes
        self.size = 0
        self.layouts = layouts
        self.memory = None
        self.shape = None
        self.handler = handler

    def rearrange(self, shape, memory=None):
        shape_changed = self.shape != shape[:2]
        self.shape = shape[:2]
        self.size = self.get_size(self.shape)
        relocated = self._resize_internal_memory(memory)
        if relocated or shape_changed:
            self._lay_out()

    def get_size(self, shape):
        nr_timesteps, nr_sequences = shape[:2]
        return nr_timesteps * nr_sequences * sum(self.hub_sizes)

    def _resize_internal_memory(self, memory):
        if memory is None:
            assert self.memory is not None, "No memory found"
            assert self.handler.size(self.memory) >= self.size, "Insufficient Memory"
            return False

        if memory is self.memory:
            return False

        mem_size = self.handler.size(memory)
        assert mem_size >= self.size, \
            "Given memory is too small: {} < {}".format(mem_size, self.size)
        self.memory = memory
        return True

    def _lay_out(self):
        nr_timesteps, nr_sequences = self.shape
        i = 0
        for hub_feature_size, layout in zip(self.hub_sizes, self.layouts):
            hub_shape = (nr_timesteps, nr_sequences, hub_feature_size)
            hub_size = nr_timesteps * nr_sequences * hub_feature_size
            hub_buffer = self.handler.slice(self.memory, slice(i, i+hub_size))
            hub_buffer = self.handler.reshape(hub_buffer, hub_shape)
            i += hub_size
            for layer_name, feature_slice in layout.items():
                self[layer_name] = self.handler.slice(
                    hub_buffer, (slice(None), slice(None), feature_slice))


class BufferManager(object):
    # TODO needs refactor, because it essentially does everything twice
    def __init__(self, param_buffer, in_buffer, out_buffer,
                 handler=default_handler):
        self.parameters = param_buffer
        self.gradient = copy(param_buffer)
        self.inputs = in_buffer
        self.outputs = out_buffer
        self.in_deltas = copy(in_buffer)
        self.out_deltas = copy(out_buffer)
        self.fwd_shape = None
        self.bwd_shape = None
        self.param_memory = None
        self.grad_memory = None
        self.handler = handler
        self.fwd_memory = self.handler.empty
        self.bwd_memory = self.handler.empty

    def reset(self):
        self.fwd_shape = None
        self.bwd_shape = None
        self.param_memory = None
        self.grad_memory = None
        self.fwd_memory = self.handler.empty
        self.bwd_memory = self.handler.empty

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
            self.handler.set(self.param_memory, params)
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

        if self.handler.size(self.fwd_memory) < in_size:
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
        param_layout = create_param_layout(layers)
        view_factories = {n: l.create_param_view for n, l in layers.items()}
        param_buffer = ParameterBuffer(param_layout, view_factories)

        buffer_hub_layouts = create_in_out_layout(layers)
        hub_sizes, source_hubs, sink_hubs = zip(*buffer_hub_layouts)
        out_buffer = InOutBuffer(hub_sizes, source_hubs)
        in_buffer = InOutBuffer(hub_sizes, sink_hubs)
        return cls(param_buffer, in_buffer, out_buffer)
