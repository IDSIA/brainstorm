#!/usr/bin/env python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import numpy as np
from brainstorm.layout import create_param_layout, create_in_out_layout


class ParameterBuffer(dict):
    """
    Handles the parameters of the network.
    The buffer is allocated at initialization, and the views for all the
    layers are created.
    """
    def __init__(self, param_layout, view_factories, memory=None):
        super(ParameterBuffer, self).__init__()
        self.size, self.layout = param_layout
        if memory is None:
            self.memory = np.zeros(self.size)
        else:
            assert memory.size == self.size, \
                "Given memory is too small {} < {}".format(memory.size,
                                                           self.size)
            self.memory = memory

        for layer_name in self.layout:
            view = view_factories[layer_name](self.get_raw(layer_name))
            self[layer_name] = view

    def get_raw(self, layer_name=None):
        """
        Get the part of the memory that corresponds to the given layer, or the
        the whole buffer if none is specified.
        """
        if layer_name is None:
            return self.memory
        else:
            return self.memory.__getitem__(self.layout[layer_name])


class InOutBuffer(dict):
    """
    Handles input or output buffers. The memory is allocated on demand.
    There should always be one of this object for the inputs and one for the
    outputs with corresponding layouts that share the same memory region.
    """
    def __init__(self, hub_sizes, layouts):
        super(InOutBuffer, self).__init__()
        self.hub_sizes = hub_sizes
        self.size = 0
        self.layouts = layouts
        self.memory = None
        self.shape = None

    def rearrange_buffer(self, shape, memory=None):
        self.size = self._get_size(shape)
        relocated = self._resize_internal_memory(memory)
        self._lay_out(shape, relocated)

    def _get_size(self, shape):
        nr_timesteps, nr_sequences = shape[:2]
        return nr_timesteps * nr_sequences * sum(self.hub_sizes)

    def _resize_internal_memory(self, memory=None):
        if memory is not None:
            assert memory.size >= self.size
            self.memory = memory
            return True
        elif self.memory is None or self.memory.size < self.size:
            self.memory = np.zeros(self.size)
            return True
        return False

    def _lay_out(self, shape, relocate=False):
        if self.shape == shape and not relocate:
            return
        self.shape = shape
        nr_timesteps, nr_sequences = shape[:2]
        i = 0
        for hub_feature_size, layout in zip(self.hub_sizes, self.layouts):
            hub_size = hub_feature_size * nr_timesteps * nr_sequences
            hub_buffer = self.memory[i:i+hub_size].reshape((nr_timesteps,
                                                            nr_sequences,
                                                            hub_feature_size))
            i += hub_size
            for layer_name, feature_slice in layout.items():
                self[layer_name] = hub_buffer[:, :, feature_slice]


class BufferManager(object):
    def __init__(self, param_buffer, in_buffer, out_buffer):
        self.parameters = param_buffer
        self.inputs = in_buffer
        self.outputs = out_buffer
        self.shape = None

    def rearrange(self, shape):
        """
        Resize the buffers and prepare them.
        :param shape: Tuple specifying the dimensions. Only the first two are
            used. They should be (nr_timesteps, nr_sequences).
        :type shape: tuple[int]
        """
        if self.shape == shape:
            return
        self.shape = shape[:2]
        # do nothing to the parameters
        self.inputs.rearrange_buffer(self.shape)
        self.outputs.rearrange_buffer(self.shape, self.inputs.buffer)

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
