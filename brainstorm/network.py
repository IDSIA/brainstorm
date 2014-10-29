#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import numpy as np
from brainstorm.architecture import instantiate_layers_from_architecture
from brainstorm.buffers import create_param_layout, create_in_out_layout


def build_network_from_architecture(architecture):
    layers = instantiate_layers_from_architecture(architecture)
    param_layout = create_param_layout(layers)
    in_out_layout = create_in_out_layout(layers)


class ParameterBuffer(dict):
    """
    Handles the parameters of the network.
    The buffer is allocated at initialization, and the views for all the
    layers are created.
    """
    def __init__(self, param_layout, layers, buffer=None):
        super(ParameterBuffer, self).__init__()
        self.size, self.layout = param_layout
        if buffer is None:
            self.buffer = np.zeros(self.size)
        else:
            assert buffer.size == self.size
            self.buffer = buffer

        for layer_name in self.layout:
            view = layers.create_param_view(self.get_raw(layer_name))
            self[layer_name] = view

    def get_raw(self, layer_name=None):
        """
        Get the part of the buffer that corresponds to the given layer, or the
        the whole buffer if none is specified.
        """
        if layer_name is None:
            return self.buffer
        else:
            return self.buffer.__getitem__(self.layout[layer_name])


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
        self.buffer = None

    def rearrange_buffer(self, timesteps, sequences, buffer=None):
        self.size = timesteps * sequences * sum(self.hub_sizes)
        if buffer is not None:
            assert buffer.size >= self.size
            self.buffer = buffer

        if not self.buffer or self.buffer.size < self.size:
            self.buffer = np.zeros(self.size)

        i = 0
        for hub_feature_size, layout in zip(self.hub_sizes, self.layouts):
            hub_size = hub_feature_size * timesteps * sequences
            hub_buffer = self.buffer[i:i+hub_size].reshape((timesteps,
                                                            sequences,
                                                            hub_feature_size))
            i += hub_size
            for layer_name, feature_slice in layout.items():
                self[layer_name] = hub_buffer[:, :, feature_slice]


class Network(object):
    pass