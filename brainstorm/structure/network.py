#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from brainstorm.structure.architecture import (
    instantiate_layers_from_architecture)
from brainstorm.structure.buffers import BufferManager


def build_network_from_architecture(architecture):
    layers = instantiate_layers_from_architecture(architecture)
    buffer_manager = BufferManager.create_from_layers(layers)
    return Network(layers, buffer_manager)


class Network(object):
    def __init__(self, layers, buffer_manager):
        self.layers = layers
        self.buffer = buffer_manager

    def forward_pass(self, input_data):
        assert self.layers['InputLayer'].out_size == input_data.shape[2],\
            "Input dimensions mismatch of InputLayer({}) and data ({})".format(
                self.layers['InputLayer'].out_size, input_data.shape[2])
        self.buffer.rearrange(input_data.shape)
        self.buffer.outputs['InputLayer'][:] = input_data
        for layer_name, layer in self.layers.items()[1:]:
            parameters = self.buffer.parameters[layer_name]
            input_buffer = self.buffer.inputs[layer_name]
            output_buffer = self.buffer.outputs[layer_name]
            layer.forward_pass(parameters, input_buffer, output_buffer)

        return self.buffer.outputs[self.layers.keys()[-1]]
