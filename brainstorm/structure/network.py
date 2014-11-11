#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from brainstorm.structure.architecture import (
    instantiate_layers_from_architecture)
from brainstorm.structure.buffers import BufferManager
from brainstorm.structure.view_references import resolve_references
from brainstorm.initializers import evaluate_initializer
from brainstorm.randomness import Seedable


def build_network_from_architecture(architecture):
    layers = instantiate_layers_from_architecture(architecture)
    buffer_manager = BufferManager.create_from_layers(layers)
    return Network(layers, buffer_manager)


class Network(Seedable):
    def __init__(self, layers, buffer_manager, seed=None):
        super(Network, self).__init__(seed)
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

    def initialize(self, init_dict=None, seed=None, **kwargs):
        init_refs = _update_references_with_dict(init_dict, kwargs)
        self.buffer.parameters.rearrange()
        initializers, fallback = resolve_references(self.buffer.parameters,
                                                    init_refs)
        init_rnd = self.rnd.create_random_state(seed)
        for layer_name, views in self.buffer.parameters.items():
            if views is None:
                continue
            for view_name, view in views.items():
                init = initializers[layer_name][view_name]
                assert len(init) <= 1, "Multiple initializers for {}.{}: {}" \
                                       "".format(layer_name, view_name, init)
                assert len(init) > 0, "No initializer for {}.{}".format(
                    layer_name, view_name)
                fb = fallback[layer_name][view_name]
                assert len(fb) <= 1, "Multiple fallbacks for {}.{}: {}".format(
                    layer_name, view_name, fb)
                fb = fb.pop() if len(fb) else None
                view[:] = evaluate_initializer(init.pop(), view.shape, fb,
                                               seed=init_rnd.generate_seed())


def _update_references_with_dict(refs, ref_dict):
    if refs is None:
        references = dict()
    elif isinstance(refs, dict):
        references = refs
    else:
        references = {'default': refs}

    if set(references.keys()) & set(ref_dict.keys()):
        raise TypeError('Conflicting values for %s!' %
                        sorted(set(references.keys()) & set(ref_dict.keys())))

    references.update(ref_dict)

    return references