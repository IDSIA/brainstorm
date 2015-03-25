#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from brainstorm.structure.architecture import (
    instantiate_layers_from_architecture)
from brainstorm.structure.buffers import BufferManager
from brainstorm.structure.layout import create_layout
from brainstorm.structure.view_references import resolve_references
from brainstorm.initializers import evaluate_initializer
from brainstorm.randomness import Seedable
from brainstorm.structure.architecture import generate_architecture
from brainstorm.handlers import default_handler
from brainstorm.utils import NetworkValidationError


def build_net(some_layer):
    arch = generate_architecture(some_layer)
    return build_network_from_architecture(arch)


def build_network_from_architecture(architecture):
    layers = instantiate_layers_from_architecture(architecture)
    buffer_sizes, max_time_offset, layout = create_layout(layers)
    buffer_manager = BufferManager(layout, buffer_sizes, max_time_offset)
    return Network(layers, buffer_manager)


class Network(Seedable):
    def __init__(self, layers, buffer_manager, seed=None,
                 handler=default_handler):
        super(Network, self).__init__(seed)
        self.layers = layers
        self.buffer = buffer_manager
        self.errors = None
        self.handler = None
        self.set_memory_handler(handler)

    def set_memory_handler(self, new_handler):
        self.handler = new_handler
        self.buffer.set_memory_handler(new_handler)
        for layer in self.layers.values():
            layer.set_handler(new_handler)

    def provide_external_data(self, data):
        time_size, batch_size = data[next(iter(data))].shape[:2]
        self.buffer.resize(time_size, batch_size)
        for name, val in data.items():
            self.handler.copy_to(self.buffer.forward.InputLayer.outputs[name], val)

    def forward_pass(self):
        for layer_name, layer in list(self.layers.items())[1:]:
            layer.forward_pass(self.buffer.forward[layer_name])

    def backward_pass(self):
        for layer_name, layer in reversed(list(self.layers.items())[1:]):
            layer.backward_pass(self.buffer.forward[layer_name],
                                self.buffer.backward[layer_name])

    def initialize(self, init_dict=None, seed=None, **kwargs):
        init_refs = _update_references_with_dict(init_dict, kwargs)
        all_parameters = {k: v.parameters
                          for k, v in self.buffer.forward.items()}
        initializers, fallback = resolve_references(all_parameters, init_refs)
        print(initializers, fallback)
        init_rnd = self.rnd.create_random_state(seed)
        for layer_name, views in all_parameters.items():
            if views is None:
                continue
            for view_name, view in views.items():
                init = initializers[layer_name][view_name]
                fb = fallback[layer_name][view_name]
                if len(init) > 1:
                    raise NetworkValidationError(
                        "Multiple initializers for {}.{}: {}".format(
                            layer_name, view_name, init))

                if len(init) == 0:
                    raise NetworkValidationError("No initializer for {}.{}".
                                                 format(layer_name, view_name))
                if len(fb) > 1:
                    raise NetworkValidationError(
                        "Multiple fallbacks for {}.{}: {}".format(
                            layer_name, view_name, fb))

                fb = fb.pop() if len(fb) else None
                self.handler.set_from_numpy(
                    view,
                    evaluate_initializer(init.pop(), view.shape, fb,
                                         seed=init_rnd.generate_seed()))


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
