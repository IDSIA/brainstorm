#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from brainstorm.structure.architecture import (
    instantiate_layers_from_architecture)
from brainstorm.structure.buffers import BufferManager
from brainstorm.structure.view_references import resolve_references
from brainstorm.initializers import evaluate_initializer
from brainstorm.randomness import Seedable
from brainstorm.describable import create_from_description
from brainstorm.structure.architecture import (generate_injectors,
                                               generate_architecture)
from brainstorm.handlers import default_handler


def build_net(some_layer):
    arch = generate_architecture(some_layer)
    injs = generate_injectors(some_layer)
    return build_network_from_architecture(arch, injs)


def build_network_from_architecture(architecture, injections):
    layers = instantiate_layers_from_architecture(architecture)
    injectors = create_from_description(injections)
    buffer_manager = BufferManager.create_from_layers(layers)
    return Network(layers, buffer_manager, injectors)


class Network(Seedable):
    def __init__(self, layers, buffer_manager, injectors=None, seed=None,
                 handler=default_handler):
        super(Network, self).__init__(seed)
        self.layers = layers
        self.buffer = buffer_manager
        self.injectors = injectors or {}
        self.errors = None
        self.handler = None
        self.set_memory_handler(handler)

    @property
    def output(self):
        return self.buffer.outputs[list(self.layers.keys())[-1]]

    @property
    def in_deltas(self):
        return self.buffer.out_deltas['InputLayer']

    def set_memory_handler(self, new_handler):
        self.handler = new_handler
        self.buffer.set_memory_handler(new_handler)
        for layer in self.layers.values():
            layer.set_handler(new_handler)

    def forward_pass(self, input_data, training_pass=False):
        assert self.layers['InputLayer'].out_size == input_data.shape[2],\
            "Input dimensions mismatch of InputLayer({}) and data ({})".format(
                self.layers['InputLayer'].out_size, input_data.shape[2])
        self.errors = None
        self.buffer.rearrange_fwd(input_data.shape)
        self.handler.set(self.buffer.outputs['InputLayer'], input_data)
        for layer_name, layer in list(self.layers.items())[1:]:
            parameters = self.buffer.parameters[layer_name]
            input_buffer = self.buffer.inputs[layer_name]
            output_buffer = self.buffer.outputs[layer_name]
            layer.forward_pass(parameters, input_buffer, output_buffer)

    def calculate_errors(self, targets):
        self._calculate_deltas_and_error(targets)
        return self.errors

    def _calculate_deltas_and_error(self, targets):
        assert self.injectors, "No error injectors!"
        if self.errors is None:
            self.errors = {}
            self.buffer.rearrange_bwd()

            for inj_name, injector in self.injectors.items():
                error, deltas = injector(
                    self.buffer.outputs[injector.layer],
                    targets.get(injector.target_from))
                self.handler.copyto(self.buffer.out_deltas[injector.layer],
                                    deltas)
                self.errors[inj_name] = error

    def backward_pass(self, targets):
        self._calculate_deltas_and_error(targets)
        self.handler.fill(self.buffer.gradient[:], 0.)
        for layer_name, layer in reversed(list(self.layers.items())[1:]):
            parameters = self.buffer.parameters[layer_name]
            input_buffer = self.buffer.inputs[layer_name]
            output_buffer = self.buffer.outputs[layer_name]
            in_delta_buffer = self.buffer.in_deltas[layer_name]
            out_delta_buffer = self.buffer.out_deltas[layer_name]
            gradient_buffer = self.buffer.gradient[layer_name]
            layer.backward_pass(parameters, input_buffer, output_buffer,
                                in_delta_buffer, out_delta_buffer,
                                gradient_buffer)

    def initialize(self, init_dict=None, seed=None, **kwargs):
        init_refs = _update_references_with_dict(init_dict, kwargs)
        self.buffer.rearrange_parameters()
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
                self.handler.set(
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
