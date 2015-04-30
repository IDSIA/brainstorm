#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np

from brainstorm.structure.architecture import (
    instantiate_layers_from_architecture)
from brainstorm.structure.buffers import BufferManager
from brainstorm.structure.layout import create_layout
from brainstorm.structure.view_references import (resolve_references,
                                                  prune_view_references,
                                                  order_and_copy_modifiers)
from brainstorm.initializers import evaluate_initializer, ArrayInitializer
from brainstorm.randomness import Seedable
from brainstorm.structure.architecture import generate_architecture
from brainstorm.handlers import default_handler
from brainstorm.utils import NetworkValidationError
from brainstorm.layers.loss_layer import LossLayerImpl


def build_net(some_layer):
    arch = generate_architecture(some_layer)
    return build_network_from_architecture(arch)


def build_network_from_architecture(architecture):
    layers = instantiate_layers_from_architecture(architecture)
    buffer_sizes, max_context_size, layout = create_layout(layers)
    buffer_manager = BufferManager(layout, buffer_sizes, max_context_size)
    return Network(layers, buffer_manager)


def get_loss_layers(layers):
    return [name for name, l in layers.items() if isinstance(l, LossLayerImpl)]


class Network(Seedable):
    def __init__(self, layers, buffer_manager, seed=None,
                 handler=default_handler):
        super(Network, self).__init__(seed)
        self.layers = layers
        self.loss_layers = get_loss_layers(layers)
        self.buffer = buffer_manager
        self.errors = None
        self.handler = None
        self.set_memory_handler(handler)
        self.weight_modifiers = {}
        self.gradient_modifiers = {}

    def set_memory_handler(self, new_handler):
        self.handler = new_handler
        self.buffer.set_memory_handler(new_handler)
        for layer in self.layers.values():
            layer.set_handler(new_handler)

    def provide_external_data(self, data):
        time_size, batch_size = data[next(iter(data))].shape[:2]
        self.buffer.resize(time_size, batch_size)
        for name, val in data.items():
            self.handler.copy_to(self.buffer.forward.InputLayer.outputs[name],
                                 val)

    def get_context(self):
        return self.buffer.get_context()

    def forward_pass(self, training_pass=False, context=None):
        if context is None:
            self.buffer.clear_context()
        else:
            self.buffer.apply_context(context)
        for layer_name, layer in list(self.layers.items())[1:]:
            layer.forward_pass(self.buffer.forward[layer_name], training_pass)

    def backward_pass(self):
        self.buffer.clear_backward_buffers()
        for layer_name, layer in reversed(list(self.layers.items())[1:]):
            layer.backward_pass(self.buffer.forward[layer_name],
                                self.buffer.backward[layer_name])
        self.apply_gradient_modifiers()

    def get_loss_value(self):
        loss = 0.
        for loss_layer_name in self.loss_layers:
            loss += float(self.handler.get_numpy_copy(
                self.buffer.forward[loss_layer_name].outputs.loss))
        return loss

    def initialize(self, default_or_init_dict=None, seed=None, **kwargs):
        """Initialize the weights of the network.

        Initialization can be specified in two equivalent ways:
          1) just a default initializer:
          >> net.initialize(bs.Gaussian())
          Note that this is equivalent to:
          >> net.initialize(default=bs.Gaussian())

          2) by passing a dictionary:
          >> net.initialize({'RegularLayer': bs.Uniform(),
                                'LstmLayer': bs.Gaussian()})

          3) by using keyword arguments:
          >> net.initialize(RegularLayer=bs.Uniform(),
                            LstmLayer=bs.Uniform())

        All following explanations will be with regards to the dictionary style
        of initialization, because it is the most general one.

        Note: It is not recommended to combine 2) and 3) but if they are, then
        keyword arguments take precedence.

        Each initialization consists of a layer-pattern and that maps to an
        initializer or a weight-pattern dictionary.

        Layer patterns can take the following forms:
          1) {'layer_name': INIT_OR_SUBDICT}
             Matches all the weights of the layer named layer_name
          2) {'layer_*': INIT_OR_SUBDICT}
             Matches all layers with a name that starts with 'layer_'
             The wild-card '*' can appear at arbitrary positions and even
             multiple times in one path.

        There are two special layer patterns:
          3) {'default': INIT}
             Matches all weights that are not matched by any other path-pattern
          4) {'fallback': INIT}
             Set a fallback initializer for every weight. It will only be
             evaluated for the weights for which the regular initializer failed
             with an InitializationError.
             (This is useful for initializers that require a certain shape of
              weights and will not work otherwise. The fallback will then be
              used for all cases when that initializer failed.)

        The weight-pattern sub-dictionary follows the same form as the layer-
        pattern:
          1) {'layer_pattern': {'a': INIT_A, 'b': INIT_B}}
          2) {'layer_pattern': {'a*': INIT}
          3) {'layer_pattern': {'default': INIT}
          4) {'layer_pattern': {'fallback': INIT}


        An initializer can either be a scalar, something that converts to a
        numpy array of the correct shape or an Initializer object.
        So for example:
        >> net.initialize(
            default=0,
            RnnLayer={'b': [1, 2, 3, 4, 5]},
            ForwardLayer=bs.Gaussian())

        Note: Each view must match exactly one initialization and up to one
        fallback to be unambiguous. Otherwise the initialization will fail.

        You can specify a seed to make the initialization reproducible:
        >> net.initialize({'default': bs.Gaussian()}, seed=1234)
        """
        init_refs = _update_references_with_dict(default_or_init_dict, kwargs)
        all_parameters = {k: v.parameters
                          for k, v in self.buffer.forward.items()
                          if 'parameters' in v}
        replace_lists_with_array_initializers(init_refs)
        initializers, fallback = resolve_references(all_parameters, init_refs)
        init_rnd = self.rnd.create_random_state(seed)
        for layer_name, views in sorted(all_parameters.items()):
            if views is None:
                continue
            for view_name, view in sorted(views.items()):
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

    def set_weight_modifiers(self, default_or_mod_dict=None, **kwargs):
        """Install WeightModifiers in the network that can change the weights.

        They can be run manually using net.apply_weight_modifiers(), but they
        will also be called by the trainer after each weight update.

        WeightModifiers can be set for specific weights in the same way as
        initializers can, but there is no 'fallback'.
        (so look there for details)

        A modifier can be a WeightModifier object or a list of them.
        So for example:
        >> net.set_weight_modifiers(
            default=bs.ClipWeights(-1, 1)
            FullyConnectedLayer={'W': [bs.RescaleIncomingWeights(),
                                       bs.MaskWeights(my_mask)]}
            )

        Note: The order in which WeightModifiers appear in the list matters,
        because it is the same order in which they will be executed.
        """
        weight_mod_refs = _update_references_with_dict(default_or_mod_dict,
                                                       kwargs)
        all_parameters = {k: v.parameters
                          for k, v in self.buffer.forward.items()
                          if 'parameters' in v}
        weight_mods, fallback = resolve_references(all_parameters,
                                                   weight_mod_refs)

        assert not prune_view_references(fallback), \
            'fallback is not supported for weight modifiers'
        weight_mods = prune_view_references(weight_mods)
        self.weight_modifiers = order_and_copy_modifiers(weight_mods)

    def set_gradient_modifiers(self, default_or_mod_dict=None, **kwargs):
        """
        Install WeightModifiers in the network that can change the gradient.

        They can be run manually using net.apply_gradient_modifiers(), but they
        will also be called by the network after each backward pass.

        Gradient modifiers can be set for specific weights in the same way as
        initializers can, but there is no 'fallback'.
        (so look there for details)

        A modifier can be a WeightModifier object or a list of them.
        So for example:
        >> net.set_gradient_modifiers(
            default=bs.ClipWeights(-1, 1)
            FullyConnectedLayer={'W': [bs.ClipWeights(),
                                       bs.MaskWeights(my_mask)]}
            )

        Note: The order in which WeightModifiers appear in the list matters,
        because it is the same order in which they will be executed.
        """
        gradient_mod_refs = _update_references_with_dict(default_or_mod_dict,
                                                         kwargs)
        all_parameters = {k: v.parameters
                          for k, v in self.buffer.forward.items()
                          if 'parameters' in v}
        gradient_mods, fallback = resolve_references(all_parameters,
                                                     gradient_mod_refs)

        assert not prune_view_references(fallback), \
            'fallback is not supported for gradient modifiers'
        gradient_mods = prune_view_references(gradient_mods)
        self.gradient_modifiers = order_and_copy_modifiers(gradient_mods)

    def apply_weight_modifiers(self):
        for layer_name, views in self.weight_modifiers.items():
            for view_name, weight_mods in views.items():
                for wm in weight_mods:
                    wm.rnd.set_seed(self.rnd.generate_seed())
                    wm(self.handler,
                       self.buffer.forward[layer_name].parameters[view_name])

    def apply_gradient_modifiers(self):
        for layer_name, views in self.gradient_modifiers.items():
            for view_name, gradient_mods in views.items():
                for gm in gradient_mods:
                    gm.rnd.set_seed(self.rnd.generate_seed())
                    gm(self.handler,
                       self.buffer.backward[layer_name].parameters[view_name])


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


def replace_lists_with_array_initializers(ref_dict):
    for key, value in ref_dict.items():
        if isinstance(value, dict):
            replace_lists_with_array_initializers(value)
        elif isinstance(value, (list, np.ndarray)):
            ref_dict[key] = ArrayInitializer(value)
