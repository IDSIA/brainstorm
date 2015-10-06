#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from collections import OrderedDict
from brainstorm.structure.construction import ConstructionWrapper
from brainstorm.utils import flatten_time, flatten_time_and_features
from brainstorm.layers.base_layer import BaseLayerImpl
from brainstorm.structure.buffer_structure import (StructureTemplate,
                                                   BufferStructure)
from brainstorm.handlers.base_handler import Handler

def L1Decay(name=None):
    """Add L1 regularization to the activations of a layer."""
    return ConstructionWrapper.create('L1Decay', name=name)


class L1DecayLayerImpl(BaseLayerImpl):

    expected_inputs = {'default': StructureTemplate('T', 'B', '...')}
    expected_kwargs = {}

    def setup(self, kwargs, in_shapes):
        outputs = OrderedDict()
        outputs['loss'] = BufferStructure('T', 'B', 1)

        parameters = OrderedDict()
        internals = OrderedDict()
        internals['abs_activations'] = in_shapes['default']
        internals['sign_activations'] = BufferStructure(
            *in_shapes['default'].shape, is_backward_only=True)

        return outputs, parameters, internals

    def forward_pass(self, buffers, training_pass=True):
        # prepare
        _h = self.handler
        assert isinstance(_h, Handler)
        inputs = buffers.inputs.default
        abs_activations = buffers.internals.abs_activations
        outputs = buffers.outputs.loss

        # reshape
        flat_inputs = flatten_time_and_features(inputs)
        flat_abs_activations = flatten_time_and_features(abs_activations)
        flat_outputs = flatten_time(outputs)

        # compute
        _h.abs_t(flat_inputs, flat_abs_activations)
        _h.sum_t(flat_abs_activations, 1, flat_outputs)

    def backward_pass(self, buffers):
        _h = self.handler
        assert isinstance(_h, Handler)
        inputs = buffers.inputs.default
        sign_activations = buffers.internals.abs_activations
        output_deltas = buffers.output_deltas.loss
        input_deltas = buffers.input_deltas.default

        # reshape
        flat_inputs = flatten_time_and_features(inputs)
        flat_sign_activations = flatten_time_and_features(sign_activations)
        flat_output_deltas = flatten_time(output_deltas)
        flat_input_deltas = flatten_time_and_features(input_deltas)

        # compute
        _h.sign_t(flat_inputs, flat_sign_activations)
        _h.mult_mv(flat_sign_activations, flat_output_deltas, flat_sign_activations)
        _h.add_tt(flat_sign_activations, flat_input_deltas, flat_input_deltas)
