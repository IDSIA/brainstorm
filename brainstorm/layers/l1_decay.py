#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

from collections import OrderedDict

from brainstorm.handlers.base_handler import Handler
from brainstorm.layers.base_layer import Layer
from brainstorm.structure.buffer_structure import (BufferStructure,
                                                   StructureTemplate)
from brainstorm.structure.construction import ConstructionWrapper
from brainstorm.utils import flatten_time, flatten_time_and_features


def L1Decay(name=None):
    """Add L1 regularization to the activations of a layer."""
    return ConstructionWrapper.create(L1DecayLayerImpl, name=name)


class L1DecayLayerImpl(Layer):

    expected_inputs = {'default': StructureTemplate('T', 'B', '...')}
    expected_kwargs = {}

    def setup(self, kwargs, in_shapes):
        outputs = OrderedDict()
        outputs['loss'] = BufferStructure('T', 'B', 1)

        parameters = OrderedDict()
        internals = OrderedDict()
        internals['tmp'] = in_shapes['default']

        return outputs, parameters, internals

    def forward_pass(self, buffers, training_pass=True):
        # prepare
        _h = self.handler
        assert isinstance(_h, Handler)
        inputs = buffers.inputs.default
        tmp = buffers.internals.tmp
        outputs = buffers.outputs.loss

        # reshape
        flat_inputs = flatten_time_and_features(inputs)
        flat_tmp = flatten_time_and_features(tmp)
        flat_outputs = flatten_time(outputs)

        # compute
        _h.abs_t(flat_inputs, flat_tmp)
        _h.sum_t(flat_tmp, 1, flat_outputs)

    def backward_pass(self, buffers):
        _h = self.handler
        assert isinstance(_h, Handler)
        inputs = buffers.inputs.default
        tmp = buffers.internals.tmp
        output_deltas = buffers.output_deltas.loss
        input_deltas = buffers.input_deltas.default

        # reshape
        flat_inputs = flatten_time_and_features(inputs)
        flat_tmp = flatten_time_and_features(tmp)
        flat_output_deltas = flatten_time(output_deltas)
        flat_input_deltas = flatten_time_and_features(input_deltas)

        # compute
        _h.sign_t(flat_inputs, flat_tmp)
        _h.mult_mv(flat_tmp, flat_output_deltas, flat_tmp)
        _h.add_tt(flat_tmp, flat_input_deltas, flat_input_deltas)
