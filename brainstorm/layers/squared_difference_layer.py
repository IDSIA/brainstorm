#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

from collections import OrderedDict

from brainstorm.layers.base_layer import Layer
from brainstorm.structure.buffer_structure import (BufferStructure,
                                                   StructureTemplate)
from brainstorm.structure.construction import ConstructionWrapper
from brainstorm.utils import (LayerValidationError, flatten_time_and_features)


def SquaredDifference(name=None):
    """Create a Squared Difference layer."""
    return ConstructionWrapper.create(SquaredDifferenceLayerImpl, name=name)


class SquaredDifferenceLayerImpl(Layer):

    expected_inputs = {'inputs_1': StructureTemplate('T', 'B', '...'),
                       'inputs_2': StructureTemplate('T', 'B', '...')}
    expected_kwargs = {}

    def setup(self, kwargs, in_shapes):
        # 'inputs_1' and 'inputs_2' must have same shape
        f_shape1 = in_shapes['inputs_1'].feature_shape
        f_shape2 = in_shapes['inputs_2'].feature_shape
        if f_shape1 != f_shape2:
            raise LayerValidationError(
                "{}: inputs_1 and inputs_2 must have same feature shapes but "
                "got {} and {}".format(self.name, f_shape1, f_shape2))

        outputs = OrderedDict()
        outputs['default'] = BufferStructure('T', 'B', *f_shape1)

        internals = OrderedDict()
        feature_shape = self.in_shapes['inputs_1'].feature_shape
        internals['diff'] = BufferStructure('T', 'B', *feature_shape)
        return outputs, OrderedDict(), internals

    def forward_pass(self, buffers, training_pass=True):
        # prepare
        _h = self.handler
        inputs_1 = flatten_time_and_features(buffers.inputs.inputs_1)
        inputs_2 = flatten_time_and_features(buffers.inputs.inputs_2)
        diff = flatten_time_and_features(buffers.internals.diff)
        outputs = flatten_time_and_features(buffers.outputs.default)

        # calculate
        _h.subtract_tt(inputs_1, inputs_2, out=diff)
        _h.mult_tt(diff, diff, out=outputs)

    def backward_pass(self, buffers):
        # prepare
        _h = self.handler
        out_deltas = flatten_time_and_features(buffers.output_deltas.default)
        diff = flatten_time_and_features(buffers.internals.diff)
        dinputs_1 = flatten_time_and_features(buffers.input_deltas.inputs_1)
        dinputs_2 = flatten_time_and_features(buffers.input_deltas.inputs_2)

        tmp = _h.allocate(out_deltas.shape)
        # calculate
        _h.mult_st(2, out_deltas, out=out_deltas)
        _h.mult_add_tt(out_deltas, diff, out=dinputs_1)
        _h.mult_st(-1, diff, out=tmp)
        _h.mult_add_tt(out_deltas, tmp, out=dinputs_2)
