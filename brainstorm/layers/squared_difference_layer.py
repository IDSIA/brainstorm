#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

from collections import OrderedDict

from brainstorm.layers.base_layer import Layer
from brainstorm.structure.buffer_structure import (BufferStructure,
                                                   StructureTemplate)
from brainstorm.structure.construction import ConstructionWrapper
from brainstorm.utils import (LayerValidationError, flatten_features,
                              flatten_time, flatten_time_and_features)


def SquaredDifference(name=None):
    """Create a Squared Difference layer."""
    return ConstructionWrapper.create(SquaredDifferenceLayerImpl, name=name)


class SquaredDifferenceLayerImpl(Layer):

    expected_inputs = {'inputs_1': StructureTemplate('T', 'B', '...'),
                       'inputs_2': StructureTemplate('T', 'B', '...')}
    expected_kwargs = {}

    def setup(self, kwargs, in_shapes):
        # 'inputs_1' and 'inputs_2' must have same shape
        f_size1 = in_shapes['inputs_1'].feature_size
        f_size2 = in_shapes['inputs_2'].feature_size
        if f_size1 != f_size2:
            raise LayerValidationError(
                "{}: inputs_1 and inputs_2 must have same feature sizes but "
                "got {} and {}".format(self.name,
                                       in_shapes['inputs_1'].feature_shape,
                                       in_shapes['inputs_2'].feature_shape))

        outputs = OrderedDict()
        outputs['default'] = BufferStructure('T', 'B', 1)

        internals = OrderedDict()
        feature_size = self.in_shapes['inputs_1'].feature_size
        internals['squared_diff'] = BufferStructure('T', 'B', feature_size)
        internals['grad_diff'] = BufferStructure('T', 'B', feature_size,
                                                 is_backward_only=True)
        return outputs, OrderedDict(), internals

    def forward_pass(self, buffers, training_pass=True):
        # prepare
        _h = self.handler
        inputs_1 = flatten_time_and_features(buffers.inputs.inputs_1)
        inputs_2 = flatten_time_and_features(buffers.inputs.inputs_2)
        diff = flatten_time_and_features(buffers.internals.squared_diff)
        diff_sum = flatten_time(buffers.outputs.default)

        # calculate
        _h.subtract_tt(inputs_1, inputs_2, out=diff)
        _h.mult_tt(diff, diff, out=diff)
        _h.sum_t(diff, axis=1, out=diff_sum)
        _h.mult_st(0.5, diff_sum, out=diff_sum)

    def backward_pass(self, buffers):
        # prepare
        _h = self.handler
        inputs_1 = flatten_time_and_features(buffers.inputs.inputs_1)
        inputs_2 = flatten_time_and_features(buffers.inputs.inputs_2)
        out_deltas = buffers.output_deltas.default
        grad_diff = buffers.internals.grad_diff
        dinputs_1 = flatten_time_and_features(buffers.input_deltas.inputs_1)
        dinputs_2 = flatten_time_and_features(buffers.input_deltas.inputs_2)

        tmp = _h.allocate(inputs_2.shape)
        # out_deltas has only one feature dimension due to summation,
        # so we broadcast to all feature dimensions
        _h.broadcast_t(out_deltas, 2, grad_diff)

        grad_diff = flatten_time(grad_diff)
        # calculate
        _h.subtract_tt(inputs_1, inputs_2, out=tmp)
        _h.mult_add_tt(grad_diff, tmp, dinputs_1)

        _h.subtract_tt(inputs_2, inputs_1, out=tmp)
        _h.mult_add_tt(grad_diff, tmp, dinputs_2)
