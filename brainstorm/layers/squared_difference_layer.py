#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from collections import OrderedDict
from brainstorm.structure.construction import ConstructionWrapper
from brainstorm.layers.base_layer import BaseLayerImpl
from brainstorm.utils import LayerValidationError, flatten_time, \
    flatten_time_and_features, flatten_features
from brainstorm.structure.shapes import BufferStructure, StructureTemplate


def SquaredDifference(name=None):
    """Create a Squared Difference layer.


class SquaredDifferenceLayerImpl(LayerBaseImpl):
    """
    A layer that computes half of the squared differences between two inputs,
    and sums them over feature dimensions.
    """
    inputs = {'inputs_1': ShapeTemplate('T', 'B', '...'),
              'inputs_2': ShapeTemplate('T', 'B', '...')}

    outputs = {'default': ShapeTemplate('T', 'B', 1)}

    expected_kwargs = {}

    def get_internal_structure(self):
        """
        Returns a dictionary describing the 'squared_diff' internal-state.
        """
        feature_shape = self.in_shapes['inputs_1'].feature_shape

        internals = OrderedDict()
        internals['squared_diff'] = ShapeTemplate('T', 'B', *feature_shape)
        internals['grad_diff'] = ShapeTemplate('T', 'B', *feature_shape,
                                               is_backward_only=True)
        return internals

    def _get_output_shapes(self):
        """
        Sets the shape of the 'default' output using in_shapes['inputs_1']
        """
        return {'default': ShapeTemplate('T', 'B', 1)}

    def _validate_in_shapes(self):
        """Ensure self.in_shapes are all valid.

         Raise LayerValidationError otherwise."""
        super(SquaredDifferenceLayerImpl, self)._validate_in_shapes()

        # 'inputs_1' and 'inputs_2' must have same shape
        if in_shapes['inputs_1'] != in_shapes['inputs_2']:
            raise LayerValidationError("{}: inputs_1 and inputs_2 must have "
                                       "same shape but got {} and {}"
                                       .format(self.name,
                                               in_shapes['inputs_1'],
                                               in_shapes['inputs_2']))

        outputs = OrderedDict()
        outputs['default'] = BufferStructure('T', 'B', 1)

        internals = OrderedDict()
        feature_shape = self.in_shapes['inputs_1'].feature_shape
        internals['squared_diff'] = BufferStructure('T', 'B', *feature_shape)
        internals['grad_diff'] = BufferStructure('T', 'B', *feature_shape,
                                                 is_backward_only=True)
        return outputs, OrderedDict(), internals

    def forward_pass(self, buffers, training_pass=True):
        # prepare
        _h = self.handler
        inputs_1 = buffers.inputs.inputs_1
        inputs_2 = buffers.inputs.inputs_2
        diff = buffers.internals.squared_diff
        diff_sum = buffers.outputs.default

        flat_inputs_1 = flatten_time_and_features(inputs_1)
        flat_inputs_2 = flatten_time_and_features(inputs_2)
        flat_diff = flatten_time_and_features(diff)
        flat_diff_sum = flatten_time(diff_sum)

        # calculate
        _h.subtract_tt(flat_inputs_1, flat_inputs_2, out=flat_diff)
        _h.mult_tt(flat_diff, flat_diff, out=flat_diff)
        _h.sum_t(flat_diff, axis=1, out=flat_diff_sum)
        _h.mult_st(0.5, flat_diff_sum, out=flat_diff_sum)

    def backward_pass(self, buffers):
        # prepare
        _h = self.handler
        grad_diff_sum = buffers.output_deltas.default
        grad_diff = buffers.internals.grad_diff
        grad_inputs_1 = buffers.input_deltas.inputs_1
        grad_inputs_2 = buffers.input_deltas.inputs_2
        inputs_1 = buffers.inputs.inputs_1
        inputs_2 = buffers.inputs.inputs_2
        tmp = _h.allocate(inputs_1.shape)

        # grad_diff_sum has only one feature dimension due to summation,
        # so we broadcast to all feature dimensions
        flat_grad_diff = flatten_features(grad_diff)
        _h.broadcast_features_t(grad_diff_sum, flat_grad_diff)

        # calculate
        _h.subtract_tt(inputs_1, inputs_2, out=tmp)
        _h.mult_add_tt(grad_diff, tmp, grad_inputs_1)

        _h.subtract_tt(inputs_2, inputs_1, out=tmp)
        _h.mult_add_tt(grad_diff, tmp, grad_inputs_2)
