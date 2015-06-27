#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from collections import OrderedDict
from brainstorm.layers.base_layer import LayerBaseImpl
from brainstorm.utils import LayerValidationError
from brainstorm.structure.shapes import ShapeTemplate


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
        if self.in_shapes['inputs_1'] != self.in_shapes['inputs_2']:
            raise LayerValidationError("{}: inputs_1 and inputs_2 must have "
                                       "same shape but got {} and {}"
                                       .format(self.name,
                                               self.in_shapes['inputs_1'],
                                               self.in_shapes['inputs_2']))

    def forward_pass(self, forward_buffers, training_pass=True):
        # prepare
        _h = self.handler
        inputs_1 = forward_buffers.inputs.inputs_1
        inputs_2 = forward_buffers.inputs.inputs_2
        diff = forward_buffers.internals.squared_diff
        diff_sum = forward_buffers.outputs.default

        t, b = inputs_1.shape[0], inputs_1.shape[1]
        f = _h.size(inputs_1) / (t * b)
        flat_inputs_1 = _h.reshape(inputs_1, ((t * b), f))
        flat_inputs_2 = _h.reshape(inputs_2, ((t * b), f))
        flat_diff = _h.reshape(diff, ((t * b), f))
        flat_diff_sum = _h.reshape(diff_sum, ((t * b), 1))

        # calculate
        _h.subtract_tt(flat_inputs_1, flat_inputs_2, out=flat_diff)
        _h.mult_tt(flat_diff, flat_diff, out=flat_diff)
        _h.sum_t(flat_diff, axis=1, out=flat_diff_sum)
        _h.mult_st(0.5, flat_diff_sum, out=flat_diff_sum)


        # # calculate
        # _h.subtract_tt(inputs_1, inputs_2, out=diff)
        # _h.mult_tt(diff, diff, out=diff)
        #
        # # reshape for summation
        # t, b = diff.shape[0], diff.shape[1]
        # f = _h.size(diff) / (t * b)
        # diff = _h.reshape(diff, (t, b, f))
        #
        # _h.sum_t(diff, axis=2, out=diff_sum)
        # _h.mult_st(0.5, diff_sum, out=diff_sum)

    def backward_pass(self, forward_buffers, backward_buffers):
        # prepare
        _h = self.handler
        grad_diff_sum = backward_buffers.outputs.default
        grad_diff = backward_buffers.internals.squared_diff
        grad_inputs_1 = backward_buffers.inputs.inputs_1
        grad_inputs_2 = backward_buffers.inputs.inputs_2
        inputs_1 = forward_buffers.inputs.inputs_1
        inputs_2 = forward_buffers.inputs.inputs_2
        tmp = _h.allocate(inputs_1.shape)

        # grad_diff_sum has only one feature dimension due to summation,
        # so we broadcast to all feature dimensions
        _h.broadcast_features_t(grad_diff_sum, grad_diff)

        # calculate
        _h.subtract_tt(inputs_1, inputs_2, out=tmp)
        _h.mult_add_tt(grad_diff, tmp, grad_inputs_1)

        _h.subtract_tt(inputs_2, inputs_1, out=tmp)
        _h.mult_add_tt(grad_diff, tmp, grad_inputs_2)
