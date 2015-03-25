#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from brainstorm.layers.base_layer import LayerBase
from brainstorm.utils import LayerValidationError


class SquaredDifferenceLayer(LayerBase):
    """
    A layer that computes half of the squared differences between two inputs,
    and sums them over feature dimensions.
    """
    inputs = {'inputs_1': ('T', 'B', 'F'),
              'inputs_2': ('T', 'B', 'F')
              }
    """Names and shape-templates for all inputs of this layer"""

    outputs = {'default': ('T', 'B', 1)}
    """Names and shape-templates for all outputs of this layer"""

    expected_kwargs = {}
    """Set of all kwargs that this layer accepts"""

    def _get_output_shapes(self):
        """
        Sets the shape of the 'default' output using in_shapes['inputs_1']
        """
        return {'default': (self.in_shapes['inputs_1'][0],
                            self.in_shapes['inputs_1'][1], 1)}

    def get_internal_structure(self):
        """
        Returns a dictionary describing the 'squared_diff' internal-state.
        """
        feature_shape = self.in_shapes['inputs_1'][2:]
        return {
            'squared_diff': {
                '@shape': ('T', 'B') + feature_shape,
                '@index': 0}
        }

    def _validate_in_shapes(self):
        """Ensure self.in_shapes are all valid.

         Raise LayerValidationError otherwise."""

        # All wired inputs must be supported and be at least 3-dimensional
        for input_name, in_shape in self.in_shapes.items():
            if input_name not in self.inputs:
                raise LayerValidationError(
                    'Invalid in_shapes. {} has no input named "{}". '
                    'Choices are: {}'.format(self.name, input_name,
                                             self.inputs))

            if len(in_shape) < 3:
                raise LayerValidationError(
                    "{}: in_shape ({}) for {} must be at least length 3 to be a"
                    " valid input for FramewiseMSELayer".format(
                        self.name, in_shape,
                        input_name, self.inputs[input_name]))

        # 'inputs_1' and 'inputs_2' must be wired in
        # and their first two dimensions must match
        if 'inputs_1' not in self.in_shapes or 'inputs_2' not in self.in_shapes:
            raise LayerValidationError("{} must have both 'inputs_1' and "
                                       "'inputs_2' as inputs".format(self.name))
        if self.in_shapes['inputs_1'][0] != self.in_shapes['inputs_2'][0]:
            raise LayerValidationError("{}: inputs_1 and inputs_2 must have "
                                       "same first dimensions but got {} and {}"
                                       "".format(self.name,
                                                 self.in_shapes['inputs_1'][0],
                                                 self.in_shapes['inputs_2'][0]))
        if self.in_shapes['inputs_1'][1] != self.in_shapes['inputs_2'][1]:
            raise LayerValidationError("{}: inputs_1 and inputs_2 must have "
                                       "same second dimensions but got {} and {}"
                                       "".format(self.name,
                                                 self.in_shapes['inputs_1'][1],
                                                 self.in_shapes['inputs_2'][1]))

    def _validate_out_shapes(self):
        """Ensure self.out_shapes are all valid.

            Raise LayerValidationError otherwise."""
        for output_name, out_shape in self.out_shapes.items():
            if output_name not in self.outputs:
                raise LayerValidationError(
                    'Invalid out_shapes. {} has no output named "{}". '
                    'Choices are: {}'.format(self.name, output_name,
                                             self.outputs))

        if self.out_shapes['default'][0] != self.in_shapes['inputs_1'][0]:
            raise LayerValidationError(
                '{}: default output must have same first dimension as inputs '
                '{} but got {}'.format(self.name, self.in_shapes['inputs_1'][0],
                                       self.out_shapes['default'][0])
            )
        if self.out_shapes['default'][1] != self.in_shapes['inputs_1'][1]:
            raise LayerValidationError(
                '{}: default output must have same second dimension as inputs '
                '{} but got {}'.format(self.name, self.in_shapes['inputs_1'][1],
                                       self.out_shapes['default'][1])
            )
        if len(self.out_shapes['default']) != 3:
            raise LayerValidationError(
                '{}: this layer sums over feature '
                'dimensions, so len(out_shape) must be 3, '
                'but shape is {}'.format(self.name,
                                         self.out_shapes['default'])
            )
        if self.out_shapes['default'][2] != 1:
            raise LayerValidationError(
                '{}: this layer sums over feature dimensions, so out_shape[2] '
                'must be 1, but got {}'.format(self.name,
                                               self.out_shapes['default'][2])
            )

    def forward_pass(self, forward_buffers, train_pass=True):
        # prepare
        _h = self.handler
        inputs_1 = forward_buffers.inputs.inputs_1
        inputs_2 = forward_buffers.inputs.inputs_2
        diff = forward_buffers.internals.squared_diff
        diff_sum = forward_buffers.outputs.default

        # calculate
        _h.subtract_tt(inputs_1, inputs_2, out=diff)
        _h.elem_mult_tt(diff, diff, out=diff)

        # reshape for summation
        t, b = diff.shape[0], diff.shape[1]
        f = _h.size(diff) / (t * b)
        diff = _h.reshape(diff, (t, b, f))

        _h.sum_t(diff, axis=2, out=diff_sum)
        _h.elem_mult_st(0.5, diff_sum, out=diff_sum)

    def backward_pass(self, forward_buffers, backward_buffers):
        # prepare
        _h = self.handler
        grad_diff_sum = backward_buffers.outputs.default
        grad_diff = backward_buffers.internals.squared_diff
        grad_inputs_1 = backward_buffers.inputs.inputs_1
        grad_inputs_2 = backward_buffers.inputs.inputs_2
        inputs_1 = forward_buffers.inputs.inputs_1
        inputs_2 = forward_buffers.inputs.inputs_2

        # grad_diff_sum has only one feature dimension due to summation,
        # so we broadcast to all feature dimensions
        _h.broadcast_features_t(grad_diff_sum, grad_diff)

        # calculate
        _h.subtract_tt(inputs_1, inputs_2, out=grad_inputs_1)
        _h.subtract_tt(inputs_2, inputs_1, out=grad_inputs_2)
        _h.elem_mult_tt(grad_diff, grad_inputs_1, grad_inputs_1)
        _h.elem_mult_tt(grad_diff, grad_inputs_2, grad_inputs_2)