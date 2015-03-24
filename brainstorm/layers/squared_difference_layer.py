#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from brainstorm.layers.base_layer import LayerBase
from brainstorm.utils import LayerValidationError


class SquaredDifferenceLayer(LayerBase):
    inputs = {'inputs_1': ('T', 'B', 'F'),
              'inputs_2': ('T', 'B', 'F')
              }
    """Names and shape-templates for all inputs of this layer"""

    outputs = {'default': ('T', 'B', 'F')}
    """Names and shape-templates for all outputs of this layer"""

    expected_kwargs = {}
    """Set of all kwargs that this layer accepts"""

    def _get_output_shapes(self):
        """
        Sets the shape of the 'default' output using in_shapes['inputs_1']
        """
        return {'default': self.in_shapes['inputs_1']}

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

        # 'inputs_1' and 'inputs_2' must be wired in and match first 2 dimensions
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

        if self.out_shapes['default'] != self.in_shapes['inputs_1']:
            LayerValidationError(
                '{}: shape of default output {} must match shape of both '
                'inputs {}'.format(self.name, self.out_shapes['default'],
                                   self.in_shapes['inputs_1'])
            )

    def forward_pass(self, forward_buffers):
        # prepare
        _h = self.handler
        inputs_1 = forward_buffers.inputs.inputs_1
        inputs_2 = forward_buffers.inputs.inputs_2
        diff = forward_buffers.outputs.default

        # calculate
        _h.subtract_tt(inputs_1, inputs_2, out=diff)
        _h.elem_mult_tt(diff, diff, out=diff)
        _h.elem_mult_st(0.5, diff, out=diff)

    def backward_pass(self, forward_buffers, backward_buffers):
        # prepare
        _h = self.handler
        grad_diff = backward_buffers.outputs.default
        grad_inputs_1 = backward_buffers.inputs.inputs_1
        grad_inputs_2 = backward_buffers.inputs.inputs_2
        inputs_1 = forward_buffers.inputs.inputs_1
        inputs_2 = forward_buffers.inputs.inputs_2

        _h.subtract_tt(inputs_1, inputs_2, out=grad_inputs_1)
        _h.subtract_tt(inputs_2, inputs_1, out=grad_inputs_2)
        _h.elem_mult_tt(grad_diff, grad_inputs_1, grad_inputs_1)
        _h.elem_mult_tt(grad_diff, grad_inputs_2, grad_inputs_2)