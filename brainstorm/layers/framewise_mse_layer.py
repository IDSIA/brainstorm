#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from brainstorm.layers.base_layer import LayerBase
from brainstorm.utils import LayerValidationError


class FramewiseMSELayer(LayerBase):
    inputs = {'outputs': ('T', 'B', 'F'),
              'targets': ('T', 'B', 'F'),
              'masks': ('T', 'B', 1)
              }
    """Names and shape-templates for all inputs of this layer"""

    outputs = {'errors': ('T', 'B', '1')}
    """Names and shape-templates for all outputs of this layer"""

    expected_kwargs = {}
    """Set of all kwargs that this layer accepts"""

    def _get_output_shapes(self):
        """
        Sets the shape of the 'errors' output using in_shapes['outputs']
        """
        return {'errors': (
            self.in_shapes['outputs'][0], self.in_shapes['outputs'][1], 1)}

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

        # 'outputs' and 'targets' must be wired in and match first 2 dimensions
        if 'outputs' not in self.in_shapes or 'targets' not in self.in_shapes:
            raise LayerValidationError("{} must have both 'outputs' and "
                                       "'targets' as inputs".format(self.name))
        if self.in_shapes['outputs'][0] != self.in_shapes['targets'][0]:
            raise LayerValidationError("{}: outputs and targets must have "
                                       "same first dimensions but got {} and {}"
                                       "".format(self.name,
                                                 self.in_shapes['outputs'][0],
                                                 self.in_shapes['targets'][0]))
        if self.in_shapes['outputs'][1] != self.in_shapes['targets'][1]:
            raise LayerValidationError("{}: outputs and targets must have "
                                       "same second dimensions but got {} and {}"
                                       "".format(self.name,
                                                 self.in_shapes['outputs'][1],
                                                 self.in_shapes['outputs'][1]))
        # 'masks' is optional input, but if present must be 3D and match targets
        if 'masks' in self.in_shapes:
            if len(self.in_shapes['masks']) != 3:
                raise LayerValidationError(
                    "{}: masks must be 3-dimensional but got {"
                    "}-dimensional".format(self.name,
                                           len(self.in_shapes['masks'])))
            if self.in_shapes['masks'][0] != self.in_shapes['targets'][0]:
                raise LayerValidationError(
                    "{}: masks and targets must have same first dimensions "
                    "but got {} and {}".format(self.name,
                                               self.in_shapes['masks'][0],
                                               self.in_shapes['targets'][0])
                )
            if self.in_shapes['masks'][1] != self.in_shapes['targets'][1]:
                raise LayerValidationError(
                    "{}: masks and targets must have same first dimensions "
                    "but got {} and {}".format(self.name,
                                               self.in_shapes['masks'][1],
                                               self.in_shapes['targets'][1])
                )
            self.masks = True

    def _validate_out_shapes(self):
        """Ensure self.out_shapes are all valid.

            Raise LayerValidationError otherwise."""
        for output_name, out_shape in self.out_shapes.items():
            if output_name not in self.outputs:
                raise LayerValidationError(
                    'Invalid out_shapes. {} has no output named "{}". '
                    'Choices are: {}'.format(self.name, output_name,
                                             self.outputs))
