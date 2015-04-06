#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from brainstorm.utils import LayerValidationError
from brainstorm.layers.base_layer import LayerBaseImpl


class LossLayerImpl(LayerBaseImpl):
    # TODO: handle masks and batch/sequence normalization
    # TODO: add importance factor
    expected_kwargs = {}
    inputs = {'default': ('...',)}

    outputs = {'loss': (1,)}

    def forward_pass(self, forward_buffer, training_pass=True):
        # TODO: passing axis=None works with numpy an pycuda
        # TODO: but is this the intended interface?
        self.handler.sum_t(forward_buffer.inputs.default,
                           None,
                           forward_buffer.outputs.loss.reshape(tuple()))

    def backward_pass(self, forward_buffers, backward_buffers):
        self.handler.fill(backward_buffers.inputs.default, 1.0)

    def _validate_in_shapes(self):
        """Ensure self.in_shapes are all valid.

         Raise LayerValidationError otherwise."""
        for input_name, in_shape in self.in_shapes.items():
            if input_name not in self.inputs:
                raise LayerValidationError(
                    'Invalid in_shapes. {} has no input named "{}". '
                    'Choices are: {}'.format(self.name, input_name,
                                             self.inputs))

    def _get_output_shapes(self):
            return {'loss': (1,)}
