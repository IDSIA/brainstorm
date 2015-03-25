#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from brainstorm.layers.base_layer import LayerBase
from brainstorm.utils import LayerValidationError


class NoOpLayer(LayerBase):
    """
    This layer just copies its input into its output.
    """

    def _validate_out_shapes(self):
        if self.out_shapes != self.in_shapes:
            raise LayerValidationError(
                "For {} (NoOpLayer) in_ and out_shapes must be equal, "
                "but {} != {}".format(self.name, self.in_shapes['default'],
                                      self.out_shapes['default']))

    def forward_pass(self, forward_buffers, train_pass=True):
        self.handler.copy_to(forward_buffers.inputs.default,
                             forward_buffers.outputs.default)

    def backward_pass(self, forward_buffers, backward_buffers):
        self.handler.add(backward_buffers.outputs.default,
                         backward_buffers.inputs.default,
                         out=backward_buffers.inputs.default)