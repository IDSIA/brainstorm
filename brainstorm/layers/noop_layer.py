#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from brainstorm.layers.base_layer import LayerBaseImpl


class NoOpLayerImpl(LayerBaseImpl):
    """
    This layer just copies its input into its output.
    """
    expected_kwargs = {}

    def _get_output_shapes(self):
        return self.in_shapes

    def forward_pass(self, forward_buffers, training_pass=True):
        self.handler.copy_to(forward_buffers.outputs.default,
                             forward_buffers.inputs.default)

    def backward_pass(self, forward_buffers, backward_buffers):
        self.handler.add_tt(backward_buffers.outputs.default,
                            backward_buffers.inputs.default,
                            out=backward_buffers.inputs.default)
