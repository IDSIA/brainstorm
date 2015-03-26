#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from brainstorm.layers.base_layer import LayerBase


class LossLayer(LayerBase):
    # TODO: handle masks and batch/sequence normalization
    expected_kwargs = {}
    inputs = {'default': ('T', 'B', 'F')}

    outputs = {'loss': (1,)}

    def forward_pass(self, forward_buffer, training_pass=True):
        self.handler.sum_t(forward_buffer.inputs.default,
                           None,
                           forward_buffer.outputs.loss.reshape(tuple()))

    def backward_pass(self, forward_buffers, backward_buffers):
        self.handler.fill(backward_buffers.inputs.default, 1.0)

    def _get_output_shapes(self):
            return {'loss': (1,)}
