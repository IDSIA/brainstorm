#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from brainstorm.layers.base_layer import LayerBase


class LossLayer(LayerBase):
    # TODO: actually implement
    expected_kwargs = {}
    inputs = {'default': ('T', 'B', 'F')}

    outputs = {'loss': (1,)}

    def forward_pass(self, forward_buffers):
        pass  # TODO: Sum over all inputs and write to outputs.loss

    def backward_pass(self, forward_buffers, backward_buffers):
        self.handler.fill(backward_buffers.inputs.default, 1.0)