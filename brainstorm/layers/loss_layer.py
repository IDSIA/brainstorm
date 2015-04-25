#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from brainstorm.layers.base_layer import LayerBaseImpl
from brainstorm.structure.shapes import ShapeTemplate


class LossLayerImpl(LayerBaseImpl):
    # TODO: handle masks and batch/sequence normalization
    # TODO: add importance factor
    inputs = {'default': ShapeTemplate('...')}
    outputs = {'loss': ShapeTemplate(1)}
    expected_kwargs = {}

    def forward_pass(self, forward_buffer, training_pass=True):
        # TODO: passing axis=None works with numpy an pycuda
        # TODO: but is this the intended interface?
        self.handler.sum_t(forward_buffer.inputs.default,
                           None,
                           forward_buffer.outputs.loss.reshape(tuple()))

    def backward_pass(self, forward_buffers, backward_buffers):
        self.handler.add_st(1.0, backward_buffers.inputs.default,
                            out=backward_buffers.inputs.default)

    def _get_output_shapes(self):
        return {'loss': ShapeTemplate(1)}
