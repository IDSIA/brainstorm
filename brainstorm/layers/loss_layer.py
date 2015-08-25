#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from brainstorm.layers.base_layer import LayerBaseImpl
from brainstorm.structure.shapes import ShapeTemplate


class LossLayerImpl(LayerBaseImpl):
    # TODO: handle masks
    inputs = {'default': ShapeTemplate('...')}
    outputs = {'loss': ShapeTemplate(1)}
    expected_kwargs = {'importance'}

    def _setup_hyperparameters(self):
        self.importance = self.kwargs.get('importance', 1.0)

    def _get_output_shapes(self):
        return {'loss': ShapeTemplate(1)}

    def forward_pass(self, buffers, training_pass=True):
        # TODO: passing axis=None works with numpy an pycuda
        # TODO: but is this the intended interface?
        time_size, batch_size = buffers.inputs.default.shape[:2]
        self.handler.sum_t(buffers.inputs.default,
                           None,
                           buffers.outputs.loss.reshape(tuple()))
        self.handler.mult_st(self.importance / batch_size,
                             buffers.outputs.loss,
                             buffers.outputs.loss)

    def backward_pass(self, buffers):
        time_size, batch_size = buffers.inputs.default.shape[:2]
        self.handler.add_st(self.importance / batch_size,
                            buffers.input_deltas.default,
                            buffers.input_deltas.default)
