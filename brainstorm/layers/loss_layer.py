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

    def __init__(self, name, in_shapes, incoming_connections,
                 outgoing_connections, **kwargs):
        super(LossLayerImpl, self).__init__(
            name, in_shapes, incoming_connections, outgoing_connections,
            **kwargs)
        self.importance = kwargs.get('importance', 1.0)

    def forward_pass(self, forward_buffers, training_pass=True):
        # TODO: passing axis=None works with numpy an pycuda
        # TODO: but is this the intended interface?
        time_size, batch_size = forward_buffers.inputs.default.shape[:2]
        self.handler.sum_t(forward_buffers.inputs.default,
                           None,
                           forward_buffers.outputs.loss.reshape(tuple()))
        self.handler.elem_mult_st(self.importance / batch_size,
                                  forward_buffers.outputs.loss,
                                  forward_buffers.outputs.loss)

    def backward_pass(self, forward_buffers, backward_buffers):
        time_size, batch_size = forward_buffers.inputs.default.shape[:2]
        self.handler.add_st(self.importance / batch_size, 
                            backward_buffers.inputs.default,
                            backward_buffers.inputs.default)

    def _get_output_shapes(self):
        return {'loss': ShapeTemplate(1)}
