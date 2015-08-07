#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np
from brainstorm.layers.base_layer import LayerBaseImpl
from brainstorm.structure.shapes import ShapeTemplate


class MaskLayerImpl(LayerBaseImpl):

    inputs = {'default': ShapeTemplate('T', 'B', '...'),
              'mask': ShapeTemplate('T', 'B', 1)}
    outputs = {'default': ShapeTemplate('T', 'B', '...')}

    def _get_output_shapes(self):
        return {'default': self.in_shapes['default']}

    def forward_pass(self, forward_buffers, training_pass=True):
        time_size, batch_size = forward_buffers.inputs.default.shape[:2]
        f_size = np.prod(forward_buffers.inputs.default.shape[2:])
        inp = forward_buffers.inputs.default.reshape(time_size * batch_size, f_size)
        self.handler.mult_mv(inp, forward_buffers.inputs.mask.reshape(time_size * batch_size, 1),
                             forward_buffers.outputs.default.reshape(time_size * batch_size, f_size))

    def backward_pass(self, forward_buffers, backward_buffers):
        time_size, batch_size = forward_buffers.inputs.default.shape[:2]
        f_size = np.prod(forward_buffers.inputs.default.shape[2:])
        flat_shape = (time_size * batch_size, f_size)
        out_deltas = backward_buffers.outputs.default.reshape(flat_shape)
        tmp = self.handler.allocate(flat_shape)
        self.handler.mult_mv(out_deltas, forward_buffers.inputs.mask.reshape(time_size * batch_size, 1),tmp)
        in_deltas = backward_buffers.inputs.default.reshape(flat_shape)
        self.handler.add_tt(tmp, in_deltas, in_deltas)
