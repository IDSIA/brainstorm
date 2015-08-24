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

    def forward_pass(self, buffers, training_pass=True):
        time_size, batch_size = buffers.inputs.default.shape[:2]
        f_size = np.prod(buffers.inputs.default.shape[2:])
        inp = buffers.inputs.default.reshape(time_size * batch_size, f_size)
        self.handler.mult_mv(inp, buffers.inputs.mask.reshape(time_size * batch_size, 1),
                             buffers.outputs.default.reshape(time_size * batch_size, f_size))

    def backward_pass(self, buffers):
        time_size, batch_size = buffers.inputs.default.shape[:2]
        f_size = np.prod(buffers.inputs.default.shape[2:])
        flat_shape = (time_size * batch_size, f_size)
        out_deltas = buffers.output_deltas.default.reshape(flat_shape)
        tmp = self.handler.allocate(flat_shape)
        self.handler.mult_mv(out_deltas, buffers.inputs.mask.reshape(time_size * batch_size, 1),tmp)
        in_deltas = buffers.input_deltas.default.reshape(flat_shape)
        self.handler.add_tt(tmp, in_deltas, in_deltas)
