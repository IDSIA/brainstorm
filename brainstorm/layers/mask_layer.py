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
        flat_size = time_size * batch_size
        feat_size = np.prod(buffers.inputs.default.shape[2:])

        flat_inp = buffers.inputs.default.reshape(flat_size, feat_size)
        flat_mask = buffers.inputs.mask.reshape(flat_size, 1)
        flat_out = buffers.outputs.default.reshape(flat_size, feat_size)

        self.handler.mult_mv(flat_inp, flat_mask, out=flat_out)

    def backward_pass(self, buffers):
        time_size, batch_size = buffers.inputs.default.shape[:2]
        feat_size = np.prod(buffers.inputs.default.shape[2:])
        flat_size = time_size * batch_size

        flat_shape = (time_size * batch_size, feat_size)
        flat_out_deltas = buffers.output_deltas.default.reshape(flat_shape)
        tmp = self.handler.allocate(flat_shape)
        flat_mask = buffers.inputs.mask.reshape(flat_size, 1)
        flat_in_deltas = buffers.input_deltas.default.reshape(flat_shape)

        self.handler.mult_mv(flat_out_deltas, flat_mask, tmp)
        self.handler.add_tt(tmp, flat_in_deltas, flat_in_deltas)
