#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np
from brainstorm.layers.base_layer import LayerBaseImpl
from brainstorm.structure.shapes import ShapeTemplate
from brainstorm.utils import flatten_time_and_features, flatten_time


class MaskLayerImpl(LayerBaseImpl):

    inputs = {'default': ShapeTemplate('T', 'B', '...'),
              'mask': ShapeTemplate('T', 'B', 1)}
    outputs = {'default': ShapeTemplate('T', 'B', '...')}

    def _get_output_shapes(self):
        return {'default': self.in_shapes['default']}

    def forward_pass(self, buffers, training_pass=True):
        _h = self.handler

        flat_inp = flatten_time_and_features(buffers.inputs.default)
        flat_mask = flatten_time(buffers.inputs.mask)
        flat_out = flatten_time_and_features(buffers.outputs.default)

        _h.mult_mv(flat_inp, flat_mask, out=flat_out)

    def backward_pass(self, buffers):
        _h = self.handler

        flat_out_deltas = flatten_time_and_features(
            buffers.output_deltas.default)
        tmp = self.handler.allocate(flat_out_deltas.shape)
        flat_mask = flatten_time(buffers.inputs.mask)
        flat_in_deltas = flatten_time_and_features(
            buffers.input_deltas.default)

        _h.mult_mv(flat_out_deltas, flat_mask, tmp)
        _h.add_tt(tmp, flat_in_deltas, flat_in_deltas)
