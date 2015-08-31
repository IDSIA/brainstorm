#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from collections import OrderedDict
import numpy as np
from brainstorm.layers.base_layer import LayerBaseImpl
from brainstorm.structure.shapes import ShapeTemplate
from brainstorm.utils import flatten_time_and_features, flatten_time


class DropoutLayerImpl(LayerBaseImpl):
    """
    drop_prob is the probability of a unit being dropped, i.e. 0
    """
    inputs = {'default': ShapeTemplate('T', 'B', '...')}

    outputs = {'default': ShapeTemplate('T', 'B', '...')}

    expected_kwargs = {'drop_prob'}

    def _get_output_shapes(self):
        return {'default': self.in_shapes['default']}

    def _setup_hyperparameters(self):
        self.drop_prob = self.kwargs.get('drop_prob', 0.5)

    def get_internal_structure(self):
        internals = OrderedDict()
        internals['mask'] = self.in_shapes['default']
        return internals

    def forward_pass(self, buffers, training_pass=True):
        _h = self.handler

        if training_pass:
            _h.generate_probability_mask(buffers.internals.mask,
                                         1 - self.drop_prob)
            _h.mult_tt(buffers.inputs.default, buffers.internals.mask,
                       out=buffers.outputs.default)
            _h.mult_st(1 / (1 - self.drop_prob), buffers.outputs.default,
                       out=buffers.outputs.default)
        else:
            _h.copy_to(buffers.outputs.default, buffers.inputs.default)

    def backward_pass(self, buffers):
        self.handler.mult_add_tt(buffers.output_deltas.default,
                                 buffers.internals.mask,
                                 buffers.input_deltas.default)
