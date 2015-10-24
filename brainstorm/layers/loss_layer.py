#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

from collections import OrderedDict

from brainstorm.layers.base_layer import Layer
from brainstorm.structure.buffer_structure import (BufferStructure,
                                                   StructureTemplate)
from brainstorm.structure.construction import ConstructionWrapper


def Loss(importance=1.0, name=None):
    """Create a Loss layer."""
    return ConstructionWrapper.create(LossLayerImpl, importance=importance,
                                      name=name)


class LossLayerImpl(Layer):

    expected_inputs = {'default': StructureTemplate('...')}
    expected_kwargs = {'importance'}

    def setup(self, kwargs, in_shapes):
        assert self.name != 'total_loss'

        self.importance = kwargs.get('importance', 1.0)
        self.batch_index = None
        if in_shapes['default'].scales_with_time:
            self.batch_index = 1
        elif in_shapes['default'].scales_with_batch_size:
            self.batch_index = 0

        outputs = OrderedDict()
        outputs['loss'] = BufferStructure(1)
        return outputs, OrderedDict(), OrderedDict()

    def forward_pass(self, buffers, training_pass=True):
        if self.batch_index is None:
            batch_size = 1.0
        else:
            batch_size = buffers.inputs.default.shape[self.batch_index]

        self.handler.sum_t(buffers.inputs.default,
                           None,
                           buffers.outputs.loss.reshape(tuple()))
        self.handler.mult_st(self.importance / batch_size,
                             buffers.outputs.loss,
                             buffers.outputs.loss)

    def backward_pass(self, buffers):
        if self.batch_index is None:
            batch_size = 1.0
        else:
            batch_size = buffers.inputs.default.shape[self.batch_index]
        self.handler.add_st(self.importance / batch_size,
                            buffers.input_deltas.default,
                            buffers.input_deltas.default)
