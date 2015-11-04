#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

from collections import OrderedDict

from brainstorm.layers.base_layer import Layer
from brainstorm.structure.buffer_structure import (BufferStructure,
                                                   StructureTemplate)
from brainstorm.structure.construction import ConstructionWrapper
from brainstorm.utils import (LayerValidationError, flatten_time_and_features)


def SquaredLoss(name=None):
    """Create a SquaredLoss layer which computes half of squared difference."""
    return ConstructionWrapper.create(SquaredLossLayerImpl, name=name)


class SquaredLossLayerImpl(Layer):

    expected_inputs = {'default': StructureTemplate('T', 'B', '...'),
                       'targets': StructureTemplate('T', 'B', '...')}
    expected_kwargs = {}
    computes_no_input_deltas_for = ['targets']
    takes_no_output_deltas_from = ['predictions']

    def setup(self, kwargs, in_shapes):
        # 'default' and 'targets' must have same shape
        in_shape = in_shapes['default'].feature_shape
        tar_shape = in_shapes['targets'].feature_shape
        if in_shape != tar_shape:
            raise LayerValidationError(
                "{}: default and targets must have same feature shapes but "
                "got {} and {}".format(self.name, in_shape, tar_shape))

        outputs = OrderedDict()
        outputs['predictions'] = BufferStructure('T', 'B', *in_shape)
        outputs['loss'] = BufferStructure('T', 'B', *in_shape)

        internals = OrderedDict()
        internals['diff'] = BufferStructure('T', 'B', *in_shape)
        return outputs, OrderedDict(), internals

    def forward_pass(self, buffers, training_pass=True):
        # prepare
        _h = self.handler
        x = flatten_time_and_features(buffers.inputs.default)
        t = flatten_time_and_features(buffers.inputs.targets)
        diff = flatten_time_and_features(buffers.internals.diff)
        y = flatten_time_and_features(buffers.outputs.predictions)
        loss = flatten_time_and_features(buffers.outputs.loss)

        # calculate
        _h.copy_to(x, y)
        _h.subtract_tt(x, t, out=diff)
        _h.mult_tt(diff, diff, out=loss)
        _h.mult_st(0.5, loss, out=loss)

    def backward_pass(self, buffers):
        # prepare
        _h = self.handler
        dloss = flatten_time_and_features(buffers.output_deltas.loss)
        diff = flatten_time_and_features(buffers.internals.diff)
        dx = flatten_time_and_features(buffers.input_deltas.default)

        # calculate
        _h.mult_add_tt(dloss, diff, dx)
