#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

from collections import OrderedDict

from brainstorm.layers.base_layer import Layer
from brainstorm.structure.buffer_structure import (BufferStructure,
                                                   StructureTemplate)
from brainstorm.structure.construction import ConstructionWrapper
from brainstorm.utils import (LayerValidationError, flatten_time,
                              flatten_time_and_features)


def BinomialCrossEntropy(name=None):
    """Create a Binomial Cross Entropy Layer.

    Calculate the Binomial Cross Entropy between outputs and **binary** targets

    Cross entropy is by definition asymmetric, therefore the inputs are named
    'default' for the network outputs and 'targets' for the binary targets.
    This layer only calculates the deltas for the default inputs.
    Also note that this implementation only works for **binary** targets and
    outputs in the range 0 to 1.
    For outputs outside that range or non-binary targets the result is
    undefined.
    """
    return ConstructionWrapper.create(BinomialCrossEntropyLayerImpl,
                                      name=name)


class BinomialCrossEntropyLayerImpl(Layer):

    expected_inputs = {'default': StructureTemplate('T', 'B', '...'),
                       'targets': StructureTemplate('T', 'B', '...')}

    expected_kwargs = {}

    computes_no_input_deltas_for = ['targets']

    def setup(self, kwargs, in_shapes):
        if in_shapes['default'] != in_shapes['targets']:
            raise LayerValidationError("{}: default and targets must have the "
                                       "same shapes but got {} and {}"
                                       .format(self.name,
                                               in_shapes['default'],
                                               in_shapes['targets']))
        outputs = OrderedDict()
        outputs['default'] = BufferStructure('T', 'B', 1)

        feature_shape = in_shapes['default'].feature_shape
        internals = OrderedDict()
        internals['cee'] = BufferStructure('T', 'B', *feature_shape)
        internals['ceed'] = BufferStructure('T', 'B', *feature_shape,
                                            is_backward_only=True)

        return outputs, OrderedDict(), internals

    def forward_pass(self, buffers, training_pass=True):
        # prepare
        _h = self.handler
        y = buffers.inputs.default
        t = buffers.inputs.targets
        cee = buffers.internals.cee
        cee_sum = buffers.outputs.default

        # the binomial cross entropy error is given by
        # - t * ln(y) - (1-t) * ln(1-y)
        tmp = _h.ones(cee.shape)
        _h.subtract_tt(tmp, y, cee)     # cee = 1-y
        _h.subtract_tt(tmp, t, tmp)     # tmp  = 1-t
        _h.clip_t(cee, 1e-6, 1.0, cee)
        _h.log_t(cee, cee)              # cee = ln(1-y)
        _h.mult_tt(tmp, cee, tmp)  # tmp = (1-t) * ln(1-y)

        _h.clip_t(y, 1e-6, 1.0, cee)
        _h.log_t(cee, cee)              # cee = ln(y)
        _h.mult_tt(t, cee, cee)    # cee = t * ln(y)

        _h.add_tt(tmp, cee, cee)        # cee = (1-t) * ln(1-y) + t * ln(y)

        # reshape for summation
        cee = flatten_time_and_features(cee)
        cee_sum = flatten_time(cee_sum)
        _h.sum_t(cee, axis=1, out=cee_sum)
        _h.mult_st(-1, cee_sum, cee_sum)  # * -1

    def backward_pass(self, buffers):
        # prepare
        _h = self.handler
        ceed_sum = buffers.output_deltas.default
        ceed = buffers.internals.ceed
        tmp = _h.allocate(ceed.shape)

        y = buffers.inputs.default
        t = buffers.inputs.targets

        yd = buffers.input_deltas.default

        # the derivative of the binomial cross entropy error is given by
        # (y - t) / (y - y²)

        _h.mult_tt(y, y, ceed)       # ceed = y²
        _h.subtract_tt(y, ceed, ceed)     # ceed = y - y²
        _h.clip_t(ceed, 1e-6, 1.0, ceed)  # clip

        _h.subtract_tt(y, t, tmp)         # tmp = y - t

        _h.divide_tt(tmp, ceed, ceed)     # ceed = (y - t) / (y - y²)

        # ceed_sum has only one feature dimension due to summation,
        # so we broadcast to all feature dimensions
        _h.broadcast_t(ceed_sum, 2, tmp)
        _h.mult_tt(ceed, tmp, ceed)

        _h.add_tt(ceed, yd, yd)
