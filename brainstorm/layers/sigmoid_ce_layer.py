#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

from collections import OrderedDict

from brainstorm.handlers.base_handler import Handler
from brainstorm.layers.base_layer import Layer
from brainstorm.structure.buffer_structure import (BufferStructure,
                                                   StructureTemplate)
from brainstorm.structure.construction import ConstructionWrapper
from brainstorm.utils import (LayerValidationError, flatten_time_and_features,
                              flatten_time)


def SigmoidCE(name=None):
    """Create a sigmoid layer with integrated Binomial Cross Entropy loss.

    Applies the sigmoid activation function on 'default' input and puts the
    results (per-label probabilities) in 'probabilities'.

    It also takes as 'targets' a binary vector and computes the binomial
    cross-entropy loss. The resulting losses are stored in the 'loss' output.

    WARNING:
        This layer does not compute derivatives wrt the 'targets' input.
        It also does not use the deltas coming in from the 'probabilities'.
    """
    return ConstructionWrapper.create(SigmoidCELayerImpl, name=name)


class SigmoidCELayerImpl(Layer):

    expected_inputs = {'default': StructureTemplate('T', 'B', '...'),
                       'targets': StructureTemplate('T', 'B', '...')}

    def setup(self, kwargs, in_shapes):
        in_shape = in_shapes['default'].feature_shape
        tar_shape = in_shapes['targets'].feature_shape

        if tar_shape != in_shape:
            raise LayerValidationError('input and targets must have the same '
                                       'shapes. But got {} != {}'
                                       .format(in_shape, tar_shape))

        outputs = OrderedDict()
        outputs['probabilities'] = BufferStructure('T', 'B', *in_shape)
        outputs['loss'] = BufferStructure('T', 'B', 1)

        internals = OrderedDict()
        internals['cee'] = BufferStructure('T', 'B', *in_shape)
        internals['dcee'] = BufferStructure('T', 'B', *in_shape,
                                            is_backward_only=True)
        return outputs, OrderedDict(), internals

    def forward_pass(self, buffers, training_pass=True):
        _h = self.handler
        assert isinstance(_h, Handler)

        inputs = flatten_time_and_features(buffers.inputs.default)
        targets = flatten_time_and_features(buffers.inputs.targets)
        cee = flatten_time_and_features(buffers.internals.cee)
        loss = flatten_time_and_features(buffers.outputs.loss)
        prob = flatten_time_and_features(buffers.outputs.probabilities)

        # Apply sigmoid
        _h.sigmoid(inputs, prob)

        # the binomial cross entropy error is given by
        # - (t * ln(y) + (1-t) * ln(1-y))
        tmp = _h.ones(prob.shape)
        _h.subtract_tt(tmp, prob, cee)     # cee = 1-y
        _h.subtract_tt(tmp, targets, tmp)     # tmp  = 1-t
        _h.clip_t(cee, 1e-6, 1.0, cee)
        _h.log_t(cee, cee)              # cee = ln(1-y)
        _h.mult_tt(tmp, cee, tmp)  # tmp = (1-t) * ln(1-y)

        _h.clip_t(prob, 1e-6, 1.0, cee)
        _h.log_t(cee, cee)              # cee = ln(y)
        _h.mult_tt(targets, cee, cee)    # cee = t * ln(y)

        _h.add_tt(tmp, cee, cee)        # cee = (1-t) * ln(1-y) + t * ln(y)

        # reshape for summation
        _h.sum_t(cee, axis=1, out=loss)
        _h.mult_st(-1, loss, loss)  # * -1

    def backward_pass(self, buffers):
        # prepare
        _h = self.handler
        assert isinstance(_h, Handler)

        dinputs = flatten_time_and_features(buffers.input_deltas.default)
        dloss = flatten_time(buffers.output_deltas.loss)
        dcee = flatten_time_and_features(buffers.internals.cee)
        targets = flatten_time_and_features(buffers.inputs.targets)
        prob = flatten_time_and_features(buffers.outputs.probabilities)

        _h.subtract_tt(prob, targets, dcee)  # y - t
        _h.mult_mv(dcee, dloss, dcee)        # out_delta * (y - t)
        _h.add_tt(dcee, dinputs, dinputs)
