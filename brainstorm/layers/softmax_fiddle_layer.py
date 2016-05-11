#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

from collections import OrderedDict

from brainstorm.layers.base_layer import Layer
from brainstorm.structure.buffer_structure import (BufferStructure,
                                                   StructureTemplate)
from brainstorm.structure.construction import ConstructionWrapper
from brainstorm.utils import (LayerValidationError, flatten_all_but_last, flatten_time_and_features)


def SoftmaxFiddle(name=None):
    """Create a softmax layer with integrated Multinomial Cross Entropy loss.

    Applies the softmax activation function on 'default' input and puts
    results (per-class probabilities) in 'predictions'.

    It also takes 'targets' as input (typically a one-hot vector),
    and computes the multinomial cross-entropy loss. The resulting losses are
    stored in the 'loss' output.

    For pixel/voxel-wise classification, the `channel` dimension must be
    right-most (known as NHWC or NDHWC format).

    WARNING:
        This layer does not compute derivatives wrt the 'targets' input.
        It also does not use the deltas coming in from the 'predictions'.
    """
    return ConstructionWrapper.create(SoftmaxFiddleLayerImpl, name=name)


class SoftmaxFiddleLayerImpl(Layer):

    expected_inputs = {'default': StructureTemplate('T', 'B', '...'),
                       'targets': StructureTemplate('T', 'B', '...')}

    computes_no_input_deltas_for = ['targets']
    takes_no_output_deltas_from = ['predictions']

    def setup(self, kwargs, in_shapes):
        in_shape = in_shapes['default'].feature_shape
        tar_shape = in_shapes['targets'].feature_shape

        if len(tar_shape) != len(in_shape):
            raise LayerValidationError('Default input and targets must have '
                                       'the same number of dimensions.')
        if tar_shape != in_shape:
            raise LayerValidationError('All dimensions must match '
                                       'for default input and targets.')

        outputs = OrderedDict()
        outputs['predictions'] = BufferStructure('T', 'B', *in_shape)
        outputs['loss'] = BufferStructure('T', 'B', *in_shape)

        internals = OrderedDict()
        internals['dcee'] = BufferStructure('T', 'B', *in_shape,
                                            is_backward_only=True)
        return outputs, OrderedDict(), internals

    def forward_pass(self, buffers, training_pass=True):
        # prepare
        _h = self.handler
        inputs = buffers.inputs.default
        targets = buffers.inputs.targets
        predictions = buffers.outputs.predictions
        loss = buffers.outputs.loss

        # reshape
        flat_inputs = flatten_all_but_last(inputs)
        flat_probs = flatten_all_but_last(predictions)
        flat_loss = flatten_all_but_last(loss)
        flat_targets = flatten_all_but_last(targets)

        # softmax
        _h.softmax_m(flat_inputs, flat_probs)

        # the multinomial cross entropy error is given by
        # - sum over i: p_i * ln(y_i)
        _h.copy_to(flat_probs, flat_loss)
        _h.clip_t(flat_loss, 1e-6, 1.0, flat_loss)
        _h.log_t(flat_loss, flat_loss)
        _h.mult_tt(flat_loss, flat_targets, flat_loss)
        _h.mult_st(-1, loss, loss)

    def backward_pass(self, buffers):
        # prepare
        _h = self.handler
        targets = flatten_time_and_features(buffers.inputs.targets)
        probs = flatten_time_and_features(buffers.outputs.predictions)

        dinputs = flatten_time_and_features(buffers.input_deltas.default)
        dloss = flatten_time_and_features(buffers.output_deltas.loss)

        dcee = flatten_time_and_features(buffers.internals.dcee)

        # derivative of multinomial cross-entropy error wrt softmax:
        # y - t

        _h.subtract_tt(probs, targets, dcee)  # y - t
        _h.mult_mv(dcee, dloss, dcee)  # out_delta * (y - t)
        _h.add_tt(dcee, dinputs, dinputs)
