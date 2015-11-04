#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

from collections import OrderedDict

from brainstorm.layers.base_layer import Layer
from brainstorm.structure.buffer_structure import (BufferStructure,
                                                   StructureTemplate)
from brainstorm.structure.construction import ConstructionWrapper
from brainstorm.utils import (LayerValidationError, flatten_all_but_last)


def SoftmaxCE(name=None):
    """Create a softmax layer with integrated Multinomial Cross Entropy loss.

    Applies the softmax activation function on 'default' input and puts
    results (per-class probabilities) in 'predictions'.

    It also takes class indices (0-based) as the 'targets' input,
    and computes the multinomial cross-entropy loss. The resulting losses are
    stored in the 'loss' output.

    For pixel/voxel-wise classification, the `channel` dimension must be
    right-most (known as NHWC or NDHWC format).

    WARNING:
        This layer does not compute derivatives wrt the 'targets' input.
        It also does not use the deltas coming in from the 'predictions'.
    """
    return ConstructionWrapper.create(SoftmaxCELayerImpl, name=name)


class SoftmaxCELayerImpl(Layer):

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
        if tar_shape[:-1] != in_shape[:-1]:
            raise LayerValidationError('All dimensions except last must match '
                                       'for default input and targets.')
        if tar_shape[-1] != 1:
            raise LayerValidationError('Last dimension of targets must be '
                                       'size 1.')

        outputs = OrderedDict()
        outputs['predictions'] = BufferStructure('T', 'B', *in_shape)
        outputs['loss'] = BufferStructure('T', 'B', *tar_shape)

        internals = OrderedDict()
        internals['t_bin'] = BufferStructure('T', 'B', *in_shape,
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
        # now our targets are indices so all p_i = 0 except for i=t
        _h.fill(loss, 0.)
        _h.index_m_by_v(flat_probs, flat_targets, flat_loss)
        _h.clip_t(flat_loss, 1e-6, 1.0, flat_loss)
        _h.log_t(loss, loss)
        _h.mult_st(-1, loss, loss)

    def backward_pass(self, buffers):
        # prepare
        _h = self.handler
        targets = buffers.inputs.targets
        probs = buffers.outputs.predictions

        dinputs = buffers.input_deltas.default
        dloss = buffers.output_deltas.loss
        t_bin = buffers.internals.t_bin

        # reshape
        flat_probs = flatten_all_but_last(probs)
        flat_targets = flatten_all_but_last(targets)
        flat_t_bin = flatten_all_but_last(t_bin)
        flat_dloss = flatten_all_but_last(dloss)
        flat_dinputs = flatten_all_but_last(dinputs)

        # derivative of multinomial cross-entropy error wrt softmax:
        # y - t
        _h.binarize_v(flat_targets, flat_t_bin)
        _h.mult_st(-1, flat_t_bin, flat_t_bin)
        _h.add_tt(flat_t_bin, flat_probs, flat_t_bin)
        _h.mult_mv(flat_t_bin, flat_dloss, flat_t_bin)
        _h.add_tt(flat_t_bin, flat_dinputs, flat_dinputs)
