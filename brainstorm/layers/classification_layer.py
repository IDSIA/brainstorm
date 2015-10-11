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


def Classification(size, name=None):
    """Create a softmax layer with integrated Multinomial Loss.

    Operates like a FullyConnectedLayer with softmax activation function
    on 'default' input and puts results (per-class probabilities) in
    'probabilities'.

    It also takes class numbers as the 'targets' input, and computes the
    multinomial cross-entropy loss. The resulting losses are stored in the
    'loss' output.

    WARNING:
        This layer does not compute derivatives wrt the 'targets' input.
        It also does not use the deltas coming in from the 'probabilities'.
    """
    return ConstructionWrapper.create(ClassificationLayerImpl, size=size,
                                      name=name)


class ClassificationLayerImpl(Layer):

    expected_inputs = {'default': StructureTemplate('T', 'B', '...'),
                       'targets': StructureTemplate('T', 'B', 1)}
    expected_kwargs = {'size'}

    def setup(self, kwargs, in_shapes):
        in_size = in_shapes['default'].feature_size
        self.size = kwargs.get('size', in_size)

        if not (isinstance(self.size, int) and self.size > 0):
            raise LayerValidationError('Size must be a positive integer, '
                                       'but was {}'.format(self.size))

        outputs = OrderedDict()
        outputs['probabilities'] = BufferStructure('T', 'B', self.size)
        outputs['loss'] = BufferStructure('T', 'B', 1)

        parameters = OrderedDict()
        parameters['W'] = BufferStructure(self.size, in_size)
        parameters['bias'] = BufferStructure(self.size)

        internals = OrderedDict()
        internals['Ha'] = BufferStructure('T', 'B', self.size)
        internals['dHa'] = BufferStructure('T', 'B', self.size,
                                           is_backward_only=True)
        return outputs, parameters, internals

    def forward_pass(self, buffers, training_pass=True):
        # prepare
        _h = self.handler
        W, bias = buffers.parameters
        inputs = buffers.inputs.default
        targets = buffers.inputs.targets
        probabilities = buffers.outputs.probabilities
        loss = buffers.outputs.loss
        Ha = buffers.internals.Ha

        # reshape
        flat_input = flatten_time_and_features(inputs)
        flat_probs = flatten_time(probabilities)
        flat_Ha = flatten_time(Ha)
        flat_loss = flatten_time(loss)
        flat_targets = flatten_time(targets)

        # calculate activation
        _h.dot_mm(flat_input, W, flat_Ha, transb=True)
        _h.add_mv(flat_Ha, bias.reshape((1, bias.shape[0])), flat_Ha)

        # softmax
        _h.softmax_m(flat_Ha, flat_probs)

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
        W, bias = buffers.parameters
        inputs = buffers.inputs.default
        targets = buffers.inputs.targets
        probs = buffers.outputs.probabilities

        dW, dbias = buffers.gradients
        dinputs = buffers.input_deltas.default
        dloss = buffers.output_deltas.loss
        dHa = buffers.internals.dHa

        # reshape
        flat_inputs = flatten_time_and_features(inputs)
        flat_probs = flatten_time(probs)
        flat_targets = flatten_time(targets)
        flat_dHa = flatten_time(dHa)
        flat_dloss = flatten_time(dloss)
        flat_dinputs = flatten_time_and_features(dinputs)

        # derivative of multinomial cross-entropy error wrt softmax:
        # y - t
        _h.binarize_v(flat_targets, flat_dHa)
        _h.mult_st(-1, flat_dHa, flat_dHa)
        _h.add_tt(flat_dHa, flat_probs, flat_dHa)
        _h.mult_mv(flat_dHa, flat_dloss, flat_dHa)

        # calculate in_deltas and gradients
        _h.dot_add_mm(flat_dHa, W, out=flat_dinputs)
        _h.dot_mm(flat_dHa, flat_inputs, dW, transa=True)
        _h.sum_t(flat_dHa, axis=0, out=dbias)
