#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from collections import OrderedDict
from brainstorm.layers.base_layer import LayerBaseImpl
from brainstorm.utils import LayerValidationError
from brainstorm.structure.shapes import ShapeTemplate


class ClassificationLayerImpl(LayerBaseImpl):
    """
    Softmax layer with integrated Multinomial-Cross-Entropy.

    Operates like a FullyConnectedLayer with softmax activation function
    on 'default' input and puts results in 'output'.

    It also takes class numbers as the 'targets' input, and computes the
    multinomial cross-entropy loss. The resulting losses are stored in the
    'loss' output.

    WARNING: This layer does not compute derivatives wrt the 'targets' input
    and it also does not use the deltas coming in from the 'outputs'.
    """

    inputs = {'default': ShapeTemplate('T', 'B', 'F'),
              'targets': ShapeTemplate('T', 'B', 1)
              }

    outputs = {'output': ShapeTemplate('T', 'B', 'F'),
               'loss': ShapeTemplate('T', 'B', 1)}

    expected_kwargs = {'size'}

    def _get_output_shapes(self):
        s = self.kwargs.get('size', self.in_shapes.get('default')[2])
        if not isinstance(s, int):
            raise LayerValidationError('size must be int but was {}'.format(s))

        return {'output': ShapeTemplate('T', 'B', s),
                'loss': ShapeTemplate('T', 'B', 1)}

    def get_internal_structure(self):
        internals = OrderedDict()
        size = self.out_shapes['output'].feature_size
        internals['Ha'] = ShapeTemplate('T', 'B', size)
        return internals

    def get_parameter_structure(self):
        in_size = self.in_shapes['default'].feature_size
        out_size = self.out_shapes['output'].feature_size

        parameters = OrderedDict()
        parameters['W'] = ShapeTemplate(out_size, in_size)
        parameters['b'] = ShapeTemplate(out_size)
        return parameters

    def forward_pass(self, forward_buffers, training_pass=True):
        # prepare
        _h = self.handler
        W, bias = forward_buffers.parameters
        inputs = forward_buffers.inputs.default
        targets = forward_buffers.inputs.targets
        output = forward_buffers.outputs.output
        loss = forward_buffers.outputs.loss
        Ha = forward_buffers.internals.Ha

        # reshape
        t, b, f = inputs.shape
        i = t * b
        flat_input = inputs.reshape((i, f))
        flat_output = output.reshape((i, self.out_shapes['output'][2]))
        flat_Ha = Ha.reshape((i, self.out_shapes['output'][2]))
        flat_loss = loss.reshape((i, 1))
        flat_targets = targets.reshape(i, 1)

        # calculate activation
        _h.dot_mm(flat_input, W, flat_Ha, transb='T')
        _h.add_mv(flat_Ha, bias, flat_Ha)

        # softmax
        _h.softmax_m(flat_Ha, flat_output)

        # the multinomial cross entropy error is given by
        # - sum over i: p_i * ln(y_i)
        # now our targets are indices so all p_i = 0 except for i=t
        _h.fill(loss, 0.)
        _h.index_m_by_v(flat_output, flat_targets, flat_loss)
        _h.log_t(loss, loss)
        _h.mult_st(-1, loss, loss)

    def backward_pass(self, forward_buffers, backward_buffers):
        # prepare
        _h = self.handler
        W, bias = forward_buffers.parameters
        inputs = forward_buffers.inputs.default
        targets = forward_buffers.inputs.targets
        output = forward_buffers.outputs.output

        dW, dbias = backward_buffers.parameters
        dinput = backward_buffers.inputs.default
        dloss = backward_buffers.outputs.loss
        dHa = backward_buffers.internals.Ha

        # reshape
        t, b, f_in = inputs.shape
        f_out = output.shape[2]
        i = t * b
        flat_input = inputs.reshape((i, f_in))
        flat_output = output.reshape((i, f_out))
        flat_targets = targets.reshape(i, 1)
        flat_dHa = dHa.reshape((i, self.out_shapes['output'][2]))
        flat_dloss = dloss.reshape((i, 1))
        flat_dinput = dinput.reshape((i, f_in))

        # derivative of multinomial cross-entropy error wrt softmax:
        # y - t
        _h.binarize_v(flat_targets, flat_dHa)
        _h.mult_st(-1, flat_dHa, flat_dHa)
        _h.add_tt(flat_dHa, flat_output, flat_dHa)
        _h.mult_mv(flat_dHa, flat_dloss, flat_dHa)

        # calculate in_deltas and gradients
        _h.dot_add_mm(flat_dHa, W, out=flat_dinput)
        _h.dot_mm(flat_dHa, flat_input, dW, transa='T')
        _h.sum_t(flat_dHa, axis=0, out=dbias)
