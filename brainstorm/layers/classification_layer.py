#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from brainstorm.layers.base_layer import LayerBaseImpl
from brainstorm.utils import LayerValidationError


class ClassificationLayerImpl(LayerBaseImpl):
    """
    Softmax layer with integrated Multinomial-Cross-Entropy.

    Operates like a FullyConnectedLayer with softmax activation function
    on 'default' input and puts results in 'output'.

    It also takes class numbers as the 'targets' input, and computes the
    multinomial cross-entropy loss. The resulting losses are stored in the
    'loss' output.

    WARNING: This layer does not compute derivatives wrt the 'targets' input
    and it also does not use the deltas coming in from the 'ouputs'.
    """

    inputs = {'default': ('T', 'B', 'F'),
              'targets': ('T', 'B', 1)
              }

    outputs = {'output': ('T', 'B', 'F'),
               'loss': ('T', 'B', 1)}

    expected_kwargs = {'shape'}

    def _get_output_shapes(self):
        s = self.kwargs.get('shape', self.in_shapes.get('default'))

        if isinstance(s, (tuple, list)):
            out_shape = tuple(s)
        else:
            assert isinstance(s, int), \
                "shape datatype not understood {}".format(type(s))
            out_shape = (s,)
        assert len(out_shape) == 1, \
            'Classification layer only works with 1D shape'

        return {'output': ('T', 'B') + out_shape,
                'loss': ('T', 'B', 1)}

    def get_internal_structure(self):
        feature_shape = self.in_shapes['default'][2:]
        return {
            'Ha': {
                '@shape': ('T', 'B') + feature_shape,
                '@index': 0
            }
        }

    def get_parameter_structure(self):
        return {
            'W': {
                '@shape': (self.in_shapes['default'][2],
                           self.out_shapes['output'][2]),
                '@index': 0},
            'b': {
                '@shape': (self.out_shapes['output'][2],),
                '@index': 1}
        }

    def _validate_in_shapes(self):
        super(ClassificationLayerImpl, self)._validate_in_shapes()
        # 'default' and 'targets' must be wired in
        if 'default' not in self.in_shapes or 'targets' not in self.in_shapes:
            raise LayerValidationError("{} must have both 'default' and "
                                       "'targets' as inputs".format(self.name))

    def forward_pass(self, forward_buffers, training_pass=True):
        # prepare
        _h = self.handler
        W, bias = forward_buffers.parameters
        input = forward_buffers.inputs.default
        targets = forward_buffers.inputs.targets
        output = forward_buffers.outputs.output
        loss = forward_buffers.outputs.loss
        Ha = forward_buffers.internals.Ha

        # reshape
        t, b, f = input.shape
        i = t * b
        flat_input = input.reshape((i, f))
        flat_output = output.reshape((i, self.out_shapes['output'][2]))
        flat_Ha = Ha.reshape((i, self.out_shapes['output'][2]))
        flat_loss = loss.reshape((i, 1))
        flat_targets = targets.reshape(i, 1)

        # calculate activation
        _h.dot_mm(flat_input, W, flat_Ha)
        _h.add_mv(flat_Ha, bias, flat_Ha)

        # softmax
        _h.softmax_m(flat_Ha, flat_output)

        # the multinomial cross entropy error is given by
        # - sum over i: p_i * ln(y_i)
        # now our targets are indices so all p_i = 0 except for i=t
        _h.fill(loss, 0.)
        _h.index_m_by_v(flat_Ha, flat_targets, flat_loss)
        _h.log_t(loss, loss)
        _h.elem_mult_st(-1, loss, loss)

    def backward_pass(self, forward_buffers, backward_buffers):
        # prepare
        _h = self.handler
        W, bias = forward_buffers.parameters
        input = forward_buffers.inputs.default
        targets = forward_buffers.inputs.targets
        output = forward_buffers.outputs.output

        dW, dbias = backward_buffers.parameters
        dinput = backward_buffers.inputs.default
        dloss = backward_buffers.outputs.loss
        dHa = backward_buffers.internals.Ha

        # reshape
        t, b, f = input.shape
        i = t * b
        flat_input = input.reshape((i, f))
        flat_output = output.reshape((i, f))
        flat_targets = targets.reshape(i, 1)
        flat_dHa = dHa.reshape((i, self.out_shapes['output'][2]))
        flat_dloss = dloss.reshape((i, 1))
        flat_dinput = dinput.reshape((i, f))

        # derivative of multinomial cross-entropy error wrt softmax:
        # y - t
        _h.binarize_v(flat_targets, flat_dHa)
        _h.elem_mult_st(-1, flat_dHa, flat_dHa)
        _h.add_tt(flat_dHa, flat_output, flat_dHa)
        _h.mult_mv(flat_dHa, flat_dloss, flat_dHa)

        # calculate in_deltas and gradients
        _h.dot_add_mm(flat_dHa, W, out=flat_dinput, transb='T')
        _h.dot_mm(flat_input, flat_dHa, dW, transa='T')
        _h.sum_t(flat_dHa, axis=0, out=dbias)
