#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from collections import OrderedDict
from brainstorm.utils import LayerValidationError
from brainstorm.layers.base_layer import LayerBaseImpl
from brainstorm.structure.shapes import ShapeTemplate


class RnnLayerImpl(LayerBaseImpl):
    expected_kwargs = {'size', 'activation_function'}

    def _setup_hyperparameters(self):
        self.act_func = None
        self.act_func_deriv = None
        self.size = self.kwargs.get('size',
                                    self.in_shapes['default'].feature_size)
        if not isinstance(self.size, int):
            raise LayerValidationError('size must be int but was {}'.
                                       format(self.size))

    def set_handler(self, new_handler):
        super(RnnLayerImpl, self).set_handler(new_handler)

        # Assign act_func and act_dunc_derivs
        activation_functions = {
            'sigmoid': (self.handler.sigmoid, self.handler.sigmoid_deriv),
            'tanh': (self.handler.tanh, self.handler.tanh_deriv),
            'linear': (lambda x, y: self.handler.copy_to(y, x),
                       lambda x, y, dy, dx: self.handler.copy_to(dx, dy)),
            'rel': (self.handler.rel, self.handler.rel_deriv)
        }

        self.act_func, self.act_func_deriv = activation_functions[
            self.kwargs.get('activation_function', 'linear')]

    def get_parameter_structure(self):
        in_size = self.in_shapes['default'].feature_size
        parameters = OrderedDict()
        parameters['W'] = ShapeTemplate(in_size, self.size)
        parameters['R'] = ShapeTemplate(self.size, self.size)
        parameters['bias'] = ShapeTemplate(self.size)
        return parameters

    def get_internal_structure(self):
        internals = OrderedDict()
        internals['Ha'] = ShapeTemplate('T', 'B', self.size)
        return internals

    def _get_output_shapes(self):
        return {'default': ShapeTemplate('T', 'B', self.size, context_size=1)}

    def forward_pass(self, forward_buffers, training_pass=True):
        # prepare
        _h = self.handler
        W, R, bias = forward_buffers.parameters
        inputs = forward_buffers.inputs.default
        outputs = forward_buffers.outputs.default
        Ha = forward_buffers.internals.Ha

        t, b, f = inputs.shape
        i = t * b
        flat_inputs = inputs.reshape((i, f))
        flat_Ha = Ha.reshape((i, Ha.shape[2]))

        _h.dot_mm(flat_inputs, W, flat_Ha)

        for t in range(inputs.shape[0]):
            # calculate outputs
            # outputs has a time offset of 1
            _h.dot_add_mm(outputs[t], R, Ha[t])
            _h.add_mv(Ha[t], bias, Ha[t])
            self.act_func(Ha[t], outputs[t + 1])

    def backward_pass(self, forward_buffers, backward_buffers):
        # prepare
        _h = self.handler
        W, R, bias = forward_buffers.parameters
        dW, dR, dbias = backward_buffers.parameters
        inputs = forward_buffers.inputs.default
        outputs = forward_buffers.outputs.default
        dinputs = backward_buffers.inputs.default
        doutputs = backward_buffers.outputs.default
        Ha = forward_buffers.internals.Ha
        dHa = backward_buffers.internals.Ha

        _h.copy_to(dHa, doutputs[1:])
        T = inputs.shape[0] - 1
        self.act_func_deriv(Ha[T], outputs[T + 1], doutputs[T + 1], dHa[T])
        for t in range(T - 1, -1, -1):
            self.act_func_deriv(Ha[t], outputs[t + 1], doutputs[t + 1], dHa[t])
            _h.dot_add_mm(dHa[t + 1], R, dHa[t], transb='T')

        t, b, f = dHa.shape
        i = t * b
        flat_dHa = dHa.reshape((i, f))
        flat_inputs = inputs.reshape((i, inputs.shape[2]))
        flat_outputs = outputs[:-1].reshape((i, outputs.shape[2]))
        flat_dinputs = dinputs.reshape((i, dinputs.shape[2]))

        # calculate in_deltas and gradients
        _h.dot_add_mm(flat_dHa, W, flat_dinputs, transb='T')

        _h.dot_add_mm(flat_inputs, flat_dHa, dW, transa='T')
        _h.dot_add_mm(flat_outputs, flat_dHa, dR, transa='T')

        dbias_tmp = _h.allocate(dbias.shape)
        _h.sum_t(flat_dHa, axis=0, out=dbias_tmp)
        _h.add_tt(dbias, dbias_tmp, dbias)

