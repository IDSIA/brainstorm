#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from collections import OrderedDict
from brainstorm.utils import LayerValidationError, flatten_time
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
            self.kwargs.get('activation_function', 'tanh')]

    def get_parameter_structure(self):
        in_size = self.in_shapes['default'].feature_size
        parameters = OrderedDict()
        parameters['W'] = ShapeTemplate(self.size, in_size)
        parameters['R'] = ShapeTemplate(self.size, self.size)
        parameters['bias'] = ShapeTemplate(self.size)
        return parameters

    def get_internal_structure(self):
        internals = OrderedDict()
        internals['Ha'] = ShapeTemplate('T', 'B', self.size, context_size=1)
        internals['dHa'] = ShapeTemplate('T', 'B', self.size, context_size=1,
                                         is_backward_only=True)
        internals['dHb'] = ShapeTemplate('T', 'B', self.size, context_size=1,
                                         is_backward_only=True)
        return internals

    def _get_output_shapes(self):
        return {'default': ShapeTemplate('T', 'B', self.size, context_size=1)}

    def forward_pass(self, buffers, training_pass=True):
        # prepare
        _h = self.handler
        W, R, bias = buffers.parameters
        inputs = buffers.inputs.default
        outputs = buffers.outputs.default
        Ha = buffers.internals.Ha

        flat_inputs = flatten_time(inputs)
        flat_H = flatten_time(Ha[:-1])

        _h.dot_mm(flat_inputs, W, flat_H, transb=True)
        _h.add_mv(flat_H, bias, flat_H)

        for t in range(inputs.shape[0]):
            _h.dot_add_mm(outputs[t - 1], R, Ha[t])
            self.act_func(Ha[t], outputs[t])

    def backward_pass(self, buffers):
        # prepare
        _h = self.handler
        W, R, bias = buffers.parameters
        dW, dR, dbias = buffers.gradients
        inputs = buffers.inputs.default
        outputs = buffers.outputs.default
        dinputs = buffers.input_deltas.default
        doutputs = buffers.output_deltas.default
        Ha, dHa, dHb = buffers.internals

        _h.copy_to(dHb, doutputs)
        T = inputs.shape[0] - 1
        self.act_func_deriv(Ha[T], outputs[T], dHb[T], dHa[T])
        for t in range(T - 1, -1, -1):
            _h.dot_add_mm(dHa[t + 1], R, dHb[t], transb=True)
            self.act_func_deriv(Ha[t], outputs[t],
                                dHb[t], dHa[t])

        flat_inputs = flatten_time(inputs)
        flat_dinputs = flatten_time(dinputs)
        flat_dHa = flatten_time(dHa[:-1])

        # calculate in_deltas and gradients
        _h.dot_add_mm(flat_dHa, W, flat_dinputs)
        _h.dot_add_mm(flat_dHa, flat_inputs, dW, transa=True)
        dbias_tmp = _h.allocate(dbias.shape)
        _h.sum_t(flat_dHa, axis=0, out=dbias_tmp)
        _h.add_tt(dbias, dbias_tmp, dbias)

        flat_outputs = flatten_time(outputs[:-2])
        flat_dHa = flatten_time(dHa[1:-1])
        _h.dot_add_mm(flat_outputs, flat_dHa, dR, transa=True)
        _h.dot_add_mm(outputs[-1], dHa[0], dR, transa=True)
