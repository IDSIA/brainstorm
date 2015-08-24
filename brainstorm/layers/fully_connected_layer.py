#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from collections import OrderedDict
from brainstorm.utils import LayerValidationError
from brainstorm.layers.base_layer import LayerBaseImpl
from brainstorm.structure.shapes import ShapeTemplate


class FullyConnectedLayerImpl(LayerBaseImpl):
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
        super(FullyConnectedLayerImpl, self).set_handler(new_handler)

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
        parameters['W'] = ShapeTemplate(self.size, in_size)
        parameters['bias'] = ShapeTemplate(self.size)
        return parameters

    def get_internal_structure(self):
        internals = OrderedDict()
        internals['H'] = ShapeTemplate('T', 'B', self.size)
        internals['dH'] = ShapeTemplate('T', 'B', self.size,
                                        is_backward_only=True)
        return internals

    def _get_output_shapes(self):
        return {'default': ShapeTemplate('T', 'B', self.size)}

    def forward_pass(self, buffers, training_pass=True):
        # prepare
        _h = self.handler
        W, bias = buffers.parameters
        inputs = buffers.inputs.default
        outputs = buffers.outputs.default
        H = buffers.internals.H

        # reshape
        t, b, f = inputs.shape
        flat_input = _h.reshape(inputs, (t * b, f))
        flat_H = _h.reshape(H, (t * b, self.out_shapes['default'][2]))

        # calculate outputs
        _h.dot_mm(flat_input, W, flat_H, transb='T')
        _h.add_mv(flat_H, bias, flat_H)
        self.act_func(H, outputs)

    def backward_pass(self, buffers):
        # prepare
        _h = self.handler
        W, bias = buffers.parameters
        dW, dbias = buffers.gradients
        inputs = buffers.inputs.default
        outputs = buffers.outputs.default
        in_deltas = buffers.input_deltas.default
        out_deltas = buffers.output_deltas.default
        H, dH = buffers.internals

        # reshape
        t, b, f = inputs.shape
        flat_input = _h.reshape(inputs, (t * b, f))
        flat_dH = _h.reshape(dH, (t * b, self.out_shapes['default'][2]))
        flat_in_delta_buffer = _h.reshape(in_deltas, (t * b, f))

        # calculate in_deltas and gradients
        self.act_func_deriv(H, outputs, out_deltas, dH)
        _h.dot_add_mm(flat_dH, W, out=flat_in_delta_buffer)
        _h.dot_mm(flat_dH, flat_input, out=dW, transa='T')
        _h.sum_t(flat_dH, axis=0, out=dbias)
