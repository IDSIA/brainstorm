#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from brainstorm.layers.base_layer import LayerBaseImpl
from brainstorm.structure.shapes import ShapeTemplate


class ElementwiseLayerImpl(LayerBaseImpl):
    """
    This layer just applies an activation function to its inputs.
    """
    expected_kwargs = {'activation_function'}
    inputs = {'default': ShapeTemplate('T', 'B', '...')}
    outputs = {'default': ShapeTemplate('T', 'B', '...')}

    def _setup_hyperparameters(self):
        self.act_func = None
        self.act_func_deriv = None

    def set_handler(self, new_handler):
        super(ElementwiseLayerImpl, self).set_handler(new_handler)

        # Assign act_func and act_dunc_derivs
        activation_functions = {
            'sigmoid': (self.handler.sigmoid, self.handler.sigmoid_deriv),
            'tanh': (self.handler.tanh, self.handler.tanh_deriv),
            'linear': (lambda x, y: self.handler.copy_to(y, x),
                       lambda x, y, dy, dx: self.handler.copy_to(dx, dy)),
            'rel': (self.handler.rel, self.handler.rel_deriv)
        }

        self.act_func, self.act_func_deriv = activation_functions[
            self.kwargs.get('activation_function', 'rel')]

    def _get_output_shapes(self):
        return self.in_shapes

    def forward_pass(self, buffers, training_pass=True):
        self.act_func(buffers.inputs.default, buffers.outputs.default)

    def backward_pass(self, buffers):
        tmp = self.handler.allocate(buffers.input_deltas.default.shape)
        self.act_func_deriv(buffers.inputs.default, buffers.outputs.default,
                            buffers.output_deltas.default, tmp)
        self.handler.add_tt(buffers.input_deltas.default, tmp,
                            buffers.input_deltas.default)
