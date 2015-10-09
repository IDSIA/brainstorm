#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

from collections import OrderedDict

from brainstorm.layers.base_layer import Layer
from brainstorm.structure.buffer_structure import StructureTemplate
from brainstorm.structure.construction import ConstructionWrapper


def Elementwise(activation='rel', name=None):
    """Create an Elementwise layer.

    This layer just applies a unit-wise function to its inputs.
    """
    return ConstructionWrapper.create('Elementwise', name=name,
                                      activation=activation)


class ElementwiseLayerImpl(Layer):

    expected_inputs = {'default': StructureTemplate('T', 'B', '...')}
    expected_kwargs = {'activation'}

    def set_handler(self, new_handler):
        super(ElementwiseLayerImpl, self).set_handler(new_handler)

        # Assign act_func and act_dunc_derivs
        activations = {
            'sigmoid': (self.handler.sigmoid, self.handler.sigmoid_deriv),
            'tanh': (self.handler.tanh, self.handler.tanh_deriv),
            'linear': (lambda x, y: self.handler.copy_to(x, y),
                       lambda x, y, dy, dx: self.handler.copy_to(dy, dx)),
            'rel': (self.handler.rel, self.handler.rel_deriv)
        }

        self.act_func, self.act_func_deriv = activations[
            self.kwargs.get('activation', 'rel')]

    def setup(self, kwargs, in_shapes):
        self.act_func = None
        self.act_func_deriv = None
        return in_shapes, OrderedDict(), OrderedDict()

    def forward_pass(self, buffers, training_pass=True):
        self.act_func(buffers.inputs.default, buffers.outputs.default)

    def backward_pass(self, buffers):
        tmp = self.handler.allocate(buffers.input_deltas.default.shape)
        self.act_func_deriv(buffers.inputs.default, buffers.outputs.default,
                            buffers.output_deltas.default, tmp)
        self.handler.add_tt(buffers.input_deltas.default, tmp,
                            buffers.input_deltas.default)
