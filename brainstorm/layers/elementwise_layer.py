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
    return ConstructionWrapper.create(ElementwiseLayerImpl, name=name,
                                      activation=activation)


class ElementwiseLayerImpl(Layer):

    expected_inputs = {'default': StructureTemplate('T', 'B', '...')}
    expected_kwargs = {'activation'}

    def setup(self, kwargs, in_shapes):
        self.activation = kwargs.get('activation', 'rel')
        return in_shapes, OrderedDict(), OrderedDict()

    def forward_pass(self, buffers, training_pass=True):
        self.handler.act_func[self.activation](buffers.inputs.default,
                                               buffers.outputs.default)

    def backward_pass(self, buffers):
        tmp = self.handler.allocate(buffers.input_deltas.default.shape)
        self.handler.act_func_deriv[self.activation](
            buffers.inputs.default, buffers.outputs.default,
            buffers.output_deltas.default, tmp)
        self.handler.add_tt(buffers.input_deltas.default, tmp,
                            buffers.input_deltas.default)
