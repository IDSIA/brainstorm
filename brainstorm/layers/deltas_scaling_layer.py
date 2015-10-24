#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

from collections import OrderedDict

from brainstorm.layers.base_layer import Layer
from brainstorm.structure.buffer_structure import StructureTemplate
from brainstorm.structure.construction import ConstructionWrapper
from brainstorm.utils import LayerValidationError


def DeltasScaling(factor, name=None):
    """Create an DeltasScaling layer.

    This layer does nothing on the forward pass, but scales the deltas flowing
    back during the backward pass by a given factor.

    This can be used to invert the deltas and set up an adversarial branch of
    the network.
    """
    return ConstructionWrapper.create(DeltasScalingLayerImpl, name=name,
                                      factor=factor)


class DeltasScalingLayerImpl(Layer):
    expected_inputs = {'default': StructureTemplate('T', 'B', '...')}
    expected_kwargs = {'factor'}

    def setup(self, kwargs, in_shapes):
        if 'factor' not in kwargs:
            raise LayerValidationError('Missing required "factor" argument')
        self.factor = kwargs['factor']
        out_shapes = in_shapes
        return out_shapes, OrderedDict(), OrderedDict()

    def forward_pass(self, buffers, training_pass=True):
        self.handler.copy_to(buffers.inputs.default, buffers.outputs.default)

    def backward_pass(self, buffers):
        self.handler.mult_add_st(self.factor,
                                 buffers.output_deltas.default,
                                 buffers.input_deltas.default)
