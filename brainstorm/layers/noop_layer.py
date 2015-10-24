#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

from collections import OrderedDict

from brainstorm.layers.base_layer import Layer
from brainstorm.structure.buffer_structure import StructureTemplate
from brainstorm.structure.construction import ConstructionWrapper


def NoOp(name=None):
    """Create a NoOp layer.

    This layer just copies its input into its output.
    """
    return ConstructionWrapper.create(NoOpLayerImpl, name=name)


class NoOpLayerImpl(Layer):

    expected_inputs = {'default': StructureTemplate('T', 'B', '...')}
    expected_kwargs = {}

    def setup(self, kwargs, in_shapes):
        return self.in_shapes, OrderedDict(), OrderedDict()

    def forward_pass(self, buffers, training_pass=True):
        self.handler.copy_to(buffers.inputs.default, buffers.outputs.default)

    def backward_pass(self, buffers):
        self.handler.add_tt(buffers.output_deltas.default,
                            buffers.input_deltas.default,
                            out=buffers.input_deltas.default)
