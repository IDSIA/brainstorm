#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from brainstorm.structure.construction import ConstructionWrapper
from brainstorm.layers.base_layer import LayerBaseImpl
from brainstorm.structure.shapes import ShapeTemplate


def NoOp(name=None):
    return ConstructionWrapper.create('NoOp', name=name)


class NoOpLayerImpl(LayerBaseImpl):
    """
    This layer just copies its input into its output.
    """
    expected_kwargs = {}
    inputs = {'default': ShapeTemplate('T', 'B', '...')}
    outputs = {'default': ShapeTemplate('T', 'B', '...')}

    def _get_output_shapes(self):
        return self.in_shapes

    def forward_pass(self, buffers, training_pass=True):
        self.handler.copy_to(buffers.outputs.default,
                             buffers.inputs.default)

    def backward_pass(self, buffers):
        self.handler.add_tt(buffers.output_deltas.default,
                            buffers.input_deltas.default,
                            out=buffers.input_deltas.default)
