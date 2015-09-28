#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from brainstorm.structure.construction import ConstructionWrapper
from brainstorm.layers.base_layer import BaseLayerImpl
from brainstorm.utils import LayerValidationError
from brainstorm.structure.shapes import BufferStructure


def Input(out_shapes, name=None):
    return ConstructionWrapper.create('Input',
                                      name=name,
                                      out_shapes=out_shapes)


class InputLayerImpl(BaseLayerImpl):
    """
    Special input layer type, that provides access to external data.

    The 'out_shapes' kwarg is required and specifies the names and shapes of
    all external inputs.
    """
    expected_kwargs = {'out_shapes'}
    inputs = {}
    outputs = {}  # special

    def _get_output_shapes(self):
        if 'out_shapes' not in self.kwargs:
            raise LayerValidationError("InputLayer requires 'out_shapes'")

        return {n: BufferStructure.from_tuple(s)
                for n, s in self.kwargs['out_shapes'].items()}

    def _validate_in_shapes(self):
        if self.in_shapes:
            raise LayerValidationError(
                'InputLayer cannot have any incoming connections!'
                '(But had these: {})'.format(self.in_shapes))

    def _validate_out_shapes(self):
        pass

    def _validate_connections(self):
        super(InputLayerImpl, self)._validate_connections()

        if self.incoming:
            raise LayerValidationError(
                'InputLayer cannot have any incoming connections!'
                '(But had these: {})'.format(self.incoming))
