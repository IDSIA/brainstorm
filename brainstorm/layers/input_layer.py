#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

from collections import OrderedDict

from brainstorm.layers.base_layer import Layer
from brainstorm.structure.buffer_structure import BufferStructure
from brainstorm.structure.construction import ConstructionWrapper
from brainstorm.utils import LayerValidationError


def Input(out_shapes):
    """Create an Input layer.
    Special input layer type, that provides access to external data.

    The 'out_shapes' keyword argument is required and specifies the names and
    shapes of all external inputs.
    """
    return ConstructionWrapper.create(InputLayerImpl, out_shapes=out_shapes)


class InputLayerImpl(Layer):

    expected_inputs = {}
    expected_kwargs = {'out_shapes'}

    def setup(self, kwargs, in_shapes):
        if 'out_shapes' not in kwargs:
            raise LayerValidationError("InputLayer requires 'out_shapes'")
        if in_shapes:
            raise LayerValidationError(
                'InputLayer cannot have any incoming connections!'
                '(But had these: {})'.format(in_shapes))

        outputs = OrderedDict()
        for n, s in self.kwargs['out_shapes'].items():
            outputs[n] = BufferStructure(*s)
        return outputs, OrderedDict(), OrderedDict()

    def _validate_connections(self):
        super(InputLayerImpl, self)._validate_connections()

        if self.incoming:
            raise LayerValidationError(
                'InputLayer cannot have any incoming connections!'
                '(But had these: {})'.format(self.incoming))
