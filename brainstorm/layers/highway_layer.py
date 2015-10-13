#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

from collections import OrderedDict

from brainstorm.layers.base_layer import Layer
from brainstorm.structure.buffer_structure import (BufferStructure,
                                                   StructureTemplate)
from brainstorm.structure.construction import ConstructionWrapper
from brainstorm.utils import LayerValidationError


def Highway(name=None):
    """Create a Highway layer."""
    return ConstructionWrapper.create(HighwayLayerImpl, name=name)


class HighwayLayerImpl(Layer):
    expected_inputs = {'H': StructureTemplate('T', 'B', '...'),
                       'T': StructureTemplate('T', 'B', '...'),
                       'x': StructureTemplate('T', 'B', '...')}

    def setup(self, kwargs, in_shapes):
        # 'H', 'T' and 'x' must have the same shape
        if in_shapes['H'] != in_shapes['T']:
            raise LayerValidationError(
                "{}: H and T must have the same shape but got {} and {}"
                .format(self.name, in_shapes['H'], in_shapes['T']))
        if in_shapes['H'] != in_shapes['x']:
            raise LayerValidationError(
                "{}: H and x must have the same shape but got {} and {}"
                .format(self.name, in_shapes['H'], in_shapes['x']))

        outputs = OrderedDict()
        outputs['default'] = BufferStructure(
            'T', 'B', *self.in_shapes['x'].feature_shape)
        return outputs, OrderedDict(), OrderedDict()

    def forward_pass(self, buffers, training_pass=True):
        # prepare
        _h = self.handler
        x = buffers.inputs.x
        H = buffers.inputs.H
        T = buffers.inputs.T
        y = buffers.outputs.default

        tmp = _h.zeros(x.shape)
        _h.subtract_tt(H, x, out=tmp)
        _h.mult_tt(T, tmp, out=tmp)
        _h.add_tt(tmp, x, out=y)

    def backward_pass(self, buffers):
        # prepare
        _h = self.handler
        x = buffers.inputs.x
        H = buffers.inputs.H
        T = buffers.inputs.T
        dx = buffers.input_deltas.x
        dH = buffers.input_deltas.H
        dT = buffers.input_deltas.T
        dy = buffers.output_deltas.default

        tmp = _h.ones(dx.shape)
        _h.subtract_tt(tmp, T, out=tmp)
        _h.mult_add_tt(tmp, dy, out=dx)

        _h.mult_add_tt(T, dy, out=dH)

        _h.subtract_tt(H, x, out=tmp)
        _h.mult_add_tt(tmp, dy, out=dT)
