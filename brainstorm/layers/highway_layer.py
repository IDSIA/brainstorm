#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from brainstorm.utils import LayerValidationError
from brainstorm.layers.base_layer import LayerBaseImpl
from brainstorm.structure.shapes import ShapeTemplate


class HighwayLayerImpl(LayerBaseImpl):
    inputs = {'H': ShapeTemplate('T', 'B', '...'),
              'T': ShapeTemplate('T', 'B', '...'),
              'x': ShapeTemplate('T', 'B', '...')}
    outputs = {'default': ShapeTemplate('T', 'B', '...')}

    def _get_output_shapes(self):
        return {'default': ShapeTemplate('T', 'B',
                                         *self.in_shapes['x'].feature_shape)}

    def _validate_in_shapes(self):
        """Ensure self.in_shapes are valid.

         Raise LayerValidationError otherwise."""
        super(HighwayLayerImpl, self)._validate_in_shapes()

        # 'H', 'T' and 'x' must have the same shape
        if self.in_shapes['H'] != self.in_shapes['T']:
            raise LayerValidationError(
                "{}: H and T must have the same shape but got {} and {}"
                .format(self.name, self.in_shapes['H'], self.in_shapes['T']))
        if self.in_shapes['H'] != self.in_shapes['x']:
            raise LayerValidationError(
                "{}: H and x must have the same shape but got {} and {}"
                .format(self.name, self.in_shapes['H'], self.in_shapes['x']))

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

        _h.mult_tt(T, dy, out=tmp)
        _h.add_tt(tmp, dH, out=dH)

        _h.subtract_tt(H, x, out=tmp)
        _h.mult_add_tt(tmp, dy, out=dT)
