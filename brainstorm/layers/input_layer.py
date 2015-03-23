#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from brainstorm.layers.base_layer import LayerBase
from brainstorm.utils import LayerValidationError


class InputLayer(LayerBase):
    """
    Special input layer type, that provides access to external data.

    The 'out_shapes' kwarg is required and specifies the names and shapes of
    all external inputs.
    """
    expected_kwargs = {'out_shapes'}
    inputs = []

    def _get_output_shapes(self):
        assert 'out_shapes' in self.kwargs, "InputLayer requires 'out_shapes'"
        return self.kwargs['out_shapes']

    def _validate_out_shapes(self):
        for output_name, shape in self.out_shapes.items():
            if not isinstance(shape, tuple):
                raise LayerValidationError(
                    'out_shape entry "{}" was not a shape'.format(shape))

    def _validate_connections(self):
        if self.incoming:
            raise LayerValidationError(
                'InputLayer cannot have any incoming connections!'
                '(But had these: {})'.format(self.incoming))

        for out_c in self.outgoing:
            if out_c.output_name not in self.out_shapes:
                raise LayerValidationError(
                    'Invalid outgoing connection ({}). {} has no output'
                    ' named "{}".\nChoices are {}.'.format(
                        out_c, self.name, out_c.output_name,
                        self.out_shapes.keys()))


