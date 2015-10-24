#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

from collections import OrderedDict

from brainstorm.layers.base_layer import Layer
from brainstorm.structure.buffer_structure import StructureTemplate
from brainstorm.structure.construction import ConstructionWrapper


def Dropout(drop_prob=0.5, name=None):
    """Create a Dropout layer.

    drop_prob is the probability of a unit being dropped, i.e. 0
    """
    return ConstructionWrapper.create(DropoutLayerImpl, drop_prob=drop_prob,
                                      name=name)


class DropoutLayerImpl(Layer):

    expected_inputs = {'default': StructureTemplate('T', 'B', '...')}
    expected_kwargs = {'drop_prob'}

    def setup(self, kwargs, in_shapes):
        self.drop_prob = kwargs.get('drop_prob', 0.5)

        outputs = OrderedDict()
        outputs['default'] = in_shapes['default']

        internals = OrderedDict()
        internals['mask'] = self.in_shapes['default']
        return outputs, OrderedDict(), internals

    def forward_pass(self, buffers, training_pass=True):
        _h = self.handler

        if training_pass:
            _h.generate_probability_mask(buffers.internals.mask,
                                         1 - self.drop_prob)
            _h.mult_tt(buffers.inputs.default, buffers.internals.mask,
                       out=buffers.outputs.default)
            _h.mult_st(1 / (1 - self.drop_prob), buffers.outputs.default,
                       out=buffers.outputs.default)
        else:
            _h.copy_to(buffers.inputs.default, buffers.outputs.default)

    def backward_pass(self, buffers):
        self.handler.mult_add_tt(buffers.output_deltas.default,
                                 buffers.internals.mask,
                                 buffers.input_deltas.default)
