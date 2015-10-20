#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

from collections import OrderedDict

from brainstorm.layers.base_layer import Layer
from brainstorm.structure.buffer_structure import (BufferStructure,
                                                   StructureTemplate)
from brainstorm.structure.construction import ConstructionWrapper
from brainstorm.utils import LayerValidationError, flatten_all_but_last


def Merge(name=None):
    """Create a layer that merges two inputs into one along the last dim"""
    return ConstructionWrapper.create(MergeLayerImpl, name=name)


class MergeLayerImpl(Layer):
    expected_inputs = {'inputs_1': StructureTemplate('...'),
                       'inputs_2': StructureTemplate('...')}
    expected_kwargs = {}

    def setup(self, kwargs, in_shapes):
        # 'inputs_1' and 'inputs_2' must have same shape except for last dim
        shape_prefix1 = in_shapes['inputs_1'].shape[:-1]
        shape_prefix2 = in_shapes['inputs_2'].shape[:-1]
        if shape_prefix1 != shape_prefix2:
            raise LayerValidationError(
                "{}: The shapes of inputs_1 and inputs_2 may only differ in "
                "the last dimension but got {} and {}".format(
                    self.name,
                    in_shapes['inputs_1'].shape,
                    in_shapes['inputs_2'].shape))

        combined_size = (in_shapes['inputs_1'].shape[-1] +
                         in_shapes['inputs_2'].shape[-1])
        out_shape = shape_prefix1 + (combined_size,)
        outputs = OrderedDict()
        outputs['default'] = BufferStructure(*out_shape)

        parameters = OrderedDict()
        internals = OrderedDict()
        return outputs, parameters, internals

    def forward_pass(self, buffers, training_pass=True):
        # prepare
        self.handler.merge_tt(buffers.inputs.inputs_1,
                              buffers.inputs.inputs_2,
                              buffers.outputs.default)

    def backward_pass(self, buffers):
        # prepare
        _h = self.handler
        self.handler.split_add_tt(buffers.output_deltas.default,
                                  buffers.input_deltas.inputs_1,
                                  buffers.input_deltas.inputs_2)
