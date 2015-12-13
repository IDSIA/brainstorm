#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

from collections import OrderedDict

from brainstorm.layers.base_layer import Layer
from brainstorm.structure.buffer_structure import StructureTemplate
from brainstorm.structure.construction import ConstructionWrapper
from brainstorm.utils import LayerValidationError, product


def Mask(name=None):
    """Create a Mask layer."""
    return ConstructionWrapper.create(MaskLayerImpl, name=name)


class MaskLayerImpl(Layer):

    expected_inputs = {'default': StructureTemplate('T', 'B', '...'),
                       'mask': StructureTemplate('T', 'B', '...')}

    computes_no_input_deltas_for = ['mask']

    def setup(self, kwargs, in_shapes):
        in_shape = in_shapes['default'].feature_shape
        expected_shape = in_shape[:-1] + (1,)

        if in_shapes['mask'].feature_shape == (1,):
            self.flatten_dim = 2
        elif in_shapes['mask'].feature_shape in [expected_shape, in_shape]:
            self.flatten_dim = len(in_shape) + 1
        else:
            raise LayerValidationError(
                "Shape of the mask did not match shape of the default inputs. "
                "Should be either ('T', 'B', 1) or {} or {}, but was {}"
                .format(('T', 'B') + expected_shape,
                        in_shapes['default'].shape,
                        in_shapes['mask']))
        outputs = OrderedDict()
        outputs['default'] = in_shapes['default']
        return outputs, OrderedDict(), OrderedDict()

    def flatten_buffer(self, buffer):
        pre = buffer.shape[:self.flatten_dim]
        post = buffer.shape[self.flatten_dim:]
        return buffer.reshape((int(product(pre)), int(product(post))))

    def forward_pass(self, buffers, training_pass=True):
        _h = self.handler

        flat_inp = self.flatten_buffer(buffers.inputs.default)
        flat_mask = self.flatten_buffer(buffers.inputs.mask)
        flat_out = self.flatten_buffer(buffers.outputs.default)

        _h.mult_mv(flat_inp, flat_mask, out=flat_out)

    def backward_pass(self, buffers):
        _h = self.handler

        flat_out_deltas = self.flatten_buffer(buffers.output_deltas.default)
        tmp = self.handler.allocate(flat_out_deltas.shape)
        flat_mask = self.flatten_buffer(buffers.inputs.mask)
        flat_in_deltas = self.flatten_buffer(buffers.input_deltas.default)

        _h.mult_mv(flat_out_deltas, flat_mask, tmp)
        _h.add_tt(tmp, flat_in_deltas, flat_in_deltas)
