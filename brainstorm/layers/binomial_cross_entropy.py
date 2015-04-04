#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

from __future__ import division, print_function, unicode_literals
from brainstorm.layers.base_layer import LayerBase
from brainstorm.utils import LayerValidationError


class BinomialCrossEntropyLayer(LayerBase):
    """
    Calculate the Binomial Cross Entropy between outputs and **binary** targets

    Cross entropy is by definition asymmetric, therefore the inputs are named
    'default' for the network outputs and 'targets' for the binary targets.
    This layer only calculates the deltas for the default inputs.
    Also note that this implementation only works for **binary** targets and
    outputs in the range 0 to 1.
    For outputs outside that range or non-binary targets the result is
    undefined.
    """

    inputs = {'default': ('T', 'B', 'F'),
              'targets': ('T', 'B', 'F')
              }

    outputs = {'default': ('T', 'B', 1)}

    expected_kwargs = {}

    def _get_output_shapes(self):
        return {'default': (self.in_shapes['default'][0],
                            self.in_shapes['default'][1], 1)}

    def get_internal_structure(self):
        feature_shape = self.in_shapes['default'][2:]
        return {
            'cee': {
                '@shape': ('T', 'B') + feature_shape,
                '@index': 0}
        }

    def _validate_in_shapes(self):
        super(BinomialCrossEntropyLayer, self)._validate_in_shapes()

        # 'default' and 'targets' must be wired in
        # and their shapes must match
        if 'default' not in self.in_shapes or 'targets' not in self.in_shapes:
            raise LayerValidationError("{} must have both 'default' and "
                                       "'targets' as inputs".format(self.name))
        if self.in_shapes['default'] != self.in_shapes['targets']:
            raise LayerValidationError("{}: default and targets must have the "
                                       "same shapes but got {} and {}"
                                       .format(self.name,
                                               self.in_shapes['default'],
                                               self.in_shapes['targets']))

    def _validate_out_shapes(self):
        super(BinomialCrossEntropyLayer, self)._validate_out_shapes()
        assert self.out_shapes['default'][:2] == self.in_shapes['default'][:2]

        if len(self.out_shapes['default']) != 3:
            raise LayerValidationError(
                '{}: this layer sums over feature '
                'dimensions, so len(out_shape) must be 3, '
                'but shape is {}'.format(self.name,
                                         self.out_shapes['default'])
            )
        if self.out_shapes['default'][2] != 1:
            raise LayerValidationError(
                '{}: this layer sums over feature dimensions, so out_shape[2] '
                'must be 1, but got {}'.format(self.name,
                                               self.out_shapes['default'][2])
            )

    def forward_pass(self, forward_buffers, training_pass=True):
        # prepare
        _h = self.handler
        y = forward_buffers.inputs.default
        t = forward_buffers.inputs.targets
        cee = forward_buffers.internals.cee
        cee_sum = forward_buffers.outputs.default

        # the binomial cross entropy error is given by
        # - t * ln(y) - (1-t) * ln(1-y)
        # which for binary t simplifies to
        # - ln(1 - t + y)
        _h.fill(cee, 1.0)               # 1
        _h.subtract_tt(cee, t, cee)     # - t
        _h.add_tt(cee, y, cee)          # + y
        _h.clip_t(cee, 1e-6, 1.0, cee)  # clip
        _h.log_t(cee, cee)              # log

        # reshape for summation
        t, b = cee.shape[:2]
        f = _h.size(cee) / (t * b)
        cee = cee.reshape((t, b, f))

        _h.sum_t(cee, axis=2, out=cee_sum)
        _h.elem_mult_st(-1, cee_sum, cee_sum)  # * -1

    def backward_pass(self, forward_buffers, backward_buffers):
        # prepare
        _h = self.handler
        ceed_sum = backward_buffers.outputs.default
        ceed = backward_buffers.internals.cee

        y = forward_buffers.inputs.default
        t = forward_buffers.inputs.targets

        yd = backward_buffers.inputs.default

        # the derivative of the binomial cross entropy error is given by
        # (y - t) / (y - y²)
        # which for binary t simplifies to
        # 1 / (1 - t - y)
        tmp = _h.ones(ceed.shape)       # 1
        _h.subtract_tt(tmp, t, tmp)     # - t
        _h.subtract_tt(tmp, y, tmp)     # - y
        _h.clip_t(tmp, 1e-6, 1.0, tmp)  # clip

        # ceed_sum has only one feature dimension due to summation,
        # so we broadcast to all feature dimensions
        _h.broadcast_features_t(ceed_sum, ceed)
        _h.divide_tt(ceed, tmp, ceed)

        _h.add_tt(ceed, yd, yd)

