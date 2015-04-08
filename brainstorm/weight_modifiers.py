#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np
from brainstorm.randomness import Seedable
from brainstorm.describable import Describable
from brainstorm.utils import WeightModificationError


class WeightModifier(Seedable, Describable):

    __undescribed__ = {'layer_name', 'view_name'}

    def __init__(self):
        super(WeightModifier, self).__init__()
        self.layer_name = ''
        self.view_name = ''

    def __call__(self, handler, view):
        raise NotImplementedError()

    def __repr__(self):
        return "<{}.{}.{}}>".format(self.layer_name, self.view_name,
                                    self.__class__.__name__)


class RescaleIncomingWeights(WeightModifier):

    """
    Rescales the incoming weights for every neuron to sum to one (target_sum).
    Ignores Biases.

    Should be added to the network via the set_weight_modifiers method like so:

    >> net.set_weight_modifiers(RnnLayer={'HX': RescaleIncomingWeights()})

    See Network.set_weight_modifiers for more information on how to control
    which weights to affect.
    """

    __default_values__ = {'target_sum': 1.0}

    def __init__(self, target_sum=1.0):
        super(RescaleIncomingWeights, self).__init__()
        self.target_sum = target_sum

    def __call__(self, handler, view):
        if not len(view.shape) == 2:  # only works for two dimensional inputs
            raise WeightModificationError(
                '{} only works for two dimensional parameters'
                .format(self.__class__.__name__))

        column_sum = handler.allocate((1, view.shape[1]))
        handler.sum_t(view, axis=0, out=column_sum)
        handler.elem_mult_st(self.target_sum, column_sum, column_sum)
        handler.divide_mv(view, column_sum, view)

    def __repr__(self):
        return "<{}.{}.RescaleIncomingWeights to {:0.4f}>"\
            .format(self.layer_name, self.view_name, self.target_sum)


class ClipWeights(WeightModifier):

    """
    Clips (limits) the weights to be between low and high.
    Defaults to low=-1 and high=1.

    Should be added to the network via the set_weight_modifiers method like so:

    >> net.set_weight_modifiers(RnnLayer={'HR': ClipWeights()})

    See Network.set_weight_modifiers for more information on how to control
    which weights to affect.
    """

    def __init__(self, low=-1., high=1.):
        super(ClipWeights, self).__init__()
        self.low = low
        self.high = high

    def __call__(self, handler, view):
        handler.clip_t(view, self.low, self.high, view)

    def __repr__(self):
        return "<{}.{}.ClipWeights [{:0.4f}; {:0.4f}]>"\
            .format(self.layer_name, self.view_name, self.low, self.high)


class MaskWeights(WeightModifier):

    """
    Multiplies the weights elementwise with the mask.

    This can be used to clamp some of the weights to zero.

    Should be added to the network via the set_weight_modifiers method like so:

    >> net.set_weight_modifiers(RnnLayer={'HR': MaskWeights(M)})

    See Network.set_weight_modifiers for more information on how to control
    which weights to affect.
    """

    __undescribed__ = {'device_mask'}

    def __init__(self, mask):
        super(MaskWeights, self).__init__()
        assert isinstance(mask, np.ndarray)
        self.mask = mask
        self.device_mask = None

    def __call__(self, handler, view):
        if (self.device_mask is None or
                not isinstance(self.device_mask, handler.array_type)):
            self.device_mask = handler.allocate(self.mask.shape)
            handler.set_from_numpy(self.device_mask, self.mask)
        handler.elem_mult_tt(view, self.device_mask, view)


class FreezeWeights(WeightModifier):

    """
    Prevents the weights from changing at all.

    If the weights argument is left at None it will remember the first weights
    it sees and resets them to that every time.

    Should be added to the network via the set_constraints method like so:
    >> net.set_constraints(RnnLayer={'HR': FreezeWeights()})
    See Network.set_constraints for more information on how to control which
    weights to affect.
    """

    __undescribed__ = {'weights', 'device_weights'}

    def __init__(self, weights=None):
        super(FreezeWeights, self).__init__()
        self.weights = weights
        self.device_weights = None

    def __call__(self, handler, view):
        if self.weights is None:
            self.weights = handler.get_numpy_copy(view)

        if (self.device_weights is None or
                not isinstance(self.device_weights, handler.array_type)):
            self.device_weights = handler.allocate(self.weights.shape)
            handler.set_from_numpy(self.device_weights, self.weights)

        handler.copy_to(view, self.device_weights)
