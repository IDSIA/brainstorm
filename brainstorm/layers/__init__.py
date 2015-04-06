#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from functools import partial
import sys
from brainstorm.utils import get_inheritors
from brainstorm.structure.construction import ConstructionWrapper

# need to import every layer implementation here
from brainstorm.layers.base_layer import LayerBaseImpl
from brainstorm.layers.input_layer import InputLayerImpl
from brainstorm.layers.noop_layer import NoOpLayerImpl
from brainstorm.layers.fully_connected_layer import FullyConnectedLayerImpl
from brainstorm.layers.squared_difference_layer import \
    SquaredDifferenceLayerImpl
from brainstorm.layers.binomial_cross_entropy_layer import \
    BinomialCrossEntropyLayerImpl
from brainstorm.layers.loss_layer import LossLayerImpl


def construction_layer_for(layer_impl):
    layer_name = Layer.__name__
    assert layer_name.endswith('Impl'), \
        "{} should end with 'Impl'".format(layer_name)
    layer_name = layer_name[:-4]
    return partial(ConstructionWrapper.create, layer_name)


construction_layers = {}
for Layer in get_inheritors(LayerBaseImpl):
    layer_name = Layer.__name__[:-4]
    construction_layers[layer_name] = construction_layer_for(Layer)


this_module = sys.modules[__name__]  # this module
for name, cl in construction_layers.items():
    setattr(this_module, name, cl)


# somehow str is needed because in __all__ unicode does not work
__all__ = ['construction_layer_for'] + [str(a) for a in construction_layers]
