#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function
from functools import partial
import sys
from brainstorm.utils import get_inheritors
from brainstorm.structure.construction import ConstructionWrapper

from brainstorm.layers.base_layer import BaseLayerImpl
from brainstorm.layers.batch_normalization_layer import BatchNorm
from brainstorm.layers.binomial_cross_entropy_layer import BinomialCrossEntropy
from brainstorm.layers.classification_layer import Classification
from brainstorm.layers.convolution_layer_2d import Convolution2D
from brainstorm.layers.dropout_layer import Dropout
from brainstorm.layers.elementwise_layer import Elementwise
from brainstorm.layers.fully_connected_layer import FullyConnected
from brainstorm.layers.highway_layer import Highway
from brainstorm.layers.input_layer import Input
from brainstorm.layers.loss_layer import Loss
from brainstorm.layers.lstm_layer import Lstm
from brainstorm.layers.lstm_opt_layer import LstmOpt
from brainstorm.layers.mask_layer import Mask
from brainstorm.layers.noop_layer import NoOp
from brainstorm.layers.pooling_layer_2d import Pooling2D
from brainstorm.layers.rnn_layer import Recurrent
from brainstorm.layers.squared_difference_layer import SquaredDifference
from brainstorm.layers.l1_decay import L1Decay
from brainstorm.layers.l2_decay import L2Decay

CONSTRUCTION_LAYERS = {}

# ------------------------ Automatic Construction Layers ----------------------


def construction_layer_for(layer_impl):
    layer_name = layer_impl.__name__
    assert layer_name.endswith('LayerImpl'), \
        "{} should end with 'LayerImpl'".format(layer_name)
    layer_name = layer_name[:-9]
    return partial(ConstructionWrapper.create, layer_name)


for Layer in get_inheritors(BaseLayerImpl):
    layer_name = Layer.__name__[:-9]
    if layer_name not in CONSTRUCTION_LAYERS:
        CONSTRUCTION_LAYERS[layer_name] = construction_layer_for(Layer)


this_module = sys.modules[__name__]  # this module
for name, cl in CONSTRUCTION_LAYERS.items():
    if not hasattr(this_module, name):
        setattr(this_module, name, cl)


# somehow str is needed because in __all__ unicode does not work
__all__ = ['construction_layer_for'] + [str(a) for a in CONSTRUCTION_LAYERS]
