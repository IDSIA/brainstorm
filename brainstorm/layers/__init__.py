#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function
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
from brainstorm.layers.classification_layer import ClassificationLayerImpl
from brainstorm.layers.loss_layer import LossLayerImpl
from brainstorm.layers.mask_layer import MaskLayerImpl
from brainstorm.layers.lstm_layer import LstmLayerImpl
from brainstorm.layers.rnn_layer import RnnLayerImpl


CONSTRUCTION_LAYERS = {}

# ---------------- Specialized Construction Layers ----------------------------
# defined explicitly to provide improved autocompletion


def Input(out_shapes, name=None):
    return ConstructionWrapper.create('Input',
                                      name=name,
                                      out_shapes=out_shapes)


def FullyConnected(size, activation_function='linear', name=None):
    return ConstructionWrapper.create('FullyConnected',
                                      size=size,
                                      name=name,
                                      activation_function=activation_function)


def Loss(name=None):
    return ConstructionWrapper.create('Loss', name=name)


def Lstm(size, activation_function='tanh', name=None):
    return ConstructionWrapper.create('Lstm',
                                      size=size,
                                      name=name,
                                      activation_function=activation_function)


def LstmOpt(size, activation_function='tanh', name=None):
    return ConstructionWrapper.create('LstmOpt',
                                      size=size,
                                      name=name,
                                      activation_function=activation_function)


def BinomialCrossEntropy(name=None):
    return ConstructionWrapper.create('BinomialCrossEntropy',
                                      name=name)


def Classification(size, name=None):
    return ConstructionWrapper.create('Classification',
                                      size=size,
                                      name=name)


def Rnn(size, activation_function='tanh', name=None):
    return ConstructionWrapper.create('Rnn',
                                      size=size,
                                      name=name,
                                      activation_function=activation_function)


def SquaredDifference(name=None):
    return ConstructionWrapper.create('SquaredDifference',
                                      name=name)


def Mask(name=None):
    return ConstructionWrapper.create('Mask', name=name)


# ------------------------ Automatic Construction Layers ----------------------

def construction_layer_for(layer_impl):
    layer_name = layer_impl.__name__
    assert layer_name.endswith('LayerImpl'), \
        "{} should end with 'LayerImpl'".format(layer_name)
    layer_name = layer_name[:-9]
    return partial(ConstructionWrapper.create, layer_name)


for Layer in get_inheritors(LayerBaseImpl):
    layer_name = Layer.__name__[:-9]
    if layer_name not in CONSTRUCTION_LAYERS:
        CONSTRUCTION_LAYERS[layer_name] = construction_layer_for(Layer)


this_module = sys.modules[__name__]  # this module
for name, cl in CONSTRUCTION_LAYERS.items():
    if not hasattr(this_module, name):
        setattr(this_module, name, cl)


# somehow str is needed because in __all__ unicode does not work
__all__ = ['construction_layer_for'] + [str(a) for a in CONSTRUCTION_LAYERS]
