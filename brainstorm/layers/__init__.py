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
from brainstorm.layers.highway_layer import HighwayLayerImpl
from brainstorm.layers.squared_difference_layer import \
    SquaredDifferenceLayerImpl
from brainstorm.layers.binomial_cross_entropy_layer import \
    BinomialCrossEntropyLayerImpl
from brainstorm.layers.classification_layer import ClassificationLayerImpl
from brainstorm.layers.loss_layer import LossLayerImpl
from brainstorm.layers.mask_layer import MaskLayerImpl
from brainstorm.layers.lstm_layer import LstmLayerImpl
from brainstorm.layers.lstm_opt_layer import LstmOptLayerImpl
from brainstorm.layers.rnn_layer import RnnLayerImpl
from brainstorm.layers.dropout_layer import DropoutLayerImpl
from brainstorm.layers.convolution_layer_2d import Convolution2DLayerImpl
from brainstorm.layers.pooling_layer_2d import Pooling2DLayerImpl
from brainstorm.layers.batch_normalization_layer import BatchNormLayerImpl
from brainstorm.layers.elementwise_layer import ElementwiseLayerImpl

CONSTRUCTION_LAYERS = {}

# ---------------- Specialized Construction Layers ----------------------------
# defined explicitly to provide improved auto-completion


def Input(out_shapes, name=None):
    return ConstructionWrapper.create('Input',
                                      name=name,
                                      out_shapes=out_shapes)


def FullyConnected(size, activation_function='rel', name=None):
    return ConstructionWrapper.create('FullyConnected',
                                      size=size,
                                      name=name,
                                      activation_function=activation_function)


def Highway(name=None):
    return ConstructionWrapper.create('Highway', name=name)


def Loss(importance=1.0, name=None):
    return ConstructionWrapper.create('Loss', importance=importance, name=name)


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


def Convolution2D(num_filters, kernel_size, stride=(1, 1), padding=0,
                  activation_function='rel', name=None):
    return ConstructionWrapper.create('Convolution2D',
                                      num_filters=num_filters,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=padding,
                                      activation_function=activation_function,
                                      name=name)


def Pooling2D(kernel_size, type='max', stride=(1, 1), padding=0, name=None):
    return ConstructionWrapper.create('Pooling2D',
                                      kernel_size=kernel_size,
                                      type=type,
                                      stride=stride,
                                      padding=padding,
                                      name=name)


def Dropout(drop_prob=0.5, name=None):
    return ConstructionWrapper.create('Dropout', drop_prob=drop_prob,
                                      name=name)


def BatchNorm(name=None, decay=0.9, epsilon=1.0e-5):
    return ConstructionWrapper.create('BatchNorm', name=name, decay=decay,
                                      epsilon=epsilon)


def Elementwise(activation_function='rel', name=None):
    return ConstructionWrapper.create('Elementwise',
                                      name=name,
                                      activation_function=activation_function)


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
