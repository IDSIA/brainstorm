#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from functools import partial

from brainstorm.structure.construction import ConstructionWrapper
from brainstorm.layers.input_layer import InputLayer
from brainstorm.layers.noop_layer import NoOpLayer
from brainstorm.layers.fully_connected_layer import FullyConnectedLayer

# somehow this construction is needed because in __all__ unicode does not work
__all__ = [str(a) for a in ['InputLayer', 'NoOpLayer', 'FullyConnectedLayer']]

InputLayer = partial(ConstructionWrapper.create, "InputLayer")
NoOpLayer = partial(ConstructionWrapper.create, "NoOpLayer")
FullyConnectedLayer = partial(ConstructionWrapper.create, "FullyConnectedLayer")
