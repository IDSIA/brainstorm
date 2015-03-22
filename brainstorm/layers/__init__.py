#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from functools import partial

from brainstorm.structure.construction import ConstructionWrapper
from brainstorm.layers.python_layers import *


# somehow this construction is needed because in __all__ unicode does not work
__all__ = [str(a) for a in ['InputLayer', 'NoOpLayer', 'FeedForwardLayer']]

InputLayer = partial(ConstructionWrapper.create, "InputLayer")
NoOpLayer = partial(ConstructionWrapper.create, "NoOpLayer")
FeedForwardLayer = partial(ConstructionWrapper.create, "FeedForwardLayer")

PyCudaFFLayer = partial(ConstructionWrapper.create, "PyCudaFFLayer")