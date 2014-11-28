#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from functools import partial

from brainstorm.structure.construction import ConstructionLayer

# somehow this construction is needed because in __all__ unicode does not work
__all__ = [str(a) for a in ['InputLayer', 'NoOpLayer', 'FeedForwardLayer']]

InputLayer = partial(ConstructionLayer, "InputLayer")
NoOpLayer = partial(ConstructionLayer, "NoOpLayer")
FeedForwardLayer = partial(ConstructionLayer, "FeedForwardLayer")

PyCudaFFLayer = partial(ConstructionLayer, "PyCudaFFLayer")