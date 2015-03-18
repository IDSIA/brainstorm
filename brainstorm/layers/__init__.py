#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from functools import partial

from brainstorm.structure.construction import ConstructionWrapper

# somehow this construction is needed because in __all__ unicode does not work
__all__ = [str(a) for a in ['InputLayer', 'NoOpLayer', 'FeedForwardLayer']]

DataLayer = partial(ConstructionWrapper, "DataLayer")
InputLayer = partial(ConstructionWrapper, "DataLayer", kwargs={'data_name': "input_data"})
NoOpLayer = partial(ConstructionWrapper, "NoOpLayer")
FeedForwardLayer = partial(ConstructionWrapper, "FeedForwardLayer")

PyCudaFFLayer = partial(ConstructionWrapper, "PyCudaFFLayer")