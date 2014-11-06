#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from functools import partial

from brainstorm.structure.construction import ConstructionLayer


InputLayer = partial(ConstructionLayer, "InputLayer")
NoOpLayer = partial(ConstructionLayer, "NoOpLayer")
FeedForwardLayer = partial(ConstructionLayer, "FeedForwardLayer")
