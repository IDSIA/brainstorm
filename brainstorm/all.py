#!/usr/bin/env python
# coding=utf-8
"""
Convenience namespace containing all relevant brainstorm objects and functions.
"""
from __future__ import division, print_function, unicode_literals

from brainstorm.describable import get_description, create_from_description
from brainstorm.randomness import global_rnd
from brainstorm.structure import Network, generate_architecture

from brainstorm.layers import *
from brainstorm.hooks import *
from brainstorm.training.steppers import *
from brainstorm.handlers import *
from brainstorm.tools import *
from brainstorm.initializers import *
from brainstorm.value_modifiers import *
from brainstorm.data_iterators import *
