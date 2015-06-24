#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function
from brainstorm.handlers.numpy_handler import NumpyHandler
from brainstorm.handlers.debug_handler import DebugHandler

import numpy as np

try:
    from brainstorm.handlers.pycuda_handler import PyCudaHandler
except ImportError as e:
    print(e)

default_handler = NumpyHandler(np.float32)

__all__ = ['NumpyHandler', 'PyCudaHandler', 'default_handler']
