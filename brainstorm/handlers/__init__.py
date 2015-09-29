#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function
from brainstorm.handlers.numpy_handler import NumpyHandler
from brainstorm.handlers.debug_handler import DebugHandler
from brainstorm.optional import has_pycuda, has_nervanagpu
import numpy as np

if has_pycuda:
    from brainstorm.handlers.pycuda_handler import PyCudaHandler

if has_nervanagpu:
    from brainstorm.handlers.nervanagpu_handler import NervanaGPUHandler

default_handler = NumpyHandler(np.float32)

__all__ = ['NumpyHandler', 'PyCudaHandler', 'default_handler',
           'NervanaGPUHandler']
