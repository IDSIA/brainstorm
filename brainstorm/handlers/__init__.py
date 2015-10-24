#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function
from brainstorm.handlers.numpy_handler import NumpyHandler
from brainstorm.handlers.debug_handler import DebugHandler
from brainstorm.optional import has_pycuda, pycuda_mock, has_nervanagpu, \
    nervanagpu_mock
import numpy as np

if has_pycuda:
    from brainstorm.handlers.pycuda_handler import PyCudaHandler
else:
    PyCudaHandler = pycuda_mock

if has_nervanagpu:
    from brainstorm.handlers.nervanagpu_handler import NervanaGPUHandler
else:
    NervanaGPUHandler = nervanagpu_mock

default_handler = NumpyHandler(np.float32)

__all__ = ['NumpyHandler', 'PyCudaHandler', 'default_handler',
           'NervanaGPUHandler']
