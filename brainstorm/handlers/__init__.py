#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from brainstorm.handlers.numpy_handler import NumpyHandler
import numpy as np

try:
    from pycuda import gpuarray
    import scikits.cuda.linalg as culinalg
    import scikits.cuda.misc as cumisc
    from brainstorm.handlers.pycuda_handler import PyCudaHandler
except ImportError:
    pass

default_handler = NumpyHandler(np.float32)