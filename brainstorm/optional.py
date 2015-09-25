#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import warnings

try:
    import pycuda
    from pycuda import gpuarray, cumath
    import pycuda.driver as drv
    import pycuda.autoinit
    from pycuda.elementwise import ElementwiseKernel
    from pycuda.compiler import SourceModule
    from pycuda.curandom import XORWOWRandomNumberGenerator
    import skcuda.linalg as culinalg
    import skcuda.misc as cumisc
    has_pycuda = True
except ImportError:
    has_pycuda = False

has_cudnn = False
if has_pycuda:
    try:
        import ctypes
        import libcudnn as cudnn
        has_cudnn = True
    except ImportError:
        pass