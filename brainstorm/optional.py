#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

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

try:
    import pycuda
    import pycuda.autoinit
    from nervanagpu import NervanaGPU
    has_nervanagpu = True
except ImportError:
    has_nervanagpu = False

__all__ = ['has_pycuda', 'has_cudnn', 'has_nervanagpu']
