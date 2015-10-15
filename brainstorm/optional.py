#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals


class MissingDependencyMock(object):
    def __init__(self, depends_on):
        self.depends_on = depends_on

    def __getattribute__(self, item):
        raise ImportError('Depends on missing "{}" package.'
                          .format(object.__getattribute__(self, 'depends_on')))

    def __call__(self, *args, **kwargs):
        raise ImportError('Depends on missing "{}" package.'
                          .format(object.__getattribute__(self, 'depends_on')))


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
    import bokeh
    has_bokeh = True
except ImportError:
    has_bokeh = False


__all__ = ['has_pycuda', 'has_cudnn', 'has_bokeh', 'MissingDependencyMock']
