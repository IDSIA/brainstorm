#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import sys
import six


class MissingDependencyMock(object):
    def __init__(self, error):
        self.error = error

    def __getattribute__(self, item):
        if item == "__doc__":
            return
        if item == "__class__":
            return MissingDependencyMock
        else:
            print("Attempted to access '{}' of a MissingDependencyMock object."
                  " Reraising the import error:".format(item), file=sys.stderr)
            six.reraise(*object.__getattribute__(self, 'error'))

    def __call__(self, *args, **kwargs):
        print("Attempted to call MissingDependencyMock object. "
              "Reraising the import error:", file=sys.stderr)
        six.reraise(*object.__getattribute__(self, 'error'))


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
    pycuda_mock = None
except ImportError as e:
    has_pycuda = False
    pycuda_mock = MissingDependencyMock(sys.exc_info())


try:
    import bokeh
    has_bokeh = True
except ImportError as e:
    has_bokeh = False
    bokeh_mock = MissingDependencyMock(sys.exc_info())


__all__ = ['has_pycuda', 'pycuda_mock', 'has_bokeh', 'bokeh_mock',
           'MissingDependencyMock']
