#!/usr/bin/env python
# coding=utf-8
"""
This module contains meta-information about the brainstorm package.

It is kept simple and separate from the main module, because this information
is also read by the setup.py. And during installation the brainstorm module
might not be importable yet.
"""
from __future__ import division, print_function, unicode_literals

__all__ = ("__version__", "__author__", "__url__", "__license__")

__version__ = "0.5"
__author__ = "The Swiss AI Lab IDSIA"
__url__ = "https://github.com/IDSIA/brainstorm"
__license__ = "MIT"
