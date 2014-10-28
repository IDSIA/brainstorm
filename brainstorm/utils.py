#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import re


PYTHON_IDENTIFIER = re.compile("^[_a-zA-Z][_a-zA-Z0-9]*$")


class InvalidArchitectureError(Exception):
    """
    Exception that is thrown if attempting to build an invalid architecture.
    (E.g. circle)
    """
    pass