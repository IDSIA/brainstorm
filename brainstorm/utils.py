#!/usr/bin/env python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import re


PYTHON_IDENTIFIER = re.compile("^[_a-zA-Z][_a-zA-Z0-9]*$")

NAME_BLACKLIST = {'default', 'fallback'}


def is_valid_layer_name(identifier):
    if identifier in NAME_BLACKLIST:
        return False
    return PYTHON_IDENTIFIER.match(identifier) is not None


class ValidationError(Exception):
    pass


class NetworkValidationError(ValidationError):
    """
    Exception that is thrown if attempting to build an invalid architecture.
    (E.g. circle)
    """


class LayerValidationError(ValidationError):
    pass


def get_inheritors(cls):
    """
    Get a set of all classes that inherit from the given class.
    """
    subclasses = set()
    work = [cls]
    while work:
        parent = work.pop()
        for child in parent.__subclasses__():
            if child not in subclasses:
                subclasses.add(child)
                work.append(child)
    return subclasses
