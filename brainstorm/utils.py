#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import re


PYTHON_IDENTIFIER = re.compile("^[_a-zA-Z][_a-zA-Z0-9]*$")


def is_valid_python_identifier(identifier):
    return PYTHON_IDENTIFIER.match(identifier) is not None


class InvalidArchitectureError(Exception):
    """
    Exception that is thrown if attempting to build an invalid architecture.
    (E.g. circle)
    """
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