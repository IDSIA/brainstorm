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
    Exception that is thrown e.g. if attempting to build an invalid
    architecture. (E.g. circle)
    """


class LayerValidationError(ValidationError):
    pass


class IteratorValidationError(ValidationError):
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


def flatten(container):
    """Iterate nested lists in flat order."""
    for i in container:
        if isinstance(i, (list, tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i


def convert_to_nested_indices(container, start_idx=None):
    """Return nested lists of indices with same structure as container."""
    if start_idx is None:
        start_idx = [0]
    for i in container:
        if isinstance(i, (list, tuple)):
            yield list(convert_to_nested_indices(i, start_idx))
        else:
            yield start_idx[0]
            start_idx[0] += 1


def sort_by_index_key(x):
    """
    Used as key in sorted() to order items of a dictionary by the @index
    entries in its child-dicts if present.
    """
    if isinstance(x[1], dict) and '@index' in x[1]:
        return x[1]['@index']
    else:
        return -1