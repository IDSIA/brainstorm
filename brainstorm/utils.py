#!/usr/bin/env python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
from datetime import datetime
from functools import reduce  # Valid in Python 2.6+, required in Python 3
import math
import operator
import re

from brainstorm.__about__ import __version__


product = lambda arr: reduce(operator.mul, arr, 1)


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


class StructureValidationError(ValidationError):
    pass


class InitializationError(Exception):
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


def get_by_path(d, path):
    """Access an element of the dict d using the (possibly dotted) path.

    For example, 'foo.bar.baz' would return d['foo']['bar']['baz'].

    Args:
        d (dict):
            (nested) dictionary
        path (str):
            path to access the dictionary

    Returns:
        the value corresponding to d[p1][p2]...

    Raises:
        KeyError:
            if any key along the path was not found.
    """
    current_node = d
    for p in path.split('.'):
        try:
            current_node = current_node[p]
        except KeyError:
            raise KeyError(
                'Path "{}" could not be resolved. Key "{}" missing. Available '
                'keys are: [{}]'.format(
                    path, p, ", ".join(sorted(current_node.keys()))))
    return current_node


def get_normalized_path(*args):
    path_parts = []
    for a in args:
        assert '@' not in a, a
        path_parts.extend(a.replace('..', '@').split('.'))

    assert '' not in path_parts, str(path_parts)

    normalized_parts = []
    for p in path_parts:
        while p.startswith('@'):
            normalized_parts.pop()
            p = p[1:]
        normalized_parts.append(p)
    return ".".join(normalized_parts)


def flatten_time(array):
    assert len(array.shape) >= 3, "Time can be flattened only for arrays "\
                                  "with at least 3 dimensions."
    t, b, f = array.shape[0], array.shape[1], array.shape[2:]
    return array.reshape((t * b,) + f)


def flatten_time_and_features(array):
    assert len(array.shape) >= 3, "Time & features can be flattened only "\
                                  "for arrays with at least 3 dimensions."
    t, b, f = array.shape[0], array.shape[1], array.shape[2:]
    return array.reshape((t * b, int(product(f))))


def flatten_features(array, start_idx=2):
    return array.reshape(array.shape[:start_idx] +
                         (int(product(array.shape[start_idx:])),))


def flatten_all_but_last(array):
    return array.reshape((int(product(array.shape[:-1])), array.shape[-1]))


def flatten_keys(dictionary):
    """
    Flattens the keys for a nested dictionary using dot notation. This
    returns all the keys which can be accessed via `get_by_path`.

    Example:
        For example, {'a': None, 'b': {'x': None}} would return ['a', 'b.x']

    Args:
        dictionary (dict):
            A dictionary which should be flattened.

    Returns:
        list[str]: list of flattened keys
    """
    if not isinstance(dictionary, dict):
        return []
    keys = []
    for k, v in dictionary.items():
        if isinstance(v, dict):
            for x in flatten_keys(v):
                keys.append(k + '.' + x)
        else:
            keys.append(k)
    return keys


def progress_bar(maximum, prefix='[',
                 bar='====1====2====3====4====5====6====7====8====9====0',
                 suffix='] Took: {0}\n'):
    i = 0
    start_time = datetime.utcnow()
    out = prefix
    while i < len(bar):
        progress = yield out
        j = math.trunc(progress / maximum * len(bar))
        out = bar[i: j]
        i = j
    elapsed_str = str(datetime.utcnow() - start_time)[: -5]
    yield out + suffix.format(elapsed_str)


def silence():
    while True:
        yield ''


def get_brainstorm_info():
    info = 'Created with brainstorm {}'.format(__version__)
    return info.encode()
