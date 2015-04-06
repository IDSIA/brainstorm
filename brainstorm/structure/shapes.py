#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np
from brainstorm.utils import ValidationError


def ensure_tuple_or_none(a):
    if a is None:
        return a
    elif isinstance(a, tuple):
        return a
    elif isinstance(a, list):
        return tuple(a)
    else:
        return a,


def validate_shape_template(shape):
    if not isinstance(shape, tuple):
        raise ValidationError("shape must be of type tuple (but was {})"
                              .format(type(shape)))
    if len(shape) == 0:
        raise ValidationError("shape must be non-empty (length > 0)")

    if shape[0] == 'T':
        if len(shape) == 1 or shape[1] != 'B':
            raise ValidationError(
                "time-sized shapes need to have 'B' as second dimension "
                "(but was {})".format(shape[1]))
        first_feature_dim = 2
    elif shape[0] == 'B':
        first_feature_dim = 1
    else:
        first_feature_dim = 0

    if len(shape) <= first_feature_dim:
            raise ValidationError("need at least one feature dimension of "
                                  "type int (but shape was {}".format(shape))

    if not all([isinstance(s, int) for s in shape[first_feature_dim:]]):
        raise ValidationError("all feature dimensions need to be of type int!"
                              "(but shape was {})".format(shape))
    return first_feature_dim


def combine_input_shapes(shapes):
    """
    Concatenate the given sizes on the outermost feature dimension.
    Check that the other dimensions match.
    :param shapes: list of size-tuples or integers
    :type shapes: list[tuple[int]] or list[int]
    :return: tuple[int]
    """
    if not shapes:
        return 0,
    tupled_shapes = [ensure_tuple_or_none(s) for s in shapes]
    dimensions = [len(s) for s in tupled_shapes]
    if min(dimensions) != max(dimensions):
        raise ValueError('Dimensionality mismatch. {}'.format(tupled_shapes))
    first_feature_indices = [validate_shape_template(s) for s in tupled_shapes]
    some_shape = tupled_shapes[0]
    if min(first_feature_indices) != max(first_feature_indices):
        raise ValidationError('All buffer shapes need to have the same type. '
                              'But were: {}'.format(tupled_shapes))

    first_feature_index = first_feature_indices[0]
    fixed_feature_shape = some_shape[first_feature_index + 1:]

    if not all([s[first_feature_index + 1:] == fixed_feature_shape
                for s in tupled_shapes]):
        raise ValueError('Feature size mismatch. {}'.format(tupled_shapes))
    summed_shape = sum(s[first_feature_index] for s in tupled_shapes)
    return (some_shape[:first_feature_index] +
            (summed_shape,) +
            fixed_feature_shape)


def get_feature_size(shape):
    """Get the feature size of a shape-template."""
    buffer_type = validate_shape_template(shape)
    return int(np.array(shape[buffer_type:]).prod())
