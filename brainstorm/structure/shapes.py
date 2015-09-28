#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np
from brainstorm.utils import ValidationError, ShapeValidationError


class StructureTemplate(object):
    # The following signature unfortunately is not python2 compatible:
    # def __init__(self, *args, context_size=0, backward_only=False):
    def __init__(self, *args, **kwargs):
        self.shape = args
        self.context_size = kwargs.get('context_size', 0)
        expected_kwargs = {'context_size', 'is_backward_only'}
        if not set(kwargs.keys()) <= expected_kwargs:
            raise TypeError('Unexpected keyword argument {}'
                            .format(set(kwargs.keys()) - expected_kwargs))

        if 'T' in self.shape:
            self.first_feature_dim = 2
        elif 'B' in self.shape:
            self.first_feature_dim = 1
        else:
            self.first_feature_dim = 0

        self.is_backward_only = kwargs.get('is_backward_only', False)
        self.validate()

    @property
    def feature_shape(self):
        return self.shape[self.first_feature_dim:]

    def validate(self):
        if len(self.shape) == 0:
            raise ShapeValidationError("shape must be non-empty (nr dims > 0)")

        if 'T' in self.shape and self.shape[:2] != ('T', 'B'):
            raise ShapeValidationError(
                "Shapes that scale with time need to start with ('T', 'B')"
                "(but started with {})".format(self.shape[:2]))

        if 'T' not in self.shape and 'B' in self.shape and \
                self.shape[:1] != ('B',):
            raise ShapeValidationError(
                "Shapes that scale with batch-size need to start with 'B'"
                "(but started with {})".format(self.shape[:1]))

        # validate feature dimensions
        if len(self.shape) < self.first_feature_dim:
            raise ShapeValidationError(
                "need at least one feature dimension"
                "(but shape was {})".format(self.shape))

        if '...' in self.shape:
            if self.feature_shape != ('...',):
                raise ShapeValidationError(
                    'Wildcard-shapes can ONLY have a single feature dimension'
                    ' entry "...". (But had {})'.format(self.feature_shape))

        elif 'F' in self.shape:
            # TODO: Is this condition necessary?
            if not all([f == 'F' for f in self.feature_shape]):
                raise ShapeValidationError(
                    'The feature dimensions of shapes with feature templates '
                    '("F") have to consist only of "F"s. (But was {})'
                    .format(self.feature_shape))
        else:
            if not all([isinstance(f, int) for f in self.feature_shape]):
                raise ShapeValidationError(
                    'The feature dimensions have to be all-integer. But was {}'
                    .format(self.feature_shape))

        # validate context_size
        if not isinstance(self.context_size, int) or self.context_size < 0:
            raise ShapeValidationError(
                "context_size has to be a non-negative integer, but was {}"
                .format(self.context_size))

        if self.context_size and 'T' not in self.shape:
            raise ShapeValidationError("context_size is only available for "
                                       "shapes that scale with time.")

    def matches(self, shape):
        assert isinstance(shape, BufferStructure)

        if '...' not in self.shape and shape.nr_dims != len(self.shape):
            return False
        if '...' in self.shape and shape.nr_dims < len(self.shape):
            return False

        for s, t in zip(shape.shape, self.shape):
            if t == s:
                continue
            if t == 'F' and isinstance(s, int):
                continue
            if t == '...':
                continue
            if t != s:
                return False
        return True

    def __repr__(self):
        return "<<<{}>>>".format(self.shape)


class BufferStructure(object):
    @classmethod
    def from_tuple(cls, shape):
        return cls(*shape)

    # The following signature unfortunately is not python2 compatible:
    # def __init__(self, *args, context_size=0, backward_only=False):
    def __init__(self, *args, **kwargs):
        expected_kwargs = {'context_size', 'is_backward_only'}
        if not set(kwargs.keys()) <= expected_kwargs:
            raise TypeError('Unexpected keyword argument {}'
                            .format(set(kwargs.keys()) - expected_kwargs))

        self.shape = args
        self.context_size = kwargs.get('context_size', 0)
        self.is_backward_only = kwargs.get('is_backward_only', False)

        if 'T' in self.shape:
            self.buffer_type = 2
        elif 'B' in self.shape:
            self.buffer_type = 1
        else:
            self.buffer_type = 0
        self.first_feature_dim = self.buffer_type

        self.validate()

    @property
    def scales_with_time(self):
        return 'T' in self.shape

    @property
    def scales_with_batch_size(self):
        return 'B' in self.shape

    @property
    def scaling_shape(self):
        return self.shape[:self.first_feature_dim]

    @property
    def feature_shape(self):
        return self.shape[self.first_feature_dim:]

    @property
    def feature_size(self):
        return int(np.prod(self.feature_shape))

    @property
    def nr_dims(self):
        return len(self.shape)

    def validate(self):
        if len(self.shape) == 0:
            raise ShapeValidationError("shape must be non-empty (nr dims > 0)")

        if self.scales_with_time and self.shape[:2] != ('T', 'B'):
            raise ShapeValidationError(
                "Shapes that scale with time need to start with ('T', 'B')"
                "(but started with {})".format(self.shape[:2]))

        if not self.scales_with_time and self.scales_with_batch_size and \
                self.shape[:1] != ('B',):
            raise ShapeValidationError(
                "Shapes that scale with batch-size need to start with 'B'"
                "(but started with {})".format(self.shape[:1]))

        # validate feature dimensions
        if len(self.shape) <= self.first_feature_dim:
            raise ShapeValidationError(
                "need at least one feature dimension"
                "(but shape was {})".format(self.shape))

        if not all([isinstance(f, int) for f in self.feature_shape]):
                raise ShapeValidationError(
                    'The feature dimensions have to be all-integer. But was {}'
                    .format(self.feature_shape))

        # validate context_size
        if not isinstance(self.context_size, int) or self.context_size < 0:
            raise ShapeValidationError(
                "context_size has to be a non-negative integer, but was {}"
                .format(self.context_size))

        if self.context_size and not self.scales_with_time:
            raise ShapeValidationError("context_size is only available for "
                                       "shapes that scale with time.")

    def to_json(self, i):
        descr = {
            '@shape': self.shape,
            '@index': i,
            '@type': 'array'
        }
        if self.context_size:
            descr['@context_size'] = self.context_size
        if self.is_backward_only:
            descr['@is_backward_only'] = True
        return descr

    def __eq__(self, other):
        if not isinstance(other, BufferStructure):
            return False
        return self.shape == other.shape

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self.shape)

    def __repr__(self):
        return "<{}>".format(self.shape)


def combine_buffer_structures(shapes):
    """
    Concatenate buffer structures on the last feature dimension.

    Checks that all other dimensions match.

    Args:
        shapes (list[BufferStructure]):
            list of BufferStructures to concatenate

    Returns:
        BufferStructure: The combined BufferStructure
    """
    if not shapes:
        return BufferStructure(0)
    for s in shapes:
        assert isinstance(s, BufferStructure)

    dimensions = [s.nr_dims for s in shapes]
    if min(dimensions) != max(dimensions):
        raise ValueError('Dimensionality mismatch. {}'.format(shapes))
    shape_types = [s.buffer_type for s in shapes]
    if min(shape_types) != max(shape_types):
        raise ValidationError('All buffer shapes need to have the same type. '
                              'But were: {}'.format(shape_types))
    some_shape = shapes[0]
    fixed_feature_shape = some_shape.feature_shape[:-1]

    if not all([s.feature_shape[:-1] == fixed_feature_shape for s in shapes]):
        raise ValueError('Feature size mismatch. {}'.format(shapes))

    summed_shape = sum(s.feature_shape[-1] for s in shapes)
    final_shape = (some_shape.scaling_shape +
                   (summed_shape,) +
                   fixed_feature_shape)
    return BufferStructure(*final_shape)
