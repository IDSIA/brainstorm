#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np
from brainstorm.utils import ValidationError, ShapeValidationError


class ShapeTemplate(object):
    TYPES = ['TB', 'B', '',
             'TBF+', 'BF+', 'F+',
             'TBF*', 'BF*', 'F*', ]

    @staticmethod
    def from_tuple(shape):
        if isinstance(shape, ShapeTemplate):
            return shape
        elif not isinstance(shape, (tuple, list)):
            raise ShapeValidationError(
                'Requires tuple or list but got {}'.format(shape))
        else:
            return ShapeTemplate(*shape)

    # def __init__(self, *args, context_size=0):  # Not python2 compatible
    def __init__(self, *args, **kwargs):  # Not python2 compatible
        self._shape = args
        self.context_size = kwargs.get('context_size', 0)
        if kwargs and list(kwargs.keys()) != ['context_size']:
            raise TypeError('Unexpected keyword argument {}'
                            .format(list(kwargs.keys())))
        self.shape_type = self.get_shape_type()
        self.first_feature_dim = self.get_first_feature_dim()
        self.validate()

    @property
    def nr_dims(self):
        if self.shape_type.endswith('*'):
            raise TypeError('nr dimensions not fixed')
        return len(self._shape)

    @property
    def nr_feature_dims(self):
        if self.shape_type.endswith('*'):
            raise TypeError('nr dimensions not fixed')
        return len(self._shape[self.first_feature_dim:])

    @property
    def scaling_shape(self):
        return self._shape[:self.first_feature_dim]

    @property
    def feature_shape(self):
        return self._shape[self.first_feature_dim:]

    @property
    def feature_size(self):
        if self.shape_type.endswith('*') or self.shape_type.endswith('+'):
            raise TypeError('feature size not fixed')
        return int(np.prod(self.feature_shape))

    @property
    def scales_with_time(self):
        return 'T' in self.shape_type

    @property
    def scales_with_batch_size(self):
        return 'B' in self.shape_type

    def get_shape(self, t, b, *feature_dims):
        shape = ()
        if self.scales_with_time:
            shape += (t + self.context_size,)
        if self.scales_with_batch_size:
            shape += (b,)
        if 'F' in self.shape_type:
            shape += feature_dims
        shape += self._shape[len(shape):]
        return shape

    def get_shape_type(self):
        shape_type = ''
        if 'T' in self._shape:
            shape_type += 'T'
        if 'B' in self._shape:
            shape_type += 'B'
        if '...' in self._shape:
            shape_type += 'F*'
        elif 'F' in self._shape:
            shape_type += 'F+'
        return shape_type

    def get_first_feature_dim(self):
        if 'T' in self.shape_type:
            return 2
        elif 'B' in self.shape_type:
            return 1
        else:
            return 0

    def validate(self):
        if len(self._shape) == 0:
            raise ShapeValidationError("shape must be non-empty (nr dims > 0)")

        if self.scales_with_time and self._shape[:2] != ('T', 'B'):
            raise ShapeValidationError(
                "Shapes that scale with time need to start with ('T', 'B')"
                "(but started with {})".format(self._shape[:2]))

        if not self.scales_with_time and self.scales_with_batch_size and \
                self._shape[:1] != ('B',):
            raise ShapeValidationError(
                "Shapes that scale with batch-size need to start with 'B'"
                "(but started with {})".format(self._shape[:1]))

        # validate feature dimensions
        if len(self._shape) <= self.first_feature_dim:
            raise ShapeValidationError(
                "need at least one feature dimension"
                "(but shape was {})".format(self._shape))

        if self.shape_type.endswith('*'):
            if len(self._shape) > self.first_feature_dim + 1 or\
                    self.feature_shape != ('...',):
                raise ShapeValidationError(
                    'Wildcard-shapes can ONLY have a single feature dimension'
                    ' entry "...". (But had {})'.format(self.feature_shape))

        elif self.shape_type.endswith('+'):
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

        if self.context_size and not self.scales_with_time:
            raise ShapeValidationError("context_size is only available for "
                                       "shapes that scale with time.")

    def to_json(self):
        descr = {
            '@shape': self._shape
        }
        if self.context_size:
            descr['@context_size'] = self.context_size
        return descr

    def matches(self, shape):
        if not self.shape_type.endswith('*') and len(shape) != self.nr_dims:
            return False
        if self.shape_type.endswith('*') and len(shape) < len(self._shape):
            return False

        for s, t in zip(shape, self._shape):
            if t in ['T', 'B', 'F'] and isinstance(s, int):
                continue
            if t == '...':
                continue

            if t != s:
                return False
        return True

    def __getitem__(self, item):
        return self._shape.__getitem__(item)

    def __len__(self):
        return len(self._shape)

    def __eq__(self, other):
        if not isinstance(other, ShapeTemplate):
            return False
        return self._shape == other._shape

    def __hash__(self):
        return hash(self._shape)

    def __repr__(self):
        return "<{}>".format(self._shape)


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
    :param shapes: list of shape templates
    :type shapes: list[ShapeTemplates]
    :return: ShapeTemplate
    """
    if not shapes:
        return ShapeTemplate(0)
    dimensions = [s.nr_dims for s in shapes]
    if min(dimensions) != max(dimensions):
        raise ValueError('Dimensionality mismatch. {}'.format(shapes))
    shape_types = [s.shape_type for s in shapes]
    if min(shape_types) != max(shape_types):
        raise ValidationError('All buffer shapes need to have the same type. '
                              'But were: {}'.format(shape_types))
    some_shape = shapes[0]
    fixed_feature_shape = some_shape.feature_shape[1:]

    if not all([s.feature_shape[1:] == fixed_feature_shape for s in shapes]):
        raise ValueError('Feature size mismatch. {}'.format(shapes))

    summed_shape = sum(s.feature_shape[0] for s in shapes)
    final_shape = (some_shape.scaling_shape +
                   (summed_shape,) +
                   fixed_feature_shape)
    return ShapeTemplate(*final_shape)


def get_feature_size(shape):
    """Get the feature size of a shape-template."""
    buffer_type = validate_shape_template(shape)
    return int(np.array(shape[buffer_type:]).prod())
