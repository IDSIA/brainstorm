#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import pytest
from brainstorm.structure.shapes import ShapeTemplate
from brainstorm.utils import ShapeValidationError


shapes_1 = [
    (1,),
    (1, 2, 3),
    ('B', 1),
    ('B', 1, 2, 3),
    ('T', 'B', 1),
    ('T', 'B', 1, 2, 3),
]

nr_dims_1 = [1, 3, 2, 4, 3, 5]
nr_feature_dims_1 = [1, 3, 1, 3, 1, 3]
feature_dims_1 = [(1,), (1, 2, 3), (1,), (1, 2, 3), (1,), (1, 2, 3)]
feature_size_1 = [1, 6, 1, 6, 1, 6]
scales_with_time_1 = [False, False, False, False, True, True]
scales_with_batch_size_1 = [False, False, True, True, True, True]


@pytest.mark.parametrize('shape, nr_dims', zip(shapes_1, nr_dims_1))
def test_shape_templates1_nr_dims(shape, nr_dims):
    st = ShapeTemplate(*shape)
    assert st.nr_dims == nr_dims


@pytest.mark.parametrize('shape, nr_feature_dims',
                         zip(shapes_1, nr_feature_dims_1))
def test_shape_templates1_nr_feature_dims(shape, nr_feature_dims):
    st = ShapeTemplate(*shape)
    assert st.nr_feature_dims == nr_feature_dims


@pytest.mark.parametrize('shape, feature_dims',
                         zip(shapes_1, feature_dims_1))
def test_shape_templates1_nr_feature_dims(shape, feature_dims):
    st = ShapeTemplate(*shape)
    assert st.feature_shape == feature_dims


@pytest.mark.parametrize('shape, feature_size', zip(shapes_1, feature_size_1))
def test_shape_templates1_feature_size(shape, feature_size):
    st = ShapeTemplate(*shape)
    assert st.feature_size == feature_size


@pytest.mark.parametrize('shape, scales_t', zip(shapes_1, scales_with_time_1))
def test_shape_templates1_scales_with_time(shape, scales_t):
    st = ShapeTemplate(*shape)
    assert st.scales_with_time == scales_t


@pytest.mark.parametrize('shape, scales_b',
                         zip(shapes_1, scales_with_batch_size_1))
def test_shape_templates1_scales_with_time(shape, scales_b):
    st = ShapeTemplate(*shape)
    assert st.scales_with_batch_size == scales_b


shapes_2 = [
    ('F',),
    ('F', 'F', 'F'),
    ('B', 'F'),
    ('B', 'F', 'F', 'F'),
    ('T', 'B', 'F'),
    ('T', 'B', 'F', 'F', 'F')]

nr_dims_2 = [1, 3, 2, 4, 3, 5]
nr_feature_dims_2 = [1, 3, 1, 3, 1, 3]
scales_with_time_2 = [False, False, False, False, True, True]
scales_with_batch_size_2 = [False, False, True, True, True, True]


@pytest.mark.parametrize('shape, nr_dims', zip(shapes_2, nr_dims_2))
def test_shape_templates2_nr_dims(shape, nr_dims):
    st = ShapeTemplate(*shape)
    assert st.nr_dims == nr_dims


@pytest.mark.parametrize('shape, nr_feature_dims',
                         zip(shapes_2, nr_feature_dims_2))
def test_shape_templates2_nr_feature_dims(shape, nr_feature_dims):
    st = ShapeTemplate(*shape)
    assert st.nr_feature_dims == nr_feature_dims


@pytest.mark.parametrize('shape', shapes_2)
def test_shape_templates2_feature_size(shape):
    st = ShapeTemplate(*shape)
    with pytest.raises(TypeError):
        _ = st.feature_size


@pytest.mark.parametrize('shape, scales_t', zip(shapes_2, scales_with_time_2))
def test_shape_templates2_scales_with_time(shape, scales_t):
    st = ShapeTemplate(*shape)
    assert st.scales_with_time == scales_t


@pytest.mark.parametrize('shape, scales_b',
                         zip(shapes_2, scales_with_batch_size_2))
def test_shape_templates2_scales_with_time(shape, scales_b):
    st = ShapeTemplate(*shape)
    assert st.scales_with_batch_size == scales_b


shapes_3 = [
    ('...',),
    ('B', '...'),
    ('T', 'B', '...')]

scales_with_time_3 = [False, False, True]
scales_with_batch_size_3 = [False, True, True]


@pytest.mark.parametrize('shape', shapes_3)
def test_shape_templates3_nr_dims(shape):
    st = ShapeTemplate(*shape)
    with pytest.raises(TypeError):
        _ = st.nr_dims


@pytest.mark.parametrize('shape', shapes_3)
def test_shape_templates3_nr_feature_dims(shape):
    st = ShapeTemplate(*shape)
    with pytest.raises(TypeError):
        _ = st.nr_feature_dims


@pytest.mark.parametrize('shape', shapes_3)
def test_shape_templates3_feature_size(shape):
    st = ShapeTemplate(*shape)
    with pytest.raises(TypeError):
        _ = st.feature_size


@pytest.mark.parametrize('shape, scales_t', zip(shapes_3, scales_with_time_3))
def test_shape_templates3_scales_with_time(shape, scales_t):
    st = ShapeTemplate(*shape)
    assert st.scales_with_time == scales_t


@pytest.mark.parametrize('shape, scales_b',
                         zip(shapes_3, scales_with_batch_size_3))
def test_shape_templates3_scales_with_time(shape, scales_b):
    st = ShapeTemplate(*shape)
    assert st.scales_with_batch_size == scales_b


illegal_shapes = [
    (),
    ('T',),
    ('T', 2),
    ('B', 'T', 1),
    ('T', 1, 'B', 1),
    ('T', 1, 2),
    (1, 'B', 2),
    (1, 'T', 'B', 2),
    ('T', 'B', 'T', 2),
    ('T', 'F'),
    ('T', 'B', 'F', 2),
    ('T', 'B', 1, 'F'),
    ('B', 1, 'F'),
    ('B', 'F', 2),
    ('F', 2),
    (1, 'F'),
    ('...', 2),
    (1, '...'),
    ('F', '...'),
    ('...', 'F'),
    ('B', 1, '...'),
    ('B', 'F', '...'),
    ('T', 'B', 1, '...'),
    ('T', 'B', 'F', '...'),
]


@pytest.mark.parametrize('shape', illegal_shapes)
def test_illegal_shape_template_raise(shape):
    with pytest.raises(ShapeValidationError):
        st = ShapeTemplate(*shape)


@pytest.mark.parametrize('shape, expected', [
    [('T', 'B', 1, 3), True],
    [('T', 7, 1, 3), True],
    [('T', 4, 1, 3), True],
    [(1, 'B', 1, 3), True],
    [(2, 4, 1, 3), True],
    [('T', 'B', 1), False],
    [('T', 'B', 'F', 3), False],
    [('B', 1, 3, 4), False],
    [('B', 'F', 1, 2), False],
    [(), False],
])
def test_shape_template_matches1(shape, expected):
    st = ShapeTemplate('T', 'B', 1, 3)
    assert st.matches(shape) == expected


@pytest.mark.parametrize('shape, expected', [
    [('T', 'B', 'F', 'F'), True],
    [('T', 'B', 1, 3), True],
    [('T', 7, 2, 2), True],
    [('T', 4, 1, 3), True],
    [(1, 'B', 1, 3), True],
    [(2, 4, 1, 3), True],
    [('T', 'B', 1), False],
    [('T', 'B', 'F', 3), True],
    [('B', 1, 3, 4), False],
    [('B', 'F', 1, 2), False],
    [(), False],
])
def test_shape_template_matches2(shape, expected):
    st = ShapeTemplate('T', 'B', 'F', 'F')
    assert st.matches(shape) == expected
    try:
        assert st.matches(ShapeTemplate(*shape)) == expected
    except ShapeValidationError:
        pass


@pytest.mark.parametrize('shape, expected', [
    [('T', 'B', 'F', 'F'), True],
    [('T', 'B', 11), True],
    [('T', 7, 2, 2), True],
    [('T', 4, 1, 3, 2, 1, 2, 8), True],
    [(1, 'B', 1, 3), True],
    [(2, 4, 1, 3), True],
    [('T', 'B'), False],
    [('T', 'B', 'F', 3), True],
    [('B', 1, 3, 4), False],
    [('B', 'F', 1, 2), False],
    [(), False],
])
def test_shape_template_matches3(shape, expected):
    st = ShapeTemplate('T', 'B', '...')
    assert st.matches(shape) == expected
    try:
        assert st.matches(ShapeTemplate(*shape)) == expected
    except ShapeValidationError:
        pass


@pytest.mark.parametrize('shape, expected', [
    [(1, 2, 7), True],
    [('T', 'B', 11), False],
    [('T', 4, 1, 3, 2, 1, 2, 8), False],
    [(2, 4, 1), False],
    [('T', 'B'), False],
    [('T', 'B', 'F'), False],
    [('F', 'F', 'F'), False],
    [(), False],
])
def test_shape_template_matches3(shape, expected):
    st = ShapeTemplate(1, 2, 7)
    assert st.matches(shape) == expected
    try:
        assert st.matches(ShapeTemplate(*shape)) == expected
    except ShapeValidationError:
        pass