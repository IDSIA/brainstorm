#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

import pytest

from brainstorm.structure.buffer_structure import (BufferStructure,
                                                   StructureTemplate,
                                                   combine_buffer_structures)
from brainstorm.utils import StructureValidationError

# ############################ BufferStructures ############################# #

structures = [
    (1,),
    (1, 2, 3),
    ('B', 1),
    ('B', 1, 2, 3),
    ('T', 'B', 1),
    ('T', 'B', 1, 2, 3),
]

nr_dims_1 = [1, 3, 2, 4, 3, 5]
feature_dims_1 = [(1,), (1, 2, 3), (1,), (1, 2, 3), (1,), (1, 2, 3)]
feature_size_1 = [1, 6, 1, 6, 1, 6]
scales_with_time_1 = [False, False, False, False, True, True]
scales_with_batch_size_1 = [False, False, True, True, True, True]


@pytest.mark.parametrize('shape, nr_dims', zip(structures, nr_dims_1))
def test_buffer_structures_nr_dims(shape, nr_dims):
    st = BufferStructure(*shape)
    assert st.nr_dims == nr_dims


@pytest.mark.parametrize('shape, feature_dims',
                         zip(structures, feature_dims_1))
def test_buffer_structures_nr_feature_dims(shape, feature_dims):
    st = BufferStructure(*shape)
    assert st.feature_shape == feature_dims


@pytest.mark.parametrize('shape, feature_size', zip(structures,
                                                    feature_size_1))
def test_buffer_structures_feature_size(shape, feature_size):
    st = BufferStructure(*shape)
    assert st.feature_size == feature_size


@pytest.mark.parametrize('shape, scales_t', zip(structures,
                                                scales_with_time_1))
def test_buffer_structures_scales_with_time(shape, scales_t):
    st = BufferStructure(*shape)
    assert st.scales_with_time == scales_t


@pytest.mark.parametrize('shape, scales_b',
                         zip(structures, scales_with_batch_size_1))
def test_buffer_structures_scales_with_time(shape, scales_b):
    st = BufferStructure(*shape)
    assert st.scales_with_batch_size == scales_b

illegal_structures = [
    (),
    ('T',),
    ('T', 2),
    ('B', 'T', 1),
    ('T', 1, 'B', 1),
    ('T', 1, 2),
    (1, 'B', 2),
    (1, 'T', 'B', 2),
    ('T', 'B', 'T', 2),
]


@pytest.mark.parametrize('shape', illegal_structures)
def test_illegal_buffer_structure_raise(shape):
    with pytest.raises(StructureValidationError):
        BufferStructure(*shape)

# ########################### StructureTemplates ############################ #

illegal_templates = illegal_structures + [
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


@pytest.mark.parametrize('shape', illegal_templates)
def test_illegal_structure_template_raise(shape):
    with pytest.raises(StructureValidationError):
        StructureTemplate(*shape)


@pytest.mark.parametrize('shape, expected', [
    [('T', 'B', 1, 3), True],
    [('T', 'B', 1), False],
    [('T', 'B', 3), False],
    [('B', 4, 1, 3), False],
    [(1, 3), False],
    [(1,), False],
])
def test_structure_template_matches1(shape, expected):
    st = StructureTemplate('T', 'B', 1, 3)
    assert st.matches(BufferStructure(*shape)) == expected


@pytest.mark.parametrize('shape, expected', [
    [('T', 'B', 1, 3), True],
    [('T', 'B', 4, 7), True],
    [('T', 'B', 1, 1), True],
    [('T', 'B', 1), False],
    [('T', 'B', 3), False],
    [('T', 'B', 1, 2, 3), False],
    [('B', 1, 2, 4), False],
    [(1, 3, 1, 3), False],
    [(1, 3), False],
    [(1,), False],
])
def test_structure_template_matches2(shape, expected):
    st = StructureTemplate('T', 'B', 'F', 'F')
    struct = BufferStructure(*shape)
    assert st.matches(struct) == expected


@pytest.mark.parametrize('shape, expected', [
    [('T', 'B', 11), True],
    [('T', 'B', 1, 2), True],
    [('T', 'B', 1, 2, 3, 4, 5), True],
    [('B', 1), False],
    [('B', 1, 3), False],
    [('B', 1, 3, 4), False],
    [(1, 3, 1, 3), False],
    [(1, 3), False],
    [(1,), False],
])
def test_structure_template_matches3(shape, expected):
    st = StructureTemplate('T', 'B', '...')
    struct = BufferStructure(*shape)
    assert st.matches(struct) == expected


@pytest.mark.parametrize('shape, expected', [
    [(1, 2, 7), True],
    [(1, 2, 7, 4), False],
    [('T', 'B', 7), False],
    [('B', 2, 7), False],
    [('B', 2), False],
    [(2, 2, 7), False],
    [(1, 3, 7), False],
    [(1, 2, 8), False],
    [(1, 2), False],
    [(1,), False],

])
def test_structure_template_matches4(shape, expected):
    st = StructureTemplate(1, 2, 7)
    struct = BufferStructure(*shape)
    assert st.matches(struct) == expected


def test_combine_input_sizes_tuples():
    assert combine_buffer_structures([BufferStructure(1, 4)]) == \
        BufferStructure(1, 4)

    assert combine_buffer_structures([BufferStructure(4, 1),
                                      BufferStructure(4, 3),
                                      BufferStructure(4, 6)])\
        == BufferStructure(4, 10)

    assert combine_buffer_structures([BufferStructure(4, 3, 2),
                                      BufferStructure(4, 3, 3),
                                      BufferStructure(4, 3, 2)]) == \
        BufferStructure(4, 3, 7)


def test_combine_input_sizes_tuple_templates():
    assert (combine_buffer_structures([BufferStructure('B', 4)]) ==
            BufferStructure('B', 4))
    assert (combine_buffer_structures([BufferStructure('B', 4),
                                       BufferStructure('B', 3)]) ==
            BufferStructure('B', 7))
    assert (combine_buffer_structures([BufferStructure('T', 'B', 4)]) ==
            BufferStructure('T', 'B', 4))
    assert (combine_buffer_structures([BufferStructure('T', 'B', 4),
                                       BufferStructure('T', 'B', 3)]) ==
            BufferStructure('T', 'B', 7))
    assert (combine_buffer_structures([BufferStructure('T', 'B', 3, 2, 4),
                                       BufferStructure('T', 'B', 3, 2, 3)]) ==
            BufferStructure('T', 'B', 3, 2, 7))


@pytest.mark.parametrize('sizes', [
    [BufferStructure(3, 2), BufferStructure(2, 2)],
    [BufferStructure(2), BufferStructure(1, 2)],
    [BufferStructure(2, 1, 3), BufferStructure(2, 1, 3),
     BufferStructure(2, 2, 3)],
    [BufferStructure(2, 1, 3), BufferStructure(2, 1, 1),
     BufferStructure(1, 1, 3)]
])
def test_combine_input_sizes_mismatch(sizes):
    with pytest.raises(ValueError):
        combine_buffer_structures(sizes)
