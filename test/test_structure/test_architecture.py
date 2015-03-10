#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import pytest
from brainstorm.structure.architecture import combine_input_sizes


def test_combine_input_sizes_int():
    assert combine_input_sizes([7]) == (7,)
    assert combine_input_sizes([1, 2, 3, 4]) == (10,)


def test_combine_input_sizes_int_and_unituples():
    assert combine_input_sizes([1, 2, (3,), (4,), 5]) == (15,)

def test_combine_input_sizes_tuples():
    assert combine_input_sizes([(1, 4)]) == (1, 4)

    assert combine_input_sizes([(1, 4),
                                (3, 4),
                                (6, 4)]) == (10, 4)

    assert combine_input_sizes([(2, 3, 4),
                                (3, 3, 4),
                                (2, 3, 4)]) == (7, 3, 4)


@pytest.mark.parametrize('sizes', [
    [2, (1, 2)],
    [(2, 3), (2, 2)],
    [(2,), (1, 2)],
    [(2, 1, 3), (3, 1, 3), (2, 2, 3)],
    [(2, 1, 3), (3, 1, 3), (1, 1, 2)]
])
def test_combine_input_sizes_mismatch(sizes):
    with pytest.raises(ValueError):
        combine_input_sizes(sizes)
