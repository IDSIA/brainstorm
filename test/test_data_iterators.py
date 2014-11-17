#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np
import pytest
from brainstorm.data_iterators import progress_bar, Online, Undivided
from brainstorm.targets import FramewiseTargets


# ########################### Progress Bar ####################################
def test_progress_bar():
    prefix = '<<'
    bar = '1234567890'
    suffix = '>>'
    p = progress_bar(10, prefix, bar, suffix)
    assert next(p) == prefix
    assert p.send(4) == '1234'
    assert p.send(4) == ''
    assert p.send(9) == '56789'
    assert p.send(9.999) == ''
    assert p.send(10) == '0' + suffix


# ########################### All Data Iterators ##############################

@pytest.mark.parametrize('Iterator', [Undivided, Online])
def test_data_terator_with_wrong_input_type_raises(Iterator):
    targets = FramewiseTargets(np.ones((2, 3, 1)))
    with pytest.raises(AssertionError):
        Iterator(None, targets)

    with pytest.raises(AssertionError):
        Iterator('input', targets)

    with pytest.raises(AssertionError):
        Iterator([[[1]]], targets)


@pytest.mark.parametrize('Iterator', [Undivided, Online])
def test_data_terator_with_wrong_targets_type_raises(Iterator):
    input_data = np.zeros((2, 3, 5))
    with pytest.raises(AssertionError):
        Iterator(input_data, [1])

    with pytest.raises(AssertionError):
        Iterator(input_data, '2')

    with pytest.raises(AssertionError):
        Iterator(input_data, my_target='3')


@pytest.mark.parametrize('Iterator', [Undivided, Online])
def test_data_terator_with_wrong_input_dim_raises(Iterator):
    targets = FramewiseTargets(np.ones((2, 3, 1)))
    with pytest.raises(AssertionError):
        Iterator(np.zeros((2, 3)), targets)

    with pytest.raises(AssertionError):
        Iterator(np.zeros((2, 3, 5, 7)), targets)


@pytest.mark.parametrize('Iterator', [Undivided, Online])
def test_data_terator_with_input_target_shape_mismatch_raises(Iterator):
    targets = FramewiseTargets(np.ones((2, 3, 1)))
    with pytest.raises(AssertionError):
        Iterator(np.zeros((2, 5, 7)), targets)


@pytest.mark.parametrize('Iterator', [Undivided, Online])
def test_data_terator_with_shape_mismatch_among_targets_raises(Iterator):
    targets1 = FramewiseTargets(np.ones((2, 3, 1)))
    targets2 = FramewiseTargets(np.ones((2, 5, 1)))
    with pytest.raises(AssertionError):
        Iterator(np.zeros((2, 3, 7)), targets1=targets1, targets2=targets2)
    with pytest.raises(AssertionError):
        Iterator(np.zeros((2, 5, 7)), targets1=targets1, targets2=targets2)


# ########################### Undivided #######################################

def test_undivided_default():
    input_data = np.zeros((2, 3, 5))
    targets = FramewiseTargets(np.ones((2, 3, 1)))
    iter = Undivided(input_data, targets)()
    x, t = next(iter)
    assert x is input_data
    assert t == {'default': targets}
    with pytest.raises(StopIteration):
        next(iter)


def test_undivided_named_targets():
    input_data = np.zeros((2, 3, 5))
    targets1 = FramewiseTargets(np.ones((2, 3, 1)))
    targets2 = FramewiseTargets(np.ones((2, 3, 1)))
    iter = Undivided(input_data, targets1=targets1, targets2=targets2)()
    x, t = next(iter)
    assert x is input_data
    assert t == {'targets1': targets1, 'targets2': targets2}
    with pytest.raises(StopIteration):
        next(iter)
