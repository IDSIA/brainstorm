#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np
import pytest
from brainstorm.data_iterators import progress_bar, Online, Undivided
from brainstorm.utils import IteratorValidationError
from brainstorm.handlers import default_handler


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
    targets = np.ones((2, 3, 1))
    with pytest.raises(IteratorValidationError):
        Iterator(my_data=None, my_targets=targets)

    with pytest.raises(IteratorValidationError):
        Iterator(something='input', targets=targets)

    with pytest.raises(IteratorValidationError):
        Iterator(default=[[[1]]], some_targets=targets)


@pytest.mark.parametrize('Iterator', [Undivided, Online])
def test_data_terator_with_wrong_targets_type_raises(Iterator):
    input_data = np.zeros((2, 3, 5))
    with pytest.raises(IteratorValidationError):
        Iterator(my_data=input_data, my_targets=[1])

    with pytest.raises(IteratorValidationError):
        Iterator(my_data=input_data, my_targets='2')

    with pytest.raises(IteratorValidationError):
        Iterator(my_data=input_data, my_target='3')


@pytest.mark.parametrize('Iterator', [Undivided, Online])
def test_data_terator_with_wrong_input_dim_raises(Iterator):
    targets = np.ones((2, 3, 1))
    with pytest.raises(IteratorValidationError):
        Iterator(my_data=np.zeros((2, 3)), my_targets=targets)


@pytest.mark.parametrize('Iterator', [Undivided, Online])
def test_data_terator_with_input_target_shape_mismatch_raises(Iterator):
    targets = np.ones((2, 3, 1))
    with pytest.raises(IteratorValidationError):
        Iterator(my_data=np.zeros((2, 5, 7)), my_targets=targets)


@pytest.mark.parametrize('Iterator', [Undivided, Online])
def test_data_terator_with_shape_mismatch_among_targets_raises(Iterator):
    targets1 = np.ones((2, 3, 1))
    targets2 = np.ones((2, 5, 1))
    with pytest.raises(IteratorValidationError):
        Iterator(my_data=np.zeros((2, 3, 7)),
                 targets1=targets1,
                 targets2=targets2)
    with pytest.raises(IteratorValidationError):
        Iterator(my_data=np.zeros((2, 5, 7)),
                 targets1=targets1,
                 targets2=targets2)


# ########################### Undivided #######################################

def test_undivided_default():
    input_data = np.zeros((2, 3, 5))
    targets = np.ones((2, 3, 1))
    iter = Undivided(my_data=input_data, my_targets=targets)(default_handler)
    x = next(iter)
    assert np.all(x['my_data'] == input_data)
    assert np.all(x['my_targets'] == targets)
    with pytest.raises(StopIteration):
        next(iter)


def test_undivided_named_targets():
    input_data = np.zeros((2, 3, 5))
    targets1 = np.ones((2, 3, 1))
    targets2 = np.ones((2, 3, 1))
    iter = Undivided(my_data=input_data,
                     targets1=targets1,
                     targets2=targets2)(default_handler)
    x = next(iter)
    assert np.all(x['my_data'] == input_data)
    assert np.all(x['targets1'] == targets1)
    assert np.all(x['targets2'] == targets2)

    with pytest.raises(StopIteration):
        next(iter)


# ########################### Online ##########################################

def test_online_default():
    input_data = np.zeros((4, 5, 3))
    targets = np.ones((4, 5, 1))
    it = Online(my_data=input_data,
                my_targets=targets,
                shuffle=False)(default_handler)
    x = next(it)
    assert set(x.keys()) == {'my_data', 'my_targets'}
    assert x['my_data'].shape == (4, 1, 3)
    assert x['my_targets'].shape == (4, 1, 1)
    x = next(it)
    assert set(x.keys()) == {'my_data', 'my_targets'}
    assert x['my_data'].shape == (4, 1, 3)
    assert x['my_targets'].shape == (4, 1, 1)
    x = next(it)
    assert set(x.keys()) == {'my_data', 'my_targets'}
    assert x['my_data'].shape == (4, 1, 3)
    assert x['my_targets'].shape == (4, 1, 1)
    x = next(it)
    assert set(x.keys()) == {'my_data', 'my_targets'}
    assert x['my_data'].shape == (4, 1, 3)
    assert x['my_targets'].shape == (4, 1, 1)
    x = next(it)
    assert set(x.keys()) == {'my_data', 'my_targets'}
    assert x['my_data'].shape == (4, 1, 3)
    assert x['my_targets'].shape == (4, 1, 1)
    with pytest.raises(StopIteration):
        next(it)
