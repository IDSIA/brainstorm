#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np
import pytest
from brainstorm.targets import (
    FramewiseTargets, LabelingTargets, SequencewiseTargets)


# ################### All Targets #############################################
target_types = ('TargetsClass, targets_data',
                [(FramewiseTargets, np.zeros((4, 5, 1))),
                 (LabelingTargets, [[1]]*5),
                 (SequencewiseTargets, np.zeros((1, 5, 1)))])


@pytest.mark.parametrize(*target_types)
def test_targets_with_wrong_mask_shape_raises(TargetsClass, targets_data):
    with pytest.raises(AssertionError):
        TargetsClass(targets_data, mask=np.zeros((1, 1)))

    with pytest.raises(AssertionError):
        TargetsClass(targets_data, mask=np.zeros((1, 1, 2)))

    with pytest.raises(AssertionError):
        TargetsClass(targets_data, mask=np.zeros((1, 2, 1)))


@pytest.mark.parametrize(*target_types)
def test_targets_with_mask_compute_seq_lengths(TargetsClass, targets_data):
    mask = np.array([[1, 1, 1, 0],
                     [1, 1, 0, 0],
                     [1, 0, 0, 0],
                     [1, 1, 1, 1],
                     [1, 1, 1, 0]
                     ]).T.reshape(4, 5, 1)
    t = TargetsClass(targets_data, mask=mask)
    assert np.all(t.sequence_lengths == np.array([3, 2, 1, 4, 3]))


@pytest.mark.parametrize(*target_types)
def test_targets_trim_mask(TargetsClass, targets_data):
    mask = np.ones((4, 5, 1))
    mask[2:, :, :] = 0
    tar = TargetsClass(targets_data, mask=mask, binarize_to=2)
    assert tar.shape == (4, 5, 2)
    tar.trim(3)
    assert tar.shape == (3, 5, 2)
    assert tar.mask.shape == (3, 5, 1)


@pytest.mark.parametrize(*target_types)
def test_targets_trim_shorter_than_seq_len_raises(TargetsClass, targets_data):
    mask = np.ones((4, 5, 1))
    mask[3:, :, :] = 0
    tar = TargetsClass(targets_data, mask=mask, binarize_to=2)
    with pytest.raises(AssertionError):
        tar.trim(1)


@pytest.mark.parametrize(*target_types)
def test_targets_get_item_with_unsupported_index_raises(TargetsClass,
                                                        targets_data):
    tar = TargetsClass(targets_data)
    with pytest.raises(ValueError):
        _ = tar['a']

    with pytest.raises(AssertionError):
        _ = tar[1, 2, 3]

    with pytest.raises(ValueError):
        _ = tar[[1, 2, 3]]

    with pytest.raises(ValueError):
        _ = tar[::2]

# ################### Framewise ###############################################

def test_framewisetargets_with_wrong_targets_dim_raises():
    with pytest.raises(AssertionError):
        FramewiseTargets(np.zeros((1, 1)))

    with pytest.raises(AssertionError):
        FramewiseTargets(np.zeros((1, 1, 1, 1)))


def test_framewisetargets_with_wrong_mask_length_raises():
    with pytest.raises(AssertionError):
        FramewiseTargets(np.zeros((3, 5, 2)), mask=np.zeros((4, 5, 1)))


def test_framewisetargets_binarizing_with_feature_dim_raises():
    with pytest.raises(AssertionError):
        FramewiseTargets(np.zeros((3, 5, 2)), binarize_to=2)


def test_framewisetargets_shape():
    t = FramewiseTargets(np.zeros((2, 3, 5)))
    assert t.shape == (2, 3, 5)


def test_framewisetargets_binarizing_shape():
    t = FramewiseTargets(np.zeros((2, 3, 1)), binarize_to=5)
    assert t.shape == (2, 3, 5)


def test_framewisetargets_without_mask_sequence_lenghts():
    t = FramewiseTargets(np.zeros((5, 3, 2)))
    assert np.all(t.sequence_lengths == np.array([5, 5, 5]))


def test_framewisetargets_get_item():
    mask = np.array([[1, 1, 1, 0],
                     [1, 1, 0, 0],
                     [1, 1, 0, 0],
                     [1, 1, 1, 1],
                     [1, 1, 1, 0]
                     ]).T.reshape(4, 5, 1)
    t = FramewiseTargets(np.zeros((4, 5, 1)), mask=mask, binarize_to=3)
    s = t[1:3, 2:]
    assert s.binarize_to == 3
    assert np.all(s.mask == mask[1:3, 2:])
    assert np.all(s.sequence_lengths == np.array([1, 2, 2]))
    assert np.all(s.data == np.zeros((2, 3, 1)))


def test_framewisetargets_get_item_negative_indexing():
    mask = np.array([[1, 1, 1, 0],
                     [1, 1, 0, 0],
                     [1, 1, 0, 0],
                     [1, 1, 1, 1],
                     [1, 1, 1, 0]
                     ]).T.reshape(4, 5, 1)
    t = FramewiseTargets(np.zeros((4, 5, 1)), mask=mask, binarize_to=3)
    s = t[-3:-1, -3:]
    assert s.binarize_to == 3
    assert np.all(s.mask == mask[1:3, 2:])
    assert np.all(s.sequence_lengths == np.array([1, 2, 2]))
    assert np.all(s.data == np.zeros((2, 3, 1)))


def test_framewisetargets_trim_mask():
    mask = np.ones((4, 5, 1))
    mask[2:, :, :] = 0
    tar = FramewiseTargets(np.zeros((4, 5, 1)), mask=mask, binarize_to=2)
    tar.trim(3)
    assert tar.data.shape == (3, 5, 1)


# ################### Labelling ##############################################

def test_labellingtargets_shape():
    t = LabelingTargets([[np.zeros(2)], [np.zeros(2)], [np.zeros(2)]])
    assert t.shape == (None, 3, 2)


def test_labellingtargets_binarizing_shape():
    t = LabelingTargets([[0], [1], [2]], binarize_to=5)
    assert t.shape == (None, 3, 5)


def test_labellingtargets_masked_shape():
    t = LabelingTargets([[0], [1]], binarize_to=5, mask=np.zeros((7, 2, 1)))
    assert t.shape == (7, 2, 5)


def test_labellingtargets_without_mask_sequence_lenghts():
    t = LabelingTargets([[1]]*3, binarize_to=2)
    assert np.all(t.sequence_lengths == np.array([0, 0, 0]))


def test_labellingtargets_get_item():
    mask = np.array([[1, 1, 1, 0],
                     [1, 1, 0, 0],
                     [1, 1, 0, 0],
                     [1, 1, 1, 1],
                     [1, 1, 1, 0]
                     ]).T.reshape(4, 5, 1)
    t = LabelingTargets([[1]]*5, mask=mask, binarize_to=3)
    s = t[1:3, 2:]
    assert s.binarize_to == 3
    assert np.all(s.mask == mask[1:3, 2:])
    assert np.all(s.sequence_lengths == np.array([1, 2, 2]))
    assert s.data == [[1], [1], [1]]


def test_labellingtargets_get_item_negative_indexing():
    mask = np.array([[1, 1, 1, 0],
                     [1, 1, 0, 0],
                     [1, 1, 0, 0],
                     [1, 1, 1, 1],
                     [1, 1, 1, 0]
                     ]).T.reshape(4, 5, 1)
    t = LabelingTargets([[1]]*5, mask=mask, binarize_to=3)
    s = t[-3:-1, -3:]
    assert s.binarize_to == 3
    assert np.all(s.mask == mask[-3:-1, -3:])
    assert np.all(s.sequence_lengths == np.array([1, 2, 2]))
    assert s.data == [[1], [1], [1]]


# ################### Sequencewise ###########################################

def test_sequencewisetargets_with_wrong_targets_dim_raises():
    with pytest.raises(AssertionError):
        SequencewiseTargets(np.zeros((1, 1)))

    with pytest.raises(AssertionError):
        SequencewiseTargets(np.zeros((1, 1, 1, 1)))


def test_sequencewisetargets_with_length_raises():
    with pytest.raises(AssertionError):
        SequencewiseTargets(np.zeros((3, 5, 2)))


def test_sequencewisetargets_binarizing_with_feature_dim_raises():
    with pytest.raises(AssertionError):
        SequencewiseTargets(np.zeros((3, 5, 2)), binarize_to=2)


def test_sequencewisetargets_shape():
    t = SequencewiseTargets(np.zeros((1, 3, 5)))
    assert t.shape == (1, 3, 5)


def test_sequencewisetargets_binarizing_shape():
    t = SequencewiseTargets(np.zeros((1, 3, 1)), binarize_to=5)
    assert t.shape == (1, 3, 5)


def test_sequencewisetargets_masked_shape():
    t = SequencewiseTargets(np.zeros((1, 3, 1)), mask=np.ones((7, 3, 1)),
                            binarize_to=5)
    assert t.shape == (7, 3, 5)


def test_sequencewisetargets_without_mask_sequence_lenghts():
    t = SequencewiseTargets(np.zeros((1, 3, 2)))
    assert np.all(t.sequence_lengths == np.array([0, 0, 0]))


def test_sequencewisetargets_get_item():
    mask = np.array([[1, 1, 1, 0],
                     [1, 1, 0, 0],
                     [1, 1, 0, 0],
                     [1, 1, 1, 1],
                     [1, 1, 1, 0]
                     ]).T.reshape(4, 5, 1)
    t = SequencewiseTargets(np.zeros((1, 5, 1)), mask=mask, binarize_to=3)
    s = t[1:3, 2:]
    assert s.binarize_to == 3
    assert np.all(s.mask == mask[1:3, 2:])
    assert np.all(s.sequence_lengths == np.array([1, 2, 2]))
    assert np.all(s.data == np.zeros((1, 3, 1)))


def test_sequencewisetargets_get_item_negative_indexing():
    mask = np.array([[1, 1, 1, 0],
                     [1, 1, 0, 0],
                     [1, 1, 0, 0],
                     [1, 1, 1, 1],
                     [1, 1, 1, 0]
                     ]).T.reshape(4, 5, 1)
    t = SequencewiseTargets(np.zeros((1, 5, 1)), mask=mask, binarize_to=3)
    s = t[-3:-1, -3:]
    assert s.binarize_to == 3
    assert np.all(s.mask == mask[1:3, 2:])
    assert np.all(s.sequence_lengths == np.array([1, 2, 2]))
    assert np.all(s.data == np.zeros((1, 3, 1)))
