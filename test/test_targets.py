#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np
import pytest
from brainstorm.targets import (
    FramewiseTargets, LabelingTargets, SequencewiseTargets)


@pytest.mark.parametrize('TargetsClass, targets_data',
                         [(FramewiseTargets, np.zeros((1, 1, 1))),
                          (LabelingTargets, [[1]]),
                          (SequencewiseTargets, np.zeros((1, 1, 1)))])
def test_targets_with_wrong_mask_shape_raises(TargetsClass, targets_data):
    with pytest.raises(AssertionError):
        TargetsClass(targets_data, mask=np.zeros((1, 1)))

    with pytest.raises(AssertionError):
        TargetsClass(targets_data, mask=np.zeros((1, 1, 2)))

    with pytest.raises(AssertionError):
        TargetsClass(targets_data, mask=np.zeros((1, 2, 1)))


@pytest.mark.parametrize('TargetsClass, targets_data',
                         [(FramewiseTargets, np.zeros((4, 5, 1))),
                          (LabelingTargets, [[1]]*5),
                          (SequencewiseTargets, np.zeros((1, 5, 1)))])
def test_targets_with_mask_compute_seq_lengths(TargetsClass, targets_data):
    mask = np.array([[1, 1, 1, 0],
                     [1, 1, 0, 0],
                     [1, 0, 0, 0],
                     [1, 1, 1, 1],
                     [1, 1, 1, 0],
                     ]).T.reshape(4, 5, 1)
    t = TargetsClass(targets_data, mask=mask)
    assert np.all(t.sequence_lengths == np.array([3, 2, 1, 4, 3]))


# ################### Framewise ##############################################

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


def test_sequencewisetargets_without_mask_sequence_lenghts():
    t = SequencewiseTargets(np.zeros((1, 3, 2)))
    assert np.all(t.sequence_lengths == np.array([0, 0, 0]))