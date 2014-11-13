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
