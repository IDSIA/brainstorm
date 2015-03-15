#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np
from brainstorm.targets import FramewiseTargets

#TODO: FIXME


def test_mean_squared_error():
    out = np.arange(0, 3).reshape((3, 1, 1))
    target = FramewiseTargets(np.zeros((3, 1, 1)))
    expected_err = 2.5  # 0.5 * (0 + 1 + 4)
    expected_deltas = np.array([0, 1, 2]).reshape((3, 1, 1))
    mse = MeanSquaredError('foo')
    err, deltas = mse(out, target)
    assert err == expected_err
    assert np.all(expected_deltas == deltas)


def test_mean_squared_error_normalizes_by_nr_sequences():
    out = np.arange(0, 3).reshape((1, 3, 1))
    target = FramewiseTargets(np.zeros((1, 3, 1)))
    expected_err = 2.5 / 3  # 0.5 * (0 + 1 + 4) / 3
    expected_deltas = np.array([0, 1, 2]).reshape((1, 3, 1)) / 3
    mse = MeanSquaredError('foo')
    err, deltas = mse(out, target)
    assert err == expected_err
    assert np.all(expected_deltas == deltas)


def test_mean_squared_error_masked():
    out = np.arange(0, 3).reshape((3, 1, 1))
    mask = np.array([1, 1, 0]).reshape((3, 1, 1))
    target = FramewiseTargets(np.zeros((3, 1, 1)), mask=mask)
    expected_err = 0.5  # 0.5 * (0 + 1 + 0)
    expected_deltas = np.array([0, 1, 0]).reshape((3, 1, 1))
    mse = MeanSquaredError('foo')
    err, deltas = mse(out, target)
    assert err == expected_err
    assert np.all(expected_deltas == deltas)
