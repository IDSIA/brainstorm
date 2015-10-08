#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

import numpy as np
import pytest

from brainstorm.scorers import Accuracy, Hamming, MeanSquaredError


def accuracy():
    true_labels = np.array([0, 3, 1, 2, 3, 2, 0, 1]).reshape(-1, 1)
    predictions = np.eye(4)[[0, 1, 1, 2, 3, 3, 1, 2]]
    scorer = Accuracy()
    expected = 0.5
    return scorer, true_labels, predictions, expected


def hamming():
    true_labels = np.array([[0, 1, 1],
                            [0, 1, 0],
                            [0, 0, 0],
                            [1, 1, 0]])
    predictions = np.array([[0.1, 0.7, 0.4],
                            [0.9, 0.3, .1],
                            [0.8, 0.2, 0.1],
                            [0.5, 0.4, 0.2]])  # 7 / 12 correct

    scorer = Hamming(threshold=0.5)
    expected = 7./12.
    return scorer, true_labels, predictions, expected


def mean_squared_error():
    true_labels = np.array([0, 1, 2, 3, 4, 5]).reshape(-1, 1)
    predictions = np.array([2, 1, 2, 3.5, 4, 4]).reshape(-1, 1)
    scorer = MeanSquaredError()
    expected = 0.5 * (4 + 0 + 0 + 0.25 + 0 + 1) / 6
    return scorer, true_labels, predictions, expected


scorers_to_test = [
    accuracy,
    hamming,
    mean_squared_error
]

scorer_ids = [s.__name__ for s in scorers_to_test]


@pytest.fixture(params=scorers_to_test, ids=scorer_ids)
def scorer_test_case(request):
    return request.param()


@pytest.mark.parametrize('batch_size', [1, 2, 3, 100])
def test_scorer_with_different_batchsizes(scorer_test_case, batch_size):
    scorer, true_labels, predictions, expected = scorer_test_case
    errors = []
    for i in range(0, true_labels.shape[0], batch_size):
        t = true_labels[i:i+batch_size]
        p = predictions[i:i+batch_size]
        errors.append((t.shape[0], scorer(t, p)))
    assert np.allclose(scorer.aggregate(errors), expected)


def test_scorer_with_mask(scorer_test_case):
    scorer, true_labels, predictions, expected = scorer_test_case
    errors1 = [(3, scorer(true_labels[:3], predictions[:3]))]
    mask = np.zeros((true_labels.shape[0], 1))
    mask[:3] = 1.0
    errors2 = [(3, scorer(true_labels, predictions, mask=mask))]

    assert scorer.aggregate(errors1) == scorer.aggregate(errors2)
