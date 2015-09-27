#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from brainstorm.describable import Describable
import numpy as np


class Scorer(Describable):
    def __init__(self, out_name='', targets_name='targets', mask_name='',
                 name=None):
        self.out_name = out_name
        self.targets_name = targets_name
        self.mask_name = mask_name
        self.__name__ = name if name is not None else self.__class__.__name__

    def __call__(self, true_labels, predicted, mask=None):
        pass

    @staticmethod
    def aggregate(errors):
        errors = np.array(errors)
        assert errors.ndim == 2 and errors.shape[1] == 2
        return np.sum(errors[:, 1]) / np.sum(errors[:, 0])


class Accuracy(Scorer):
    def __call__(self, true_labels, predicted, mask=None):
        if predicted.shape[1] > 1:
            predicted = predicted.argmax(1).reshape(-1, 1)
        correct = predicted == true_labels
        if mask:
            correct *= mask
        return np.sum(correct)


class Hamming(Scorer):
    def __init__(self, threshold=0.5, out_name='', targets_name='targets',
                 mask_name='', name=None):
        super(Hamming, self).__init__(out_name, targets_name, mask_name, name)
        self.threshold = threshold

    def __call__(self, true_labels, predicted, mask=None):
        correct = np.logical_xor(predicted < self.threshold, true_labels)
        if mask is not None:
            correct *= mask
        return np.sum(correct)


class MeanSquaredError(Scorer):
    def __call__(self, true_labels, predicted, mask=None):
        errors = (true_labels - predicted) ** 2
        if mask is not None:
            errors *= mask
        return 0.5 * np.sum(errors)

