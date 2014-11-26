#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np
from brainstorm.injectors.core import Injector


# ###################### Proper Error Functions for training ##################
class MeanSquaredError(Injector):
    @staticmethod
    def _framewise(outputs, targets):
        diff = outputs - targets.data
        if targets.mask is not None:
            diff *= targets.mask
        norm = outputs.shape[1]  # normalize by number of sequences
        error = 0.5 * np.sum(diff ** 2) / norm
        return error, (diff / norm)


# #################### Errors for Monitoring ##################################

class ClassificationError(Injector):
    @staticmethod
    def _framewise(outputs, targets):
        y_win = outputs.argmax(2)
        t_win = targets.argmax(2)
        if targets.mask is not None:
            errors = np.sum((y_win != t_win) * targets.mask[:, :, 0])
            total = np.sum(targets.mask)
        else:
            errors = np.sum((y_win != t_win))
            total = targets.shape[0] * targets.shape[1]
        return (errors, total), None

    @staticmethod
    def aggregate(errors):
        e = np.sum(errors, axis=0)
        return np.round(e[0] * 100. / e[1], 2)