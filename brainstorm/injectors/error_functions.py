#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np
from brainstorm.injectors.core import Injector


class MeanSquaredError(Injector):
    @staticmethod
    def _framewise(outputs, targets):
        diff = outputs - targets.data
        if targets.mask is not None:
            diff *= targets.mask
        norm = outputs.shape[1]  # normalize by number of sequences
        error = 0.5 * np.sum(diff ** 2) / norm
        return error, (diff / norm)
