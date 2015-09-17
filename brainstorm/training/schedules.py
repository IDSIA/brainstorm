#!/usr/bin/env python
# coding=utf-8
"""
Quantities like learning_rate and momentum in SgdStep, NesterovStep,
MomentumStep etc. can be changed according to schedules instead of being
constants.
Some common schedulers are provided for convenience.
"""

from __future__ import division, print_function, unicode_literals
import numpy as np
from brainstorm.describable import Describable


def get_schedule(schedule_or_const):
    if callable(schedule_or_const):
        return schedule_or_const
    else:
        return Constant(schedule_or_const)


class Constant(Describable):
    """
    Returns a constant value at every update.
    """
    def __init__(self, value):
        self.value = value

    def __call__(self):
        return self.value

    def __describe__(self):
        return self.value


class Linear(Describable):
    """
    Changes a quantity linearly from 'initial_value' to 'final_value'.

    A linear change to the quantity is made every 'interval' number of updates.
    Step size is decided by 'num_changes'.

    For example:
    Linear(0.9, 1.0, 10, 100)
    will change the quantity starting from 0.9 to 1.0 by incrementing it
    after every 100 updates such that 10 increments are made.
    """
    __undescribed__ = {
        'current_value': None,
        'update_number': 0
    }

    def __init__(self, initial_value, final_value, num_changes, interval):
        """

        :param initial_value: Value returned before the first change.
        :type initial_value: float
        :param final_value: Value returned after the last change.
        :type final_value: float
        :param num_changes: Total number of changes to be made according to a
        linear schedule.
        :type num_changes: int
        :param interval: Number of updates to wait before making each change.
        :type interval: int
        """
        self.initial_value = initial_value
        self.final_value = final_value
        self.interval = interval
        self.num_changes = num_changes
        self.update_number = 0  # initial_value should be used for first update
        self.current_value = None

    def __call__(self):
        if (self.update_number // self.interval) > self.num_changes:
            self.current_value = self.final_value
        else:
            self.current_value = self.initial_value + \
                                 ((
                                      self.final_value - self.initial_value) / self.num_changes) * \
                                 (self.update_number // self.interval)
        self.update_number += 1
        return self.current_value


class Exponential(Describable):
    """
    Changes a quantity exponentially starting from 'initial_value' by a
    factor.

    The quantity is multiplied by the given 'factor' every 'interval' number of
    updates. Once the quantity goes below 'minimum' or above 'maximum',
    it is fixed to the corresponding limit.

    For example:
    Exponential(1.0, 0.9, 100) will change the quantity starting from
    1.0 by a factor of 0.9 after every 100 updates.
    """
    __undescribed__ = {
        'current_value': None,
        'update_number': 0
    }

    def __init__(self, initial_value=0, factor=1, interval=1, minimum=-np.Inf,
                 maximum=np.Inf):
        """

        :param initial_value: Value returned before the first change.
        :type initial_value: float
        :param factor: Multiplication factor.
        :type factor: float
        :param interval: Number of updates to wait before making each change.
        :type interval: int
        :param minimum: Lower bound of the quantity.
        :type minimum: float
        :param maximum: Upper bound of the quantity.
        :type maximum: float
        """
        self.initial_value = initial_value
        self.factor = factor
        self.interval = interval
        self.minimum = minimum
        self.maximum = maximum
        self.update_number = 0
        self.current_value = None

    def __call__(self):
        self.current_value = min(self.maximum, max(
            self.minimum,
            self.initial_value * (self.factor **
                                  (self.update_number // self.interval))))
        self.update_number += 1
        return self.current_value


class MultiStep(Describable):
    """
    A schedule for switching values after a series of specified number of
    updates.

    For example:
    MultiStep(1.0, [100, 110, 120], [0.1, 0.01, 0.001]) will return 1.0 for
    the first 100 updates, 0.1 for the next 10, 0.01 for the next 10,
    and 0.001 for every following update.
    """
    __undescribed__ = {
        'current_value': None,
        'update_number': 0,
        'step_number': 0
    }

    def __init__(self, initial_value, steps, values):
        """

        :param initial_value: Initial value of the parameter
        :type initial_value: float
        :param steps: List of update numbers at which the values are switched.
        :type steps: list[int]
        :param values: List of values to set after specified update numbers.
        :type values: list[float]
        """
        assert len(steps) == len(values)
        self.initial_value = initial_value
        self.steps = steps
        self.values = values
        self.update_number = 0
        self.step_number = 0
        self.current_value = None

    def __call__(self):
        if self.step_number < len(self.steps) and \
           self.update_number == self.steps[self.step_number]:
            self.step_number += 1
        self.current_value = self.initial_value if self.step_number == 0 \
            else self.values[self.step_number - 1]
        self.update_number += 1
        return self.current_value
