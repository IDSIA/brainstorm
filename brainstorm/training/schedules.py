#!/usr/bin/env python
# coding=utf-8
"""
Quantities like learning_rate and momentum in steppers (SgdStep, NesterovStep,
MomentumStep etc.) can be changed according to schedules instead of being
constants. The schedules are added through
brainstorm.hooks.ModifyStepperAttribute.

The call signature for all schedules must be the same in order to be usable.

Note that the Trainer counts the current_epoch_nr and current_update_nr in
natural numbers (starting from 1) and they are initialized to 0. Then it
a) calls the update hooks (and then the epoch hooks if needed)
b) increments the update number (and epoch number if an epoch ends)
c) runs the stepper
and repeats in this order.

This means that when a schedule object is called, the epoch_nr argument is 0
for the first epoch,1 for the second epoch and so on, and similarly for
update_nr.

Some common schedulers are provided for convenience.
"""

from __future__ import division, print_function, unicode_literals

import numpy as np

from brainstorm.describable import Describable


class Linear(Describable):
    """
    Changes a quantity linearly from 'initial_value' to 'final_value'.

    Step size is decided by 'num_changes' every 'interval' number of
    epochs or updates as specified.

    For example:
    Linear(0.9, 1.0, 10)
    will change the quantity starting from 0.9 to 1.0 by incrementing it
    such that 10 increments are made.
    """

    def __init__(self, initial_value, final_value, num_changes):
        """
        Args:
            initial_value (float):
                Value returned before the first change.
            final_value (float):
                Value returned after the last change.
            num_changes (int):
                Total number of changes to be made according to a
                linear schedule.
        """
        self.initial_value = initial_value
        self.final_value = final_value
        self.num_changes = num_changes

    def __call__(self, epoch_nr, update_nr, timescale, interval,
                 net, stepper, logs):
        current = epoch_nr if timescale == 'epoch' else update_nr
        if (current // interval) > self.num_changes:
            return self.final_value
        else:
            new_value = self.initial_value + \
                ((self.final_value - self.initial_value) / self.num_changes) \
                * (current // interval)
        return new_value


class Exponential(Describable):
    """
    Changes a quantity exponentially starting from 'initial_value' by a
    factor.

    The quantity is multiplied by the given 'factor' every 'interval' number of
    epochs or updates as specified. Once the quantity goes below 'minimum' or
    above 'maximum', it is fixed to the corresponding limit.

    For example:
    Exponential(1.0, 0.9) will change the quantity starting from
    1.0 by a factor of 0.9.
    """

    def __init__(self, initial_value, factor, minimum=-np.Inf, maximum=np.Inf):
        """
        Args:
            initial_value (float):
                Initial value of the parameter
            factor (float):
                Multiplication factor.
            minimum (float):
                Lower bound of the quantity.
            maximum (float):
                Upper bound of the quantity.
        """
        self.initial_value = initial_value
        self.factor = factor
        self.minimum = minimum
        self.maximum = maximum

    def __call__(self, epoch_nr, update_nr, timescale, interval,
                 net, stepper, logs):
        current = epoch_nr if timescale == 'epoch' else update_nr
        new_value = self.initial_value * (self.factor ** (current // interval))
        new_value = min(self.maximum, max(self.minimum, new_value))
        return new_value


class MultiStep(Describable):
    """
    A schedule for switching values after a series of specified number of
    updates or epochs as specified.

    For example:
    MultiStep(1.0, [100, 110, 120], [0.1, 0.01, 0.001]) will return 1.0 for
    the first 100 updates/epochs, 0.1 for the next 10, 0.01 for the next 10,
    and 0.001 for every following update/epoch.
    """

    def __init__(self, initial_value, steps, values):
        """
        Args:
            initial_value (float):
                Initial value of the parameter
            steps (list[int]):
                List of update/epoch numbers at which the values are switched.
            values (list[float]):
                List of values to set after specified update numbers.
        """
        assert len(steps) == len(values)
        self.initial_value = initial_value
        self.steps = [0] + steps + [np.inf]
        self.values = [self.initial_value] + values

    def __call__(self, epoch_nr, update_nr, timescale, interval,
                 net, stepper, logs):
        assert interval == 1, "MultiStep schedule only supports unit intervals"
        current = epoch_nr if timescale == 'epoch' else update_nr
        step_number = 0
        for i in range(1, len(self.steps)):
            if self.steps[i - 1] <= current < self.steps[i]:
                step_number = i - 1
                break
        return self.values[step_number]
