#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

import numpy as np
from brainstorm.describable import Describable
from brainstorm.training.schedules import get_schedule


# ########################### Base Class ######################################

class TrainingStep(Describable):
    """
    Base class for all training steps. Defines the common interface
    """
    __undescribed__ = {'net'}

    def __init__(self):
        self.net = None

    def start(self, net):
        self.net = net
        self._initialize()

    def _initialize(self):
        pass

    def run(self, input_data, targets):
        pass


# ########################## Training Steps ###################################

class DiagnosticStep(TrainingStep):
    """
    Only prints debugging information. Does not train at all.
    Use for diagnostics only.
    """
    def _initialize(self):
        print("start DiagnosticStep with net=", self.net)

    def run(self, input_data, targets):
        print("DiagnosticStep: x.shape=", input_data.shape)
        print("DiagnosticStep: t=", targets)
        return 15


class ForwardStep(TrainingStep):
    """
    Only runs the forward pass and returns the error. Does not train the
    network at all.
    This step is usually used for validation. If this step is used during
    training it should be initialized with the use_training_pass flag set to
    true.
    """

    def __init__(self, use_training_pass=False):
        super(ForwardStep, self).__init__()
        self.use_training_pass = use_training_pass

    def run(self, input_data, targets):
        self.net.forward_pass(input_data, training_pass=self.use_training_pass)
        return self.net.calculate_errors(targets)


class SgdStep(TrainingStep):
    """
    Stochastic Gradient Descent.
    """
    def __init__(self, learning_rate=0.1):
        super(SgdStep, self).__init__()
        self.learning_rate_schedule = get_schedule(learning_rate)

    def run(self, input_data, targets):
        learning_rate = self.learning_rate_schedule()
        self.net.forward_pass(input_data, training_pass=True)
        errors = self.net.calculate_errors(targets)
        self.net.backward_pass(targets)
        self.net.buffer.parameters[:] -= (
            learning_rate * self.net.buffer.gradient[:])
        return errors

    def __init_from_description__(self, description):
        self.learning_rate_schedule = get_schedule(self.learning_rate_schedule)


class MomentumStep(TrainingStep):
    """
    Stochastic Gradient Descent with a momentum term.
    learning_rate and momentum can be scheduled using
    brainstorm.training.schedules
    If scale_learning_rate is True (default),
    learning_rate is multiplied by (1 - momentum) when used.
    """
    __undescribed__ = {'velocity'}
    __default_values__ = {'scale_learning_rate': True}

    def __init__(self, learning_rate=0.1, momentum=0.0,
                 scale_learning_rate=True):
        super(MomentumStep, self).__init__()
        self.velocity = None
        self.momentum = get_schedule(momentum)
        self.learning_rate = get_schedule(learning_rate)
        assert isinstance(scale_learning_rate, bool), \
            "scale_learning_rate must be boolen"
        self.scale_learning_rate = scale_learning_rate

    def _initialize(self):
        self.velocity = np.zeros(self.net.get_param_size())

    def run(self, input_data, targets):
        learning_rate = self.learning_rate()
        momentum = self.momentum()
        self.velocity *= momentum
        self.net.forward_pass(input_data, training_pass=True)
        errors = self.net.calculate_errors(targets)
        self.net.backward_pass(targets)
        if self.scale_learning_rate:
            dv = (1 - momentum) * learning_rate * self.net.buffer.gradient[:]
        else:
            dv = learning_rate * self.net.buffer.gradient[:]

        self.velocity -= dv
        self.net.buffer.parameters[:] += self.velocity
        return errors

    def __init_from_description__(self, description):
        self.learning_rate = get_schedule(self.learning_rate)
        self.momentum = get_schedule(self.momentum)


class NesterovStep(MomentumStep):
    """
    Stochastic Gradient Descent with a Nesterov-style momentum term.
    learning_rate and momentum can be scheduled using
    brainstorm.training.schedules
    If scale_learning_rate is True (default),
    learning_rate is multiplied by (1 - momentum) when used.
    """
    def run(self, input_data, targets):
        learning_rate = self.learning_rate()
        momentum = self.momentum()
        self.velocity *= momentum
        self.net.buffer.parameters[:] += self.velocity
        self.net.forward_pass(input_data, training_pass=True)
        errors = self.net.calculate_errors(targets)
        self.net.backward_pass(targets)
        if self.scale_learning_rate:
            dv = (1 - momentum) * learning_rate * \
                self.net.buffer.gradient[:]
        else:
            dv = learning_rate * self.net.buffer.gradient[:]

        self.velocity -= dv
        self.net.buffer.parameters[:] -= dv
        return errors
