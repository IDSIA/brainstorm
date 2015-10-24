#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

from brainstorm.describable import Describable


# ########################### Base Class ######################################

class TrainingStepper(Describable):
    """
    Base class for all training steps. Defines the common interface
    """
    __undescribed__ = {'net'}

    def __init__(self):
        self.net = None

    def start(self, net):
        self.net = net

    def run(self):
        pass


# ########################## Training Steps ###################################

class ForwardStepper(TrainingStepper):
    """
    Only runs the forward pass and returns the error. Does not train the
    network at all.
    This step is usually used for validation. If this step is used during
    training it should be initialized with the use_training_pass flag set to
    true.
    """

    def __init__(self, use_training_pass=False):
        super(ForwardStepper, self).__init__()
        self.use_training_pass = use_training_pass

    def run(self):
        self.net.forward_pass(training_pass=self.use_training_pass)
        return self.net.get_loss_value()


class SgdStepper(TrainingStepper):
    """
    Stochastic Gradient Descent.
    """
    __undescribed__ = {'update'}

    def __init__(self, learning_rate=0.1):
        super(SgdStepper, self).__init__()
        self.learning_rate = learning_rate
        self.update = None

    def start(self, net):
        super(SgdStepper, self).start(net)
        self.update = net.handler.zeros(net.buffer.parameters.shape)

    def run(self):
        self.net.forward_pass(training_pass=True)
        self.net.backward_pass()
        self.net.handler.mult_st(-self.learning_rate,
                                 self.net.buffer.gradients,
                                 out=self.update)
        self.net.handler.add_tt(self.update,
                                self.net.buffer.parameters,
                                out=self.net.buffer.parameters)


class MomentumStepper(TrainingStepper):
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
        super(MomentumStepper, self).__init__()
        self.velocity = None
        self.learning_rate = learning_rate
        self.momentum = momentum
        assert isinstance(scale_learning_rate, bool), \
            "scale_learning_rate must be True or False."
        self.scale_learning_rate = scale_learning_rate

    def start(self, net):
        super(MomentumStepper, self).start(net)
        self.velocity = net.handler.zeros(net.buffer.parameters.shape)

    def run(self):
        learning_rate = self.learning_rate
        momentum = self.momentum
        if self.scale_learning_rate:
            learning_rate *= (1 - momentum)

        self.net.forward_pass(training_pass=True)
        self.net.backward_pass()

        self.net.handler.mult_st(momentum,
                                 self.velocity,
                                 out=self.velocity)
        self.net.handler.mult_add_st(-learning_rate,
                                     self.net.buffer.gradients,
                                     out=self.velocity)
        self.net.handler.add_tt(self.velocity,
                                self.net.buffer.parameters,
                                out=self.net.buffer.parameters)


class NesterovStepper(MomentumStepper):
    """
    Stochastic Gradient Descent with a Nesterov-style momentum term.
    learning_rate and momentum can be scheduled using
    brainstorm.training.schedules
    If scale_learning_rate is True (default),
    learning_rate is multiplied by (1 - momentum) when used.
    """
    def run(self):
        learning_rate = self.learning_rate
        momentum = self.momentum
        if self.scale_learning_rate:
            learning_rate *= (1 - momentum)

        self.net.handler.mult_st(momentum,
                                 self.velocity,
                                 out=self.velocity)
        self.net.handler.add_tt(self.velocity,
                                self.net.buffer.parameters,
                                out=self.net.buffer.parameters)
        self.net.forward_pass(training_pass=True)
        self.net.backward_pass()

        self.net.handler.mult_add_st(-learning_rate,
                                     self.net.buffer.gradients,
                                     self.velocity)
        self.net.handler.mult_add_st(-learning_rate,
                                     self.net.buffer.gradients,
                                     self.net.buffer.parameters)
