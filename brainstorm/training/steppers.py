#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

import math

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

class RMSpropStepper(TrainingStepper):
    def __init__(self, learning_rate=0.001, alpha=0.9, eps=1e-6):
        super(RMSpropStepper, self).__init__()
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.eps = eps
        self.update = None
        self.scratch = None

    def start(self, net):
        super(RMSpropStepper, self).start(net)
        self.update = net.handler.zeros(net.buffer.parameters.shape)
        self.scratch = net.handler.zeros(net.buffer.parameters.shape)

    def run(self):
        self.net.forward_pass(training_pass=True)
        self.net.backward_pass()

        #self.update *= self.alpha
        self.net.handler.mult_st(self.alpha,
                                 self.update,
                                 out=self.update)

        #self.update += (1 - self.alpha) * grad * grad
        self.net.handler.mult_tt(self.net.buffer.gradients,
                                 self.net.buffer.gradients,
                                 out=self.scratch)
        self.net.handler.mult_add_st(1.0 - self.alpha,
                                     self.scratch,
                                     out=self.update)

        # grad / sqrt(update + self.eps)
        self.net.handler.add_st(self.eps,
                                self.update,
                                out=self.scratch)
        self.net.handler.sqrt_t(self.scratch,
                                out=self.scratch)
        self.net.handler.divide_tt(self.net.buffer.gradients,
                                   self.scratch,
                                   out=self.scratch)

        # param -= learning_rate * grad / sqrt(update + self.eps)
        self.net.handler.mult_add_st(-self.learning_rate,
                                     self.scratch,
                                     self.net.buffer.parameters)

class AdaDeltaStepper(TrainingStepper):
    def __init__(self, learning_rate=1.0, alpha=0.95, eps=1e-6):
        super(AdaDeltaStepper, self).__init__()
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.eps = eps
        self.accumulator = None
        self.delta_accumulator = None
        self.scratch = None
        self.scratch_2 = None

    def start(self, net):
        super(AdaDeltaStepper, self).start(net)
        self.accumulator = net.handler.zeros(net.buffer.parameters.shape)
        self.delta_accumulator = net.handler.zeros(net.buffer.parameters.shape)
        self.scratch = net.handler.zeros(net.buffer.parameters.shape)
        self.scratch_2 = net.handler.zeros(net.buffer.parameters.shape)

    def run(self):
        self.net.forward_pass(training_pass=True)
        self.net.backward_pass()

        # accumulator *= self.alpha
        self.net.handler.mult_st(self.alpha,
                                 self.accumulator,
                                 out=self.accumulator)

        # accumulator += (1 - self.alpha) * grad * grad
        self.net.handler.mult_tt(self.net.buffer.gradients,
                                 self.net.buffer.gradients,
                                 out=self.scratch)
        self.net.handler.mult_add_st(1.0 - self.alpha,
                                     self.scratch,
                                     out=self.accumulator)

        # dx = sqrt((delta_accumulator + eps) / (accumulator + eps)) * grad
        self.net.handler.add_st(self.eps,
                                 self.delta_accumulator,
                                 out=self.scratch)
        self.net.handler.add_st(self.eps,
                                 self.accumulator,
                                 out=self.scratch_2)
        self.net.handler.divide_tt(self.scratch,
                                   self.scratch_2,
                                   out=self.scratch)
        self.net.handler.sqrt_t(self.scratch,
                                out=self.scratch)
        self.net.handler.mult_tt(self.net.buffer.gradients,
                                 self.scratch,
                                 out=self.scratch_2)

        # delta_accumulator *= self.alpha
        self.net.handler.mult_st(self.alpha,
                                 self.delta_accumulator,
                                 out=self.delta_accumulator)

        # delta_accumulator += (1 - self.alpha) * dx * dx
        self.net.handler.mult_tt(self.scratch_2,
                                 self.scratch_2,
                                 out=self.scratch)
        self.net.handler.mult_add_st(1.0 - self.alpha,
                                     self.scratch,
                                     out=self.delta_accumulator)

        # param -= dx
        self.net.handler.mult_add_st(-1,
                                     self.scratch_2,
                                     self.net.buffer.parameters)

class AdamStepper(TrainingStepper):
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                 eps=1e-8):
        super(AdamStepper, self).__init__()
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.t = None
        self.v = None
        self.m = None
        self.scratch = None

    def start(self, net):
        super(AdamStepper, self).start(net)
        self.t = 0
        self.v = net.handler.zeros(net.buffer.parameters.shape)
        self.m = net.handler.zeros(net.buffer.parameters.shape)
        self.scratch = net.handler.zeros(net.buffer.parameters.shape)

    def run(self):
        self.net.forward_pass(training_pass=True)
        self.net.backward_pass()
        self.t += 1

        # alpha_t = learning_rate*sqrt(1-beta_2**t)/(1-beta_1**t)
        alpha_t = self.learning_rate*math.sqrt(1-self.beta_2**self.t)/(1-self.beta_1**self.t)

        # m += (1 - self.beta_1) * (grad - self.m)
        self.net.handler.subtract_tt(self.net.buffer.gradients,
                                     self.m, out=self.scratch)
        self.net.handler.mult_add_st(1.0 - self.beta_1,
                                     self.scratch,
                                     out=self.m)

        # v += (1 - self.beta_2) * (grad * grad - v)
        self.net.handler.mult_tt(self.net.buffer.gradients,
                                 self.net.buffer.gradients,
                                 out=self.scratch)
        self.net.handler.subtract_tt(self.scratch,
                                     self.v, out=self.scratch)
        self.net.handler.mult_add_st(1.0 - self.beta_2,
                                     self.scratch,
                                     out=self.v)

        # param -= alpha_t * m / (sqrt(v) + self.eps)
        self.net.handler.sqrt_t(self.v,
                                out=self.scratch)
        self.net.handler.add_st(self.eps,
                                 self.scratch,
                                 out=self.scratch)
        self.net.handler.divide_tt(self.m,
                                   self.scratch,
                                   out=self.scratch)
        self.net.handler.mult_add_st(-alpha_t,
                                     self.scratch,
                                     self.net.buffer.parameters)

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
