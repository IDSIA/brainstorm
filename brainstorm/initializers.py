#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np
from brainstorm.randomness import Seedable
from brainstorm.describable import Describable


# ########################### Support Classes #################################

class InitializationError(Exception):
    pass


class Initializer(Seedable, Describable):
    """
    Base Class for all initializers. It inherits from Seedable, so every
    sub-class has access to self.rnd, and it provides basic methods for
    converting from and to a description.
    """

    def __call__(self, shape):
        raise NotImplementedError()

    def _assert_2d(self, shape):
        if len(shape) != 2:
            raise InitializationError(
                "{} only works on 2D matrices, but was {}".format(
                    self.__class__.__name__, shape))


# ########################### Initializers ####################################

class Gaussian(Initializer):
    """
    Initializes the weights randomly according to a normal distribution of
    given mean and standard deviation.
    """
    __default_values__ = {'mean': 0.0}

    def __init__(self, std=0.1, mean=0.0):
        super(Gaussian, self).__init__()
        self.std = std
        self.mean = mean

    def __call__(self, shape):
        return self.rnd.randn(*shape) * self.std + self.mean


class Uniform(Initializer):
    """
    Initializes the weights randomly according to a uniform distribution over
    the interval [low, high].
    """
    __default_values__ = {'low': None}

    def __init__(self, high=0.1, low=None):
        super(Uniform, self).__init__()
        self.low = low
        self.high = high
        self.__init_from_description__(None)

    def __init_from_description__(self, description):
        if self.low is None:
            self.low = -self.high
        assert self.low < self.high, \
            "low has to be smaller than high but {} >= {}".format(self.low,
                                                                  self.high)

    def __call__(self, shape):
        v = ((self.high - self.low) * self.rnd.rand(*shape)) + self.low
        return v


class DenseSqrtFanIn(Initializer):
    """
    Initializes the weights randomly according to a uniform distribution over
    the interval [-1/sqrt(n), 1/sqrt(n)] where n is the number of inputs to
    each neuron.
    """

    __default_values__ = {'scale': 1.0}

    def __init__(self, scale=1.0):
        super(DenseSqrtFanIn, self).__init__()
        self.scale = scale

    def __call__(self, shape):
        self._assert_2d(shape)
        return self.scale * (2 * self.rnd.rand(*shape) - 1) / np.sqrt(shape[0])


class DenseSqrtFanInOut(Initializer):
    """
    Initializes the weights randomly according to a uniform distribution over
    the interval [-1/sqrt(n1+n2), 1/sqrt(n1+n2)] where n1 is the number of
    inputs to each neuron and n2 is the number of neurons in the current layer.
    Use scaling = 4*sqrt(6) for sigmoid units and sqrt(6) (used by default) for
    tanh units.
    """
    __default_values__ = {'scale': np.sqrt(6)}

    def __init__(self, scale=np.sqrt(6)):
        super(DenseSqrtFanInOut, self).__init__()
        self.scale = scale

    def __call__(self, shape):
        self._assert_2d(shape)
        n1, n2 = shape
        return self.scale * (2 * self.rnd.rand(*shape) - 1) / np.sqrt(n1 + n2)


class SparseInputs(Initializer):
    """
    Makes sure every neuron only gets activation from a certain number of input
    neurons and the rest of the weights are 0.
    The connections are initialized by evaluating the passed sub_initializer.

    Example usage:
    >> net = build_net(InputLayer(20) >> ForwardLayer(5))
    >> net.initialize(ForwardLayer=SparseInputs(Gaussian(), connections=10))
    """

    def __init__(self, sub_initializer, connections=15):
        super(SparseInputs, self).__init__()
        self.sub_initializer = sub_initializer
        self.connections = connections

    def __call__(self, shape):
        self._assert_2d(shape)
        if shape[0] < self.connections:
            raise InitializationError("Input dimension to small: {} < {}"
                                      "".format(shape[0], self.connections))

        sub_result = evaluate_initializer(self.sub_initializer, shape)
        connection_mask = np.zeros(shape)
        connection_mask[:self.connections, :] = 1.
        for i in range(shape[1]):
            self.rnd.shuffle(connection_mask[:, i])
        return sub_result * connection_mask


class SparseOutputs(Initializer):
    """
    Makes sure every neuron is propagating its activation only to a certain
    number of output neurons, and the rest of the weights are 0.
    The connections are initialized by evaluating the passed sub_initializer.

    Example usage:
    >> net = build_net(InputLayer(5) >> ForwardLayer(20))
    >> net.initialize(ForwardLayer=SparseOutputs(Gaussian(), connections=10))
    """

    def __init__(self, sub_initializer, connections=15):
        super(SparseOutputs, self).__init__()
        self.sub_initializer = sub_initializer
        self.connections = connections

    def __call__(self, shape):
        self._assert_2d(shape)
        if shape[1] < self.connections:
            raise InitializationError("Output dimension to small: {} < {}"
                                      "".format(shape[1], self.connections))
        sub_result = evaluate_initializer(self.sub_initializer, shape)
        connection_mask = np.zeros(shape)
        connection_mask[:, :self.connections] = 1.
        for i in range(shape[0]):
            self.rnd.shuffle(connection_mask[i, :])
        return sub_result * connection_mask


class EchoState(Initializer):
    """
    Classic echo state initialization. Creates a matrix with a fixed spectral
    radius (default=1.0). Spectral radius should be < 1 to satisfy
    ES-property. Only works for square matrices.

    Example usage:
    >> net = build_net(InputLayer(5) >> RnnLayer(20, act_func='tanh'))
    >> net.initialize(default=Gaussian(), RnnLayer={'HR': EchoState(0.77)})
    """

    __default_values__ = {'spectral_radius': 1.0}

    def __init__(self, spectral_radius=1.0):
        super(EchoState, self).__init__()
        self.spectral_radius = spectral_radius

    def __call__(self, shape):
        self._assert_2d(shape)
        if shape[0] != shape[1]:
            raise InitializationError("Matrix should be square but was: {}"
                                      "".format(shape))

        weights = self.rnd.uniform(-0.5, 0.5, size=shape)
        # normalizing and setting spectral radius (correct, slow):
        rho_weights = max(abs(np.linalg.eig(weights)[0]))
        return weights * (self.spectral_radius / rho_weights)


# ########################### helper methods ##################################

def evaluate_initializer(initializer, shape, fallback=None):
    if isinstance(initializer, Initializer):
        try:
            return initializer(shape)
        except InitializationError:
            if fallback is not None:
                return evaluate_initializer(fallback, shape)
            raise
    else:
        return np.array(initializer)
