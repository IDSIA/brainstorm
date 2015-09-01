#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np
from brainstorm.randomness import Seedable
from brainstorm.describable import Describable
from brainstorm.utils import InitializationError

# somehow this construction is needed because in __all__ unicode does not work
__all__ = [str(a) for a in [
    'Gaussian', 'Uniform', 'DenseSqrtFanIn', 'DenseSqrtFanInOut',
    'SparseInputs', 'SparseOutputs', 'EchoState', 'LstmOptInit']]


# ########################### Support Classes #################################

class Initializer(Seedable, Describable):
    """
    Base Class for all initializers. It inherits from Seedable, so every
    sub-class has access to self.rnd, and it provides basic methods for
    converting from and to a description.
    """

    def __call__(self, shape):
        raise NotImplementedError()

    def _assert_atleast2d(self, shape):
        if len(shape) < 2:
            raise InitializationError(
                "{} only works on >2D matrices, but shape was {}".format(
                    self.__class__.__name__, shape))


# ########################### Initializers ####################################

class ArrayInitializer(Initializer):
    def __init__(self, array):
        super(ArrayInitializer, self).__init__()
        self.array = np.array(array)

    def __call__(self, shape):
        if not self.array.shape == shape:
            raise InitializationError('Shape mismatch {} != {}'
                                      .format(self.array.shape, shape))

        return self.array

    def __describe__(self):
        return self.array.tolist()


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

    def __init__(self, low=0.1, high=None):
        super(Uniform, self).__init__()
        self.low = low
        self.high = high
        self.__init_from_description__(None)

    def __init_from_description__(self, description):
        if self.high is None:
            self.low, self.high = sorted([-self.low, self.low])
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
    each neuron. Uses scaling = sqrt(6) by default which is appropriate for
    rel units.
    """

    __default_values__ = {'scale': np.sqrt(6)}

    def __init__(self, scale=np.sqrt(6)):
        super(DenseSqrtFanIn, self).__init__()
        self.scale = scale

    def __call__(self, shape):
        self._assert_atleast2d(shape)
        num_in = np.prod(shape[1:])
        return self.scale * (2 * self.rnd.rand(*shape) - 1) / np.sqrt(num_in)


class DenseSqrtFanInOut(Initializer):
    """
    Initializes the weights randomly according to a uniform distribution over
    the interval [-1/sqrt(n1+n2), 1/sqrt(n1+n2)] where n1 is the number of
    inputs to each neuron and n2 is the number of neurons in the current layer.
    Use scaling = 4*sqrt(6) for sigmoid units, sqrt(6) for tanh units and
    sqrt(12) for rel units (used by default).
    """
    __default_values__ = {'scale': np.sqrt(12)}

    def __init__(self, scale=np.sqrt(12)):
        super(DenseSqrtFanInOut, self).__init__()
        self.scale = scale

    def __call__(self, shape):
        self._assert_atleast2d(shape)
        n1, n2 = shape[0], np.prod(shape[1:])
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
        self._assert_atleast2d(shape)
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
        self._assert_atleast2d(shape)
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
        self._assert_atleast2d(shape)
        if shape[0] != shape[1]:
            raise InitializationError("Matrix should be square but was: {}"
                                      "".format(shape))

        weights = self.rnd.uniform(-0.5, 0.5, size=shape)
        # normalizing and setting spectral radius (correct, slow):
        rho_weights = max(abs(np.linalg.eig(weights)[0]))
        return weights * (self.spectral_radius / rho_weights)


class LstmOptInit(Initializer):
    """
    Used to initialize an LstmOpt layer.
    This is useful because in an LstmOpt layer all the weights are concatenated

    The parameters (input_block, input_gate, forget_gate, and output_gate)
    can be scalars or Initializers themselves.
    """
    def __init__(self, input_block=0.0, input_gate=0.0, forget_gate=0.0,
                 output_gate=0.0):
        super(LstmOptInit, self).__init__()
        self.block_input = input_block
        self.input_gate = input_gate
        self.forget_gate = forget_gate
        self.output_gate = output_gate

    def __call__(self, shape):
        if shape[0] % 4 != 0:
            raise InitializationError("First dim of LstmOpt shape needs to be "
                                      "divisible by 4. But shape was {}"
                                      .format(shape))
        W = np.zeros(shape)
        n = shape[0] // 4
        sub_shape = (n,) + shape[1:]
        W[:n] = evaluate_initializer(self.block_input, sub_shape,
                                     seed=self.rnd.generate_seed())
        W[n:2 * n] = evaluate_initializer(self.input_gate, sub_shape,
                                          seed=self.rnd.generate_seed())
        W[2 * n:3 * n] = evaluate_initializer(self.forget_gate, sub_shape,
                                              seed=self.rnd.generate_seed())
        W[3 * n:] = evaluate_initializer(self.output_gate, sub_shape,
                                         seed=self.rnd.generate_seed())
        return W


# ########################### helper methods ##################################

def evaluate_initializer(initializer, shape, fallback=None, seed=None):
    if isinstance(initializer, Initializer):
        if seed is not None:
            initializer.rnd.set_seed(seed)
        try:
            result = initializer(shape)
        except InitializationError:
            if fallback is not None:
                return evaluate_initializer(fallback, shape, seed=seed)
            raise
    else:
        if not isinstance(initializer, (int, float)):
            raise TypeError('type {} not supported as initializer'
                            .format(type(initializer)))
        result = np.empty(shape, dtype=np.float64)
        result[:] = initializer

    return result
