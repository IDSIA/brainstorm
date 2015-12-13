#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

import numpy as np
import six

from brainstorm.describable import Describable
from brainstorm.randomness import Seedable
from brainstorm.utils import InitializationError

# somehow this construction is needed because in __all__ unicode does not work
__all__ = [str(a) for a in [
    'ArrayInitializer', 'DenseSqrtFanIn', 'DenseSqrtFanInOut', 'EchoState',
    'Gaussian', 'Identity', 'LstmOptInit', 'Orthogonal', 'RandomWalk',
    'SparseInputs', 'SparseOutputs', 'Uniform']]


# ########################### Support Classes #################################

class Initializer(Seedable, Describable):
    """
    Base class for all initializers. It inherits from Seedable, so every
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
    """
    Initializes the parameters as the values of the input array.
    """
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


class DenseSqrtFanIn(Initializer):
    """
    Initializes the parameters randomly according to a uniform distribution
    over the interval [-scale/sqrt(n), scale/sqrt(n)] where n is the number of
    inputs to each unit. Uses scale=sqrt(6) by default which is appropriate
    for rel units.

    When number of inputs and outputs are the same, this is equivalent to
    using ``DenseSqrtFanInOut``.

    Scaling:
        * rel: sqrt(6)
        * tanh: sqrt(3)
        * sigmoid: 4 * sqrt(3)
        * linear: 1

    Args:
        scale (Optional(float or str)):
            The activation function dependent scaling factor. Can be either
            float or one of ['rel', 'tanh', 'sigmoid', 'linear'].
            Defaults to 'rel'.
    """

    __default_values__ = {'scale': 'rel'}

    def __init__(self, scale='rel'):
        super(DenseSqrtFanIn, self).__init__()
        self.scale = scale

    def __call__(self, shape):
        self._assert_atleast2d(shape)
        num_in = np.prod(shape[1:])
        if isinstance(self.scale, six.string_types):
            scale = {
                'rel': np.sqrt(6),
                'tanh': np.sqrt(3),
                'sigmoid': 4 * np.sqrt(3),
                'linear': 1
            }[self.scale]
        else:
            scale = self.scale
        return scale * (2 * self.rnd.rand(*shape) - 1) / np.sqrt(num_in)


class DenseSqrtFanInOut(Initializer):
    """
    Initializes the parameters randomly according to a uniform distribution
    over the interval [-scale/sqrt(n1+n2), scale/sqrt(n1+n2)] where n1 is the
    number of inputs to each unit and n2 is the number of units in the
    current layer. Uses scale=sqrt(12) by default which is appropriate for rel
    units.

    Scaling:
        * rel: sqrt(12)
        * tanh: sqrt(6)
        * sigmoid: 4 * sqrt(6)
        * linear: 1

    Args:
        scale (Optional(float or str)):
            The activation function dependent scaling factor. Can be either
            float or one of ['rel', 'tanh', 'sigmoid', 'linear'].
            Defaults to 'rel'.

    Reference:
        Glorot, Xavier, and Yoshua Bengio.
        "Understanding the difficulty of training deep feedforward neural
        networks" International conference on artificial intelligence and
        statistics. 2010.
    """
    __default_values__ = {'scale': 'rel'}

    def __init__(self, scale='rel'):
        super(DenseSqrtFanInOut, self).__init__()
        self.scale = scale

    def __call__(self, shape):
        self._assert_atleast2d(shape)
        n1, n2 = shape[0], np.prod(shape[1:])
        if isinstance(self.scale, six.string_types):
            scale = {
                'rel': np.sqrt(12),
                'tanh': np.sqrt(6),
                'sigmoid': 4 * np.sqrt(6),
                'linear': 1
            }[self.scale]
        else:
            scale = self.scale
        return scale * (2 * self.rnd.rand(*shape) - 1) / np.sqrt(n1 + n2)


class EchoState(Initializer):
    """
    Classic echo state initialization. Creates a matrix with a fixed spectral
    radius (default=1.0). Spectral radius should be < 1 to satisfy
    ES-property. Only works for square matrices.

    Example:
        >>> net.initialize(default=Gaussian(),
                           Recurrent={'R': EchoState(0.77)})
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

        parameters = self.rnd.uniform(-0.5, 0.5, size=shape)
        # normalizing and setting spectral radius (correct, slow):
        rho_parameters = max(abs(np.linalg.eig(parameters)[0]))
        return parameters * (self.spectral_radius / rho_parameters)


class Gaussian(Initializer):
    """
    Initializes the parameters randomly according to a normal distribution of
    given mean and standard deviation.
    """
    __default_values__ = {'mean': 0.0}

    def __init__(self, std=0.1, mean=0.0):
        super(Gaussian, self).__init__()
        self.std = std
        self.mean = mean

    def __call__(self, shape):
        return self.rnd.randn(*shape) * self.std + self.mean


class Identity(Initializer):
    """
    Initialize a matrix to the (scaled) identity matrix + some noise.
    """

    def __init__(self, scale=1.0, std=0.01, enforce_square=True):
        super(Identity, self).__init__()
        self.scale = scale
        self.std = std
        self.enforce_square = enforce_square

    def __call__(self, shape):
        if len(shape) != 2:
            raise InitializationError("Works only with 2D matrices but shape "
                                      "was: {}".format(shape))
        if self.enforce_square and shape[0] != shape[1]:
            raise InitializationError("Matrix needs to be square, but was {}"
                                      "".format(shape))
        weights = np.eye(shape[0], shape[1], dtype=np.float) * self.scale
        weights += self.rnd.randn(*shape) * self.std
        return weights


class LstmOptInit(Initializer):
    """
    Used to initialize an LstmOpt layer.
    This is useful because in an LstmOpt layer all the parameters are
    concatenated for efficiency.

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
        weights = np.zeros(shape)
        n = shape[0] // 4
        sub_shape = (n,) + shape[1:]
        weights[:n] = evaluate_initializer(
            self.block_input, sub_shape, seed=self.rnd.generate_seed())
        weights[n:2 * n] = evaluate_initializer(
            self.input_gate, sub_shape, seed=self.rnd.generate_seed())
        weights[2 * n:3 * n] = evaluate_initializer(
            self.forget_gate, sub_shape, seed=self.rnd.generate_seed())
        weights[3 * n:] = evaluate_initializer(
            self.output_gate, sub_shape, seed=self.rnd.generate_seed())
        return weights


class Orthogonal(Initializer):
    """
    Orthogonal initialization.

    Reference:
    Saxe, Andrew M., James L. McClelland, and Surya Ganguli.
    "Exact solutions to the nonlinear dynamics of learning in deep linear
    neural networks." arXiv preprint arXiv:1312.6120 (2013).
    """
    def __init__(self, scale=1.0):
        super(Orthogonal, self).__init__()
        self.scale = scale

    def __call__(self, shape):
        if len(shape) != 2:
            raise InitializationError("Works only with 2D matrices but shape "
                                      "was: {}".format(shape))
        a = self.rnd.randn(*shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == shape else v
        return (self.scale * q).reshape(shape)


class RandomWalk(Initializer):
    """
    Initializes a (square) weight matrix with the random walk scheme proposed
    by:

    Sussillo, David, and L. F. Abbott.
    "Random Walk Initialization for Training Very Deep Feedforward Networks."
    arXiv:1412.6558 [cs, Stat], December 19, 2014.
    http://arxiv.org/abs/1412.6558.

    """
    __default_values__ = {'scale': None}

    def __init__(self, act_func='linear', scale=None):
        super(RandomWalk, self).__init__()
        self.act_func = act_func
        self.scale = scale

    def __call__(self, shape):
        if len(shape) != 2:
            raise InitializationError("Works only with 2D matrices but shape "
                                      "was: {}".format(shape))
        if shape[0] != shape[1]:
            raise InitializationError("Matrix needs to be square, but was {}"
                                      "".format(shape))

        N = shape[1]
        if self.scale is None:
            scale = {
                'linear': np.exp(1 / (2 * N)),
                'rel': np.sqrt(2) * np.exp(1.2 / (max(N, 6) - 2.4))
            }[self.act_func]
        else:
            scale = self.scale

        return scale * self.rnd.randn(*shape) / N


class SparseInputs(Initializer):
    """
    Makes sure every unit only gets activation from a certain number of input
    units and the rest of the parameters are 0.
    The connections are initialized by evaluating the passed sub_initializer.

    Example:
        >>> net.initialize(FullyConnected=SparseInputs(Gaussian(),
        ...                                            connections=10))
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
    Makes sure every unit is propagating its activation only to a certain
    number of output units, and the rest of the parameters are 0.
    The connections are initialized by evaluating the passed sub_initializer.

    Example:
        >>> net.initialize(FullyConnected=SparseOutputs(Gaussian(),
                                                        connections=10))
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


class Uniform(Initializer):
    """
    Initializes the parameters randomly according to a uniform distribution
    over the interval [low, high].
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
