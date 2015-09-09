#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from brainstorm.describable import Describable
import abc
import six


@six.add_metaclass(abc.ABCMeta)
class Handler(Describable):

    """
    Abstract Base Class for all handlers.

    Used mainly to ensure a common interface. When implementing new methods
    you should stick to the naming scheme. Each mathematical operation should
    have a suffix indicating what shapes of inputs it expects. This ranges from
    s for scalar, over v for vector, m for matrix to t for tensor (which means
    arbitrary shape).
    """

    # ------------------------- Allocate new memory ------------------------- #
    @abc.abstractmethod
    def allocate(self, shape):
        """ Allocate new memory with given shape but arbitrary content.

        :param shape: shape of the array
        :type shape: tuple[int]

        :returns: New array with given shape.
        """

    @abc.abstractmethod
    def zeros(self, shape):
        """ Allocate new memory with given shape and filled with zeros.

        :param shape: shape of the array
        :type shape: tuple[int]

        :returns: New array with given shape.
        """

    @abc.abstractmethod
    def ones(self, shape):
        """ Allocate new memory with given shape and filled with ones.

        :param shape: shape of the array
        :type shape: tuple[int]

        :returns: New array with given shape.
        """

    # ---------------------------- Copy and Fill ---------------------------- #
    @abc.abstractmethod
    def set_from_numpy(self, mem, arr):
        """ Set the content of an array from a given numpy array.

        :param mem: destination array that should be set
        :param arr: source numpy array
        :type arr: numpy.ndarray
        """

    @abc.abstractmethod
    def get_numpy_copy(self, mem):
        """ Return a copy of the given data as a numpy array.

        :param mem: source array that should be copied
        :returns: numpy array with same content as mem
        :rtype: numpy.ndarray
        """

    @abc.abstractmethod
    def copy_to(self, dest, src):
        """ """

    @abc.abstractmethod
    def fill(self, mem, val):
        """ """

    @abc.abstractmethod
    def create_from_numpy(self, arr):
        """ """

    # ---------------- Mathematical Operations ---------------- #

    def fill_gaussian(self, mean, std, out):
        """
        :param mean:
        :param std:
        :param out:
        :return:
        """

    def generate_probability_mask(self, mask, probability):
        """ Fill an array with zeros and ones.

        Fill an array with zeros and ones such that the probability of an
        entry being one is equal to *probability*.

        :param mask: array to will be filled
        :param probability: probability of an entry of *mask* being one
        :type mask: numpy.ndarray
        :type probability: float

        :rtype: None
        """

    @abc.abstractmethod
    def add_tt(self, a, b, out):
        """ """

    @abc.abstractmethod
    def add_st(self, s, t, out):
        """ """

    @abc.abstractmethod
    def add_mv(self, m, v, out):
        """ """

    @abc.abstractmethod
    def subtract_tt(self, a, b, out):
        """ """

    @abc.abstractmethod
    def subtract_mv(self, m, v, out):
        """ """

    @abc.abstractmethod
    def sum_t(self, a, axis, out):
        """ """

    @abc.abstractmethod
    def mult_tt(self, a, b, out):
        """ """

    @abc.abstractmethod
    def mult_st(self, a, b, out):
        """ """

    @abc.abstractmethod
    def mult_add_st(self, a, b, out):
        out[:] += a * b

    @abc.abstractmethod
    def mult_mv(self, m, v, out):
        """
        Multiply (M, N) matrix elementwise by a (1, N) vector using
        broadcasting.
        """

    @abc.abstractmethod
    def mult_add_tt(self, a, b, out):
        """ """

    @abc.abstractmethod
    def divide_tt(self, a, b, out):
        """ """

    @abc.abstractmethod
    def divide_mv(self, m, v, out):
        """
        Divide (M, N) matrix elementwise by a (1, N) vector using broadcasting.
        """

    @abc.abstractmethod
    def dot_mm(self, a, b, out, transa=False, transb=False):
        """ """

    @abc.abstractmethod
    def dot_add_mm(self, a, b, out, transa=False, transb=False):
        """ """

    @abc.abstractmethod
    def broadcast_features_t(self, a, out):
        """ """

    @abc.abstractmethod
    def clip_t(self, a, a_min, a_max, out):
        """ """

    @abc.abstractmethod
    def log_t(self, a, out):
        """ """

    @abc.abstractmethod
    def sqrt_t(self, a, out):
        """ """

    @abc.abstractmethod
    def sign_t(self, a, out):
        """ """

    @abc.abstractmethod
    def binarize_v(self, v, out):
        """ """

    @abc.abstractmethod
    def index_m_by_v(self, m, v, out):
        """ """

    @abc.abstractmethod
    def conv2d_forward_batch(self, inputs, weights, bias, outputs,
                             padding, stride):
        """ """

    @abc.abstractmethod
    def conv2d_backward_batch(self, inputs, weights, padding, stride,
                              in_deltas, out_deltas, weight_deltas,
                              bias_deltas):
        """ """

    @abc.abstractmethod
    def maxpool2d_forward_batch(self, inputs, window, outputs, padding,
                                stride):
        """ """

    @abc.abstractmethod
    def maxpool2d_backward_batch(self, inputs, window, outputs, padding,
                                 stride, in_deltas, out_deltas):
        """ """

    @abc.abstractmethod
    def avgpool2d_forward_batch(self, inputs, window, outputs, padding,
                                stride):
        """ """

    @abc.abstractmethod
    def avgpool2d_backward_batch(self, inputs, window, outputs, padding,
                                 stride, in_deltas, out_deltas):
        """ """

    # ---------------- Activation functions -----------------------------------

    @abc.abstractmethod
    def sigmoid(self, x, y):
        """ """

    @abc.abstractmethod
    def sigmoid_deriv(self, x, y, dy, dx):
        """ """

    @abc.abstractmethod
    def tanh(self, x, y):
        """ """

    @abc.abstractmethod
    def tanh_deriv(self, x, y, dy, dx):
        """ """

    @abc.abstractmethod
    def rel(self, x, y):
        """ """

    @abc.abstractmethod
    def rel_deriv(self, x, y, dy, dx):
        """ """

    @abc.abstractmethod
    def softmax_m(self, m, out):
        """Applies softmax to matrix over last dimension"""
