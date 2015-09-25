#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from brainstorm.describable import Describable
import abc
import six


@six.add_metaclass(abc.ABCMeta)
class Handler(Describable):
    """Abstract Base Class for all handlers.

    This base is used mainly to ensure a common interface and provide
    documentation for derived hanlders. When implementing new methods
    one should adhere to the naming scheme. Most mathematical operations should
    have a suffix or suffixes indicating the shapes of inputs it expects:

    `s` for scalar,
    `v` for vector (a 2D array with at least dimension equal to 1),
    `m` for matrix  (a 2D array),
    `t` for tensor (which means arbitrary shape, synonym for `array`).

    Note that these shapes are not checked by each handler itself. However,
    the DebugHandler can be used to perform these checks to ensure that
    operations are not abused.

    Attributes:
      dtype: Data type that this handler works with.
      context: Context which may be used by this handler for operation.
      EMPTY: An empty array matching this handler's type.
      rnd: A random state maintained by this handler.
      array_type (None): The type of array object that this handler works with.
    """

    # ------------------------- Allocate new memory ------------------------- #
    @abc.abstractmethod
    def allocate(self, shape):
        """Allocate new memory with given shape but arbitrary content.

        Args:
            shape (tuple[int]): Shape of the array.

        Returns:
            object: New array with given shape.
        """

    @abc.abstractmethod
    def zeros(self, shape):
        """Allocate new memory with given shape and filled with zeros.

        Args:
            shape (tuple[int]): Shape of the array.

        Returns:
            object: New array with given shape filled with zeros.
        """

    @abc.abstractmethod
    def ones(self, shape):
        """Allocate new memory with given shape and filled with ones.

        Args:
            shape (tuple[int]): Shape of the array.

        Returns:
            object: New array with given shape filled with ones.
        """

    # ---------------------------- Copy and Fill ---------------------------- #
    @abc.abstractmethod
    def set_from_numpy(self, mem, arr):
        """Set the content of an array from a given numpy array.

        Args:
            mem (array_type): Destination array that should be set.
            arr (numpy.ndarray): Source numpy array.
        Returns:
            None
        """

    @abc.abstractmethod
    def get_numpy_copy(self, mem):
        """Return a copy of the given data as a numpy array.

        Args:
            mem (array_type): Source array to be copied.

        Returns:
            numpy.ndarray: Numpy array with same content as mem.
        """

    @abc.abstractmethod
    def copy_to(self, dest, src):
        """Copy the contents of one array to another.

        Both source and destination arrays must be of this handler's supported
        type and have the same shape.

        Args:
            dest (array_type): Destination array.
            src (array_type): Source array.
        Returns:
            None
        """

    @abc.abstractmethod
    def fill(self, mem, val):
        """Fill an array with a given value.

        Args:
            mem (array_type): Array to be filled.
            val (dtype): Value to fill.
        Returns:
            None
        """

    @abc.abstractmethod
    def create_from_numpy(self, arr):
        """Create a new array with the same entries as a Numpy array.

        Args:
            arr (numpy.ndarray): Numpy array whose elements should be used
                                 to fill the new array.
        Returns:
            array_type: New array with same shape and entries as the given
                        Numpy array.
        """

    # ---------------------------- Debug helpers ---------------------------- #
    @abc.abstractmethod
    def is_fully_finite(self, a):
        """Check if all entries of the array are finite (no nans or infs).

        Args:
            a (array_type): Input array to check.
        Returns:
            bool: True if there are no infs or nans, False otherwise.
        """

    # ---------------- Mathematical Operations ---------------- #

    def fill_gaussian(self, mean, std, out):
        """Fill an array with values drawn from a Gaussian distribution.

        Args:
            mean (float): Mean of the Gaussian Distribution.
            std (float): Standard deviation of the Gaussian distribution.
            out (array_type): Target array to fill with values.

        Returns:
            None
        """

    def generate_probability_mask(self, mask, probability):
        """Fill an array with zeros and ones.

        Fill an array with zeros and ones such that the probability of an
        entry being one is equal to `probability`.

        Args:
            mask (array_type): Array to will be filled.
            probability (float): Probability of an entry of `mask` being one.
        Returns:
            None
        """

    @abc.abstractmethod
    def add_tt(self, a, b, out):
        """Add two tensors element-wise,

        Args:
            a (array_type): First array.
            b (array_type): Second array.
            out (array_type): Array into which the output is placed.
        Returns:
            None
        """

    @abc.abstractmethod
    def add_st(self, s, t, out):
        """Add a scalar to each element of a tensor.

        Args:
            s (dtype): The scalar value to be added.
            t (array_type): The array to be added.
            out (array_type): Array into which the output is placed.
        Returns:
            None
        """

    @abc.abstractmethod
    def add_mv(self, m, v, out):
        """Add a matrix to a vector with broadcasting.

        Add an (M, N) matrix to a (1, N) or (M, 1) vector using
        broadcasting such that the output is (M, N).

        Args:
            m (array_type): The first array to be added. Must be 2D.
            v (array_type): The second array to be added. Must be 2D with at
                            least one dimension of size 1 and second dimension
                            matching the corresponding size of `m`.
            out (array_type): Array into which the output is placed.
        Returns:
            None
        """

    @abc.abstractmethod
    def subtract_tt(self, a, b, out):
        """Subtract a tensor from another element-wise.

        Args:
            a (array_type): First array.
            b (array_type): Second array, to be subtracted from `a`.
            out (array_type): Array into which the output (a - b) is placed.
        Returns:
            None
        """

    @abc.abstractmethod
    def subtract_mv(self, m, v, out):
        """Subtract a vector from a matrix with broadcasting.

        Args:
            m (array_type): The first array. Must be 2D.
            v (array_type): The second array, to be subtracted from `a`. Must
                            be 2D with at least one dimension of size 1 and
                            second dimension matching the corresponding size of
                            `m`.
            out (array_type): Array into which the output is placed.
        Returns:
            None
        """

    @abc.abstractmethod
    def sum_t(self, a, axis, out):
        """Sum the elements of an array along a given axis.

        If axis is None, the sum is computed over all elements of the array.
        Otherwise, it is computed along the specified axis and the output is
        an array with ndim = a.ndim - 1.
        NOTE: Only 1D and 2D arrays are currently supported.

        Args:
            a (array_type): Array to be summed.
            axis (int): Axis over which the summation should be done.
            out (array_type): Array into which the output is placed.
        Returns:
            None
        """

    @abc.abstractmethod
    def mult_tt(self, a, b, out):
        """Multiply two tensors of the same shape element-wise.

        Args:
            a (array_type): First array.
            b (array_type): Second array.
            out (array_type): Array into which the output is placed.
        Returns:
            None
        """

    @abc.abstractmethod
    def mult_st(self, a, b, out):
        """Multiply a scalar with each element of a tensor.

        Args:
            s (dtype): The scalar value to be multiplied.
            t (array_type): The array to be multiplied.
            out (array_type): Array into which the output is placed.
        Returns:
            None
        """

    @abc.abstractmethod
    def mult_add_st(self, a, b, out):
        """Multiply a scalar with each element of a tensor and add to a tensor.

        Args:
            s (dtype): The scalar value to be multiplied.
            t (array_type): The array to be multiplied.
            out (array_type): Array into which the product is added.
        Returns:
            None
        """
        out[:] += a * b

    @abc.abstractmethod
    def mult_mv(self, m, v, out):
        """Multiply a matrix with a vector.

        Multiply an (M, N) matrix with a (1, N) or (M, 1) vector using
        broadcasting such that the output is (M, N).

        Args:
            m (array_type): The first array. Must be 2D.
            v (array_type): The second array, to be multiplied with `a`. Must
                            be 2D with at least one dimension of size 1 and
                            second dimension matching the corresponding size of
                            `m`.
            out (array_type): Array into which the output is placed.
        Returns:
            None
        """

    @abc.abstractmethod
    def mult_add_tt(self, a, b, out):
        """Multiply two tensors element-wise and add to a tensor,

        Args:
            a (array_type): First array.
            b (array_type): Second array.
            out (array_type): Array into which the output is added.
        Returns:
            None
        """

    @abc.abstractmethod
    def divide_tt(self, a, b, out):
        """Divide two tensors element-wise,

        Args:
            a (array_type): First array (dividend).
            b (array_type): Second array (divisor).
            out (array_type): Array into which the output is placed.
        Returns:
            None
        """

    @abc.abstractmethod
    def divide_mv(self, m, v, out):
        """Divide a matrix by a vector.

        Divide a (M, N) matrix element-wise by a (1, N) vector using
        broadcasting such that the output is (M, N).

        Args:
            a (array_type): First array (dividend). Must be 2D.
            b (array_type): Second array (divisor). Must be 2D with at
                            least one dimension of size 1 and second
                            dimension matching the corresponding size of `m`.
            out (array_type): Array into which the output is placed.
        Returns:
            None
        """

    @abc.abstractmethod
    def dot_mm(self, a, b, out, transa=False, transb=False):
        """Multiply two matrices.

        Only 2D arrays (matrices) are supported.

        Args:
            a (array_type): First matrix.
            b (array_type): Second matrix.
            out (array_type): Array into which the output is placed. Must
                              have correct shape for the product of the two
                              matrices.
        Returns:
            None
        """

    @abc.abstractmethod
    def dot_add_mm(self, a, b, out, transa=False, transb=False):
        """Multiply two matrices and add to a matrix.

        Only 2D arrays (matrices) are supported.

        Args:
            a (array_type): First matrix.
            b (array_type): Second matrix.
            out (array_type): Array into which the output is added. Must
                              have correct shape for the product of the two
                              matrices.
        Returns:
            None
        """

    @abc.abstractmethod
    def broadcast_features_t(self, a, out):
        """

        Args:
            out (array_type): Array into which the output is placed.
        Returns:
            None
        """

    @abc.abstractmethod
    def clip_t(self, a, a_min, a_max, out):
        """Clip (limit) the values in an array.

        Given an interval, values outside the interval are clipped to the
        interval edges. For example, if an interval of [0, 1] is specified,
        values smaller than 0 become 0, and values larger than 1 become 1.

        Args:
            a (array_type): Array containing the elements to clip.
            a_min (dtype): Minimum value.
            a_max (dtype): Maximum value.
            out (array_type): Array into which the output is placed.
        Returns:
            None
        """

    @abc.abstractmethod
    def log_t(self, a, out):
        """Compute the element-wise natural logarithm.

        The natural logarithm log is the inverse of the exponential function,
        so that `log(exp(x)) = x`.

        Args:
            a (array_type): Array whose logarithm is to be computed.
            out (array_type): Array into which the output is placed.
        Returns:
            None
        """

    @abc.abstractmethod
    def sqrt_t(self, a, out):
        """Compute the positive square-root of an array, element-wise.

        Args:
            a (array_type): Array whose square root is to be computed.
            out (array_type): Array into which the output is placed.
        Returns:
            None
        """

    @abc.abstractmethod
    def sign_t(self, a, out):
        """Compute an element-wise indication of the sign of a number.

        Output has the value 1.0 if an element is positive, 0 if it is zero,
        and -1.0 if it is negative.

        Args:
            a (array_type): Array whose sign is to be computed.
            out (array_type): Array into which the output is placed.
        Returns:
            None
        """

    @abc.abstractmethod
    def binarize_v(self, v, out):
        """

        Args:
            out (array_type): Array into which the output is placed.
        Returns:
            None
        """

    @abc.abstractmethod
    def index_m_by_v(self, m, v, out):
        """

        Args:
            out (array_type): Array into which the output is placed.
        Returns:
            None
        """

    @abc.abstractmethod
    def conv2d_forward_batch(self, inputs, weights, bias, outputs,
                             padding, stride):
        """Performs a 2D convolution on a batch of images.

        Args:
            out (array_type): Array into which the output is placed.
        Returns:
            None
        """

    @abc.abstractmethod
    def conv2d_backward_batch(self, inputs, weights, padding, stride,
                              in_deltas, out_deltas, weight_deltas,
                              bias_deltas):
        """Computes the gradients for a 2D convolution on a batch of images.

        Args:
            out (array_type): Array into which the output is placed.
        Returns:
            None
        """

    @abc.abstractmethod
    def maxpool2d_forward_batch(self, inputs, window, outputs, padding,
                                stride, argmax):
        """Performs a 2D max-pooling on a batch of images.

        Args:
            out (array_type): Array into which the output is placed.
        Returns:
            None
        """

    @abc.abstractmethod
    def maxpool2d_backward_batch(self, inputs, window, outputs, padding,
                                 stride, argmax, in_deltas, out_deltas):
        """Computes the gradients for 2D max-pooling on a batch of images.

        Args:
            out (array_type): Array into which the output is placed.
        Returns:
            None
        """

    @abc.abstractmethod
    def avgpool2d_forward_batch(self, inputs, window, outputs, padding,
                                stride):
        """Performs a 2D average-pooling on a batch of images.

        Args:
            out (array_type): Array into which the output is placed.
        Returns:
            None
        """

    @abc.abstractmethod
    def avgpool2d_backward_batch(self, inputs, window, outputs, padding,
                                 stride, in_deltas, out_deltas):
        """Computes the gradients for 2D average-pooling on a batch of images.

        Args:
            out (array_type): Array into which the output is placed.
        Returns:
            None
        """

    # ---------------- Activation functions -----------------------------------

    @abc.abstractmethod
    def sigmoid(self, x, y):
        """Compute the sigmoid function.

        `y = sigmoid(x) = 1 / (1 + exp(-x))`
        Args:
            x (array_type): Input array.
            y (array_type): Output array.
        Returns:
            None
        """

    @abc.abstractmethod
    def sigmoid_deriv(self, x, y, dy, dx):
        """Backpropagate derivatives through the sigmoid function.

        Args:
            x (array_type): Inputs to the sigmoid function.
                            This argument is not used and is present only to
                            conform with other activation functions.
            y (array_type): Outputs of the sigmoid function.
            dy (array_type): Derivatives with respect to the outputs.
            dx (array_type): Array in which the derivatives with respect to
                             the inputs are placed.
        Returns:
            None
        """

    @abc.abstractmethod
    def tanh(self, x, y):
        """Compute the tanh (hyperbolic tangent) function.

        `y = tanh(x) = (e^z - e^-z) / (e^z + e^-z)`

        Args:
            x (array_type): Input array.
            y (array_type): Output array.
        Returns:
            None
        """

    @abc.abstractmethod
    def tanh_deriv(self, x, y, dy, dx):
        """Backpropagate derivatives through the tanh function.

        Args:
            x (array_type): Inputs to the tanh function.
                            This argument is not used and is present only to
                            conform with other activation functions.
            y (array_type): Outputs of the tanh function.
            dy (array_type): Derivatives with respect to the outputs.
            dx (array_type): Array in which the derivatives with respect to
                             the inputs are placed.
        Returns:
            None
        """

    @abc.abstractmethod
    def rel(self, x, y):
        """Compute the rel (rectified linear) function.

        `y = rel(x) = max(0, x)`

        Args:
            x (array_type): Input array.
            y (array_type): Output array.
        Returns:
            None
        """

    @abc.abstractmethod
    def rel_deriv(self, x, y, dy, dx):
        """Backpropagate derivatives through the rectified linear function.

        Args:
            x (array_type): Inputs to the rel function.
                            This argument is not used and is present only to
                            conform with other activation functions.
            y (array_type): Outputs of the rel function.
            dy (array_type): Derivatives with respect to the outputs.
            dx (array_type): Array in which the derivatives with respect to
                             the inputs are placed.
        Returns:
            None
        """

    @abc.abstractmethod
    def softmax_m(self, m, out):
        """Compute the softmax function over last dimension of a matrix.

        Args:
            m (array_type): Input array.
            out (array_type): Output array.
        Returns:
            None
        """
