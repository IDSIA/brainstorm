#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np
from brainstorm.handlers.base_handler import Handler
from brainstorm.handlers import _cpuop


# noinspection PyMethodOverriding
class NumpyHandler(Handler):
    __undescribed__ = {'context', 'EMPTY'}

    def __init__(self, dtype):
        self.dtype = dtype
        self.context = 'numpy'
        self.EMPTY = np.zeros(0)

    array_type = np.ndarray
    size = staticmethod(lambda x: x.size)
    shape = staticmethod(lambda x: x.shape)
    reshape = staticmethod(lambda x, s: x.reshape(s))
    slice = staticmethod(lambda x, s: x[s])

    def __describe__(self):
        return {
            '@type': self.__class__.__name__,
            'dtype': str(np.dtype(self.dtype))
        }

    def __init_from_description__(self, description):
        self.__init__(np.dtype(description['dtype']))

    def allocate(self, size):
        return np.zeros(size, dtype=self.dtype)

    @staticmethod
    def fill(mem, val):
        mem.fill(val)

    def set_from_numpy(self, mem, arr):
        mem[:] = arr.astype(self.dtype)

    def get_numpy_copy(self, arr):
        assert type(arr) == self.array_type
        return arr.copy()

    @staticmethod
    def create_from_numpy(arr):
        return arr.copy()

    @staticmethod
    def copy_to(dest, src):
        # FIXME: change casting to 'no'
        np.copyto(dest, src, casting='same_kind')

    def zeros(self, shape):
        return np.zeros(shape=shape, dtype=self.dtype)

    def ones(self, shape):
        return np.ones(shape=shape, dtype=self.dtype)

    # ---------------- Mathematical Operations ---------------- #

    @staticmethod
    def sum_t(a, axis, out):
        assert axis is None or (len(a.shape) < 3 and (axis == 0 or axis == 1))

        if axis is not None and len(out.shape) == len(a.shape):
            keepdims = True
        else:
            keepdims = False
        np.sum(a, axis=axis, out=out, keepdims=keepdims)

    @staticmethod
    def dot_mm(a, b, out, transa='N', transb='N'):
        x = a.T if (transa == 'T') else a
        y = b.T if (transb == 'T') else b
        # np.dot(x, y, out)  # FIXME: doesn't work with strided out
        out[:] = np.dot(x, y)

    @staticmethod
    def dot_add_mm(a, b, out, transa='N', transb='N'):
        x = a.T if (transa == 'T') else a
        y = b.T if (transb == 'T') else b
        out[:] += np.dot(x, y)

    @staticmethod
    def mult_tt(a, b, out):
        np.multiply(a, b, out)

    @staticmethod
    def mult_add_tt(a, b, out):
        out[:] += a * b

    @staticmethod
    def mult_st(a, b, out):
        np.multiply(a, b, out)

    @staticmethod
    def add_tt(a, b, out):
        assert a.shape == b.shape == out.shape
        out[:] = a + b

    @staticmethod
    def add_st(s, t, out):
        out[:] = t + s

    @staticmethod
    def subtract_tt(a, b, out):
        assert a.shape == b.shape == out.shape
        out[:] = a - b

    @staticmethod
    def add_mv(m, v, out):
        """
        Add (M, N) matrix elementwise to (1, N) or (N, 1) or (N,) vector using
        broadcasting.
        """
        # TODO: Generalize to support broadcast along both dimensions
        assert len(m.shape) == 2
        assert (len(v.shape) == 2 and (v.shape[0] == 1 or v.shape[1] == 1)) \
            or (len(v.shape) == 1 and v.shape[0] == m.shape[1])
        out[:] = m + v

    @staticmethod
    def broadcast_features_t(a, out):
        assert len(a.shape) == 3
        assert a.shape[2] == 1
        assert len(out.shape) > 2
        num_extra_dims = len(out.shape) - 3
        shape_to_add = tuple([1] * num_extra_dims)
        b = np.reshape(a, a.shape + shape_to_add)

        shape_to_tile = (1, 1) + out.shape[2:]
        out[:] = np.tile(b, shape_to_tile)

    @staticmethod
    def clip_t(a, a_min, a_max, out):
        assert a_max >= a_min
        np.clip(a, a_min, a_max, out)

    @staticmethod
    def log_t(a, out):
        np.log(a, out)

    @staticmethod
    def divide_tt(a, b, out):
        out[:] = a / b

    @staticmethod
    def divide_mv(m, v, out):
        """
        Divide (M, N) matrix elementwise by a (1, N) or (N, 1) or (N,) vector
        using broadcasting.
        """
        assert len(m.shape) == 2
        assert (len(v.shape) == 2 and (v.shape[0] == 1 or v.shape[1] == 1)) \
            or (len(v.shape) == 1 and v.shape[0] == m.shape[1])
        out[:] = m / v

    @staticmethod
    def mult_mv(m, v, out):
        """
        Multiply (M, N) matrix elementwise by a (1, N) or (N, 1) or (N,) vector
        using broadcasting.
        """
        assert len(m.shape) == 2
        assert (len(v.shape) == 2 and (v.shape[0] == 1 or v.shape[1] == 1)) \
            or (len(v.shape) == 1 and v.shape[0] == m.shape[1])
        out[:] = m * v

    @staticmethod
    def binarize_v(v, out):
        out[:] = 0.
        for i in range(v.shape[0]):
            out[i, int(v[i])] = 1.0

    @staticmethod
    def index_m_by_v(m, v, out):
        for i in range(m.shape[0]):
            out[i] = m[i, int(v[i])]

    @staticmethod
    def get_im2col_map(num_input_maps, input_rows, input_cols,
                       kernel_size, stride):
        # im2col built upon http://stackoverflow.com/a/30110497
        # Returns a 2D map which performs im2col on a 3D array
        # Apply map to a 3D array using numpy.take(array, map)

        # Parameters
        col_extent = input_cols - kernel_size[1] + 1
        row_extent = input_rows - kernel_size[0] + 1

        # Get Starting block indices
        start_idx = np.arange(kernel_size[0])[:, None] * input_cols + \
            np.arange(kernel_size[1])

        # Get offsetted indices across the height and width of input array
        offset_idx = np.arange(row_extent)[:, None] * input_cols + \
            np.arange(col_extent)

        indices = start_idx.ravel()[:, None] + \
            offset_idx[::stride[0], ::stride[1]].ravel()
        adder = (np.arange(num_input_maps) * input_rows * input_cols)\
            .reshape((num_input_maps, 1, 1))

        # Extend to multiple input maps
        im2col_map = indices + adder

        # Reshape to stack input maps
        im2col_map = im2col_map.reshape((kernel_size[0] * kernel_size[1] *
                                         num_input_maps, -1))

        return im2col_map

    @classmethod
    def conv2d_forward_batch(cls, inputs, weights, bias, outputs,
                             padding, stride):

        num_filters = weights.shape[0]
        num_images, num_input_maps, input_rows, input_cols = inputs.shape
        kernel_size = (weights.shape[2], weights.shape[3])
        out_shape = outputs.shape[1:]

        im2col_map = cls.get_im2col_map(num_input_maps,
                                        input_rows + 2 * padding,
                                        input_cols + 2 * padding,
                                        kernel_size, stride)

        # reshape
        for i in range(num_images):
            # pad
            if padding == 0:
                im = inputs[i]
            else:
                im = np.zeros((inputs.shape[1], inputs.shape[2] + 2 * padding,
                               inputs.shape[3] + 2 * padding))
                im[:, padding: -padding, padding: -padding] = inputs[i]

            # Get all actual indices & index into input array for output
            col = np.take(im, im2col_map)

            # multiply
            reshaped_weights = weights.reshape(num_filters,
                                               kernel_size[0] *
                                               kernel_size[1] *
                                               num_input_maps)
            outputs[i] = np.dot(reshaped_weights, col).reshape(out_shape)

        outputs += bias.reshape((1, num_filters, 1, 1))

    @classmethod
    def conv2d_backward_batch(cls, inputs, weights, padding, stride, in_deltas,
                              out_deltas, weight_deltas, bias_deltas):
        if stride != (1, 1):
            raise NotImplementedError("Strides > 1 for ConvolutionLayer2D are "
                                      "not supported yet.")
        num_filters = weights.shape[0]
        num_images, num_input_maps, input_rows, input_cols = inputs.shape
        _, num_output_maps, output_rows, output_cols = out_deltas.shape
        kernel_size = (weights.shape[2], weights.shape[3])

        im2col_map = cls.get_im2col_map(num_input_maps,
                                        input_rows + 2 * padding,
                                        input_cols + 2 * padding,
                                        kernel_size, stride)

        dpadh = ((input_rows + 2 * padding - 1) * stride[0] + kernel_size[0] -
                 output_rows) // 2
        dpadw = ((input_cols + 2 * padding - 1) * stride[1] + kernel_size[1] -
                 output_cols) // 2
        col2im_map = cls.get_im2col_map(num_output_maps,
                                        output_rows + 2 * dpadh,
                                        output_cols + 2 * dpadw,
                                        kernel_size, (1, 1))
        weight_deltas.fill(0.0)
        bias_deltas.fill(0.0)
        for i in range(num_images):
            # pad
            if padding == 0:
                im = inputs[i]
            else:
                im = np.zeros((num_input_maps, input_rows + 2 * padding,
                               input_cols + 2 * padding))
                im[:, padding: -padding, padding: -padding] = inputs[i]

            # Get all actual indices & index into input array for final output
            col = np.take(im, im2col_map)

            # Compute gradients
            reshaped_dweights = weight_deltas.reshape(num_filters,
                                                      kernel_size[0] *
                                                      kernel_size[1] *
                                                      num_input_maps)
            reshaped_out_deltas = out_deltas[i].reshape((num_filters, -1))
            cls.dot_add_mm(reshaped_out_deltas, col, out=reshaped_dweights,
                           transb='T')
            bias_deltas += np.sum(reshaped_out_deltas, axis=1)

            # Compute in_deltas

            # But first some reshaping magic to rotate all kernels twice by 90
            prod_k = kernel_size[0] * kernel_size[1]
            _weights = np.fliplr(weights.reshape(-1, prod_k)).reshape(
                weights.shape)
            reshaped_weights = _weights.swapaxes(0, 1).reshape(num_input_maps,
                                                               prod_k *
                                                               num_filters)

            im = np.zeros((num_filters, output_rows + 2 * dpadh,
                           output_cols + 2 * dpadw))
            im[:, dpadh: -dpadh, dpadw: -dpadw] = out_deltas[i]

            col = np.take(im, col2im_map)

            # temp contains deltas WRT padded inputs
            new_shape = (num_input_maps,
                         input_rows + 2 * padding,
                         input_cols + 2 * padding)
            temp = np.dot(reshaped_weights, col).reshape(new_shape)
            # Remove padding
            if padding == 0:
                in_deltas[i] += temp
            else:
                in_deltas[i] += temp[:, padding: -padding, padding: -padding]


    def pool2d_forward_batch(self, inputs, window, outputs, padding,
                             stride, argmax):
        _cpuop.maxpool_forward(inputs, window, outputs, padding,
                               stride, argmax)




    def pool2d_backward_batch(self, inputs, window, outputs, padding, stride,
                              argmax, in_deltas, out_deltas):
        _cpuop.maxpool_backward(inputs, window, outputs, padding, stride,
                                argmax, in_deltas, out_deltas)


    # ---------------- Activation functions -----------------------------------

    @staticmethod
    def sigmoid(x, y):
        y[:] = 1. / (1. + np.exp(-x))

    @staticmethod
    def sigmoid_deriv(x, y, dy, dx):
        dx[:] = dy * y * (1. - y)

    @staticmethod
    def tanh(x, y):
        np.tanh(x, y)

    @staticmethod
    def tanh_deriv(x, y, dy, dx):
        dx[:] = dy * (1. - y * y)

    @staticmethod
    def rel(x, y):
        y[:] = x * (x > 0)

    @staticmethod
    def rel_deriv(x, y, dy, dx):
        dx[:] = dy * (y > 0)

    @staticmethod
    def softmax_m(m, out):
        """Applies softmax to matrix over last dimension"""
        maxes = np.amax(m, axis=1, keepdims=True)
        e = np.exp(m - maxes)
        out[:] = e / np.sum(e, axis=1, keepdims=True)
