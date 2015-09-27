#!/usr/bin/env python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import numpy as np
import pytest
from brainstorm.handlers import NumpyHandler
from brainstorm.optional import has_cudnn

np.random.seed(1234)

NO_CON = set()


def _conv2d_forward_batch(inputs, weights, bias, outputs, padding, stride):
    """
    Loop-based implementation of 2D convolution to check against.
    """

    num_filters = weights.shape[0]
    num_images, num_input_maps, input_height, input_width = inputs.shape
    kernel_size = (weights.shape[2], weights.shape[3])

    if padding > 0:
        im = np.zeros((inputs.shape[0], inputs.shape[1],
                       inputs.shape[2] + 2 * padding,
                       inputs.shape[3] + 2 * padding))
        im[:, :, padding: -padding, padding: -padding] = inputs
        input_height += 2 * padding
        input_width += 2 * padding
    else:
        im = inputs
    for i in range(num_images):
        x_f = -1
        for m in range(0, input_height - kernel_size[0] + 1, stride[0]):
            x_f += 1
            y_f = -1
            for n in range(0, input_width - kernel_size[1] + 1, stride[1]):
                y_f += 1
                for f in range(num_filters):
                    for k in range(num_input_maps):
                        for x in range(kernel_size[0]):
                            for y in range(kernel_size[1]):
                                outputs[i, f, x_f, y_f] \
                                    += weights[f, k, x, y] * \
                                    im[i, k, m + x, n + y]

    for i in range(num_images):
        for j in range(num_filters):
            for k in range(outputs.shape[2]):
                for l in range(outputs.shape[3]):
                    outputs[i, j, k, l] += bias[j]


def test_get_im2col_map():  # TODO
    pass


def test_conv2d_forward_batch_numpy():
    _h = NumpyHandler(np.float64)
    for input_shape in ((3, 3), (5, 4), (4, 9)):
        for nr_images in (1, 4):
            for nr_input_maps in (1, 3):
                for nr_filters in (1, 3):
                    for kernel_shape in ((1, 1), (2, 2), (3, 2)):
                        for stride in ((1, 1), (2, 2), (1, 2)):
                            for padding in (0, 1):
                                inputs = np.random.rand(nr_images,
                                                        nr_input_maps,
                                                        *input_shape)
                                weights = np.random.rand(nr_filters,
                                                         nr_input_maps,
                                                         *kernel_shape)
                                bias = np.zeros(nr_filters)

                                output_height = \
                                    (input_shape[0] + 2 * padding -
                                     kernel_shape[0]) / stride[0] + 1
                                output_width = \
                                    (input_shape[1] + 2 * padding -
                                     kernel_shape[1]) / stride[1] + 1

                                outputs = np.zeros((nr_images, nr_filters) +
                                                   (output_height,
                                                    output_width))
                                true_outputs = np.zeros((nr_images,
                                                         nr_filters) +
                                                        (output_height,
                                                         output_width))

                                _h.conv2d_forward_batch(inputs, weights, bias,
                                                        outputs, padding,
                                                        stride)

                                _conv2d_forward_batch(inputs, weights, bias,
                                                      true_outputs, padding,
                                                      stride)

                                passed = np.allclose(outputs, true_outputs)
                                if not passed:
                                    print("Failed for Inputs:", (nr_images,
                                          nr_input_maps) + input_shape)
                                    print("Filters:",
                                          (nr_filters, nr_input_maps) +
                                          kernel_shape)
                                    print("Stride: ", stride, "padding: ",
                                          padding)

                                assert passed


@pytest.mark.skipif(has_cudnn is False, reason='requires cuDNN wrappers')
def test_conv2d_forward_batch_pycuda():
    from brainstorm.handlers import PyCudaHandler
    _h = PyCudaHandler(init_cudnn=True)
    for input_shape in ((3, 3), (5, 4), (4, 9)):
        for nr_images in (1, 4):
            for nr_input_maps in (1, 3):
                for nr_filters in (1, 3):
                    for kernel_shape in ((1, 1), (2, 2), (3, 2)):
                        for stride in ((1, 1), (2, 2), (1, 2)):
                            for padding in (0, 1):

                                inputs = np.random.rand(nr_images,
                                                        nr_input_maps,
                                                        *input_shape)
                                weights = np.random.rand(nr_filters,
                                                         nr_input_maps,
                                                         *kernel_shape)
                                bias = np.random.rand(nr_filters)

                                output_height = \
                                    (input_shape[0] + 2 * padding -
                                     kernel_shape[0]) / stride[0] + 1
                                output_width = \
                                    (input_shape[1] + 2 * padding -
                                     kernel_shape[1]) / stride[1] + 1

                                true_outputs = np.zeros((nr_images,
                                                         nr_filters) +
                                                        (output_height,
                                                         output_width))

                                _conv2d_forward_batch(inputs, weights,
                                                      bias, true_outputs,
                                                      padding, stride)

                                outputs = np.zeros(
                                    (nr_images, nr_filters) +
                                    (output_height, output_width))
                                i_dev = _h.create_from_numpy(inputs)
                                w_dev = _h.create_from_numpy(weights)
                                b_dev = _h.create_from_numpy(bias)
                                o_dev = _h.create_from_numpy(outputs)
                                _h.conv2d_forward_batch(i_dev, w_dev,
                                                        b_dev, o_dev,
                                                        padding, stride)
                                outputs = _h.get_numpy_copy(o_dev)
                                passed = np.allclose(outputs, true_outputs)
                                if not passed:
                                    print("Checking Inputs:",
                                          (nr_images, nr_input_maps) +
                                          input_shape)
                                    print("Filters:",
                                          (nr_filters, nr_input_maps) +
                                          kernel_shape)
                                    print("Stride: ", stride, "padding: ",
                                          padding)
                                assert passed
