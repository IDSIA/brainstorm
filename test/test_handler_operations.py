#!/usr/bin/env python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import numpy as np
from brainstorm.handlers import NumpyHandler

np.random.seed(1234)

NO_CON = set()

def _conv2d_forward_batch(inputs, weights, bias, outputs, pad, stride):
    """
    Loop-based implementation of 2D convolution to check against.
    """

    num_filters = weights.shape[0]
    num_images, num_input_maps, input_height, input_width = inputs.shape
    kernel_size = (weights.shape[2], weights.shape[3])

    if pad > 0:
        im = np.zeros((inputs.shape[0], inputs.shape[1],
                       inputs.shape[2] + 2 * pad,
                       inputs.shape[3] + 2 * pad))
        im[:, :, pad: -pad, pad: -pad] = inputs
        input_height += 2 * pad
        input_width += 2 * pad
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

def test_conv2d_forward_batch():
    print("\n--------- Testing conv2d_forward_batch ---------")
    _h = NumpyHandler(np.float64)
    for input_shape in ((3, 3), (5, 4), (4, 9)):
        for num_images in (1, 4):
            for num_input_maps in (1, 3):
                for num_filters in (1, 3):
                    for kernel_shape in ((1, 1), (2, 2), (3, 2)):
                        for stride in ((1, 1), (2, 2), (1, 2)):
                            for pad in (0, 1):
                                print("Checking Inputs:", (num_images,
                                      num_input_maps) + input_shape)
                                print("Filters:", (num_filters, num_input_maps)
                                      + kernel_shape)
                                print("Stride: ", stride, "Pad: ", pad)
                                inputs = np.random.rand(num_images,
                                                        num_input_maps,
                                                        *input_shape)
                                weights = np.random.rand(num_filters,
                                                         num_input_maps,
                                                         *kernel_shape)
                                bias = np.zeros(num_filters)

                                output_height = \
                                    (input_shape[0] + 2 * pad -
                                     kernel_shape[0]) / stride[0] + 1
                                output_width = \
                                    (input_shape[1] + 2 * pad -
                                     kernel_shape[1]) / stride[1] + 1

                                outputs = np.zeros((num_images, num_filters) +
                                                   (output_height,
                                                    output_width))
                                true_outputs = np.zeros((num_images,
                                                         num_filters)
                                                        + (output_height,
                                                           output_width))

                                _h.conv2d_forward_batch(inputs, weights, bias,
                                                        outputs, pad, stride)

                                _conv2d_forward_batch(inputs, weights, bias,
                                                      true_outputs, pad, stride)

                                assert np.allclose(outputs, true_outputs)

