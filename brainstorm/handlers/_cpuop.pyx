# coding=utf-8
from __future__ import division, print_function

cimport numpy as np
from cython.view cimport array as cvarray
from libc.float cimport FLT_MAX, DBL_MAX

import cython
import numpy as np


ctypedef fused DTYPE_t:
    np.float32_t
    np.float64_t

cdef inline DTYPE_t dtype_t_max(DTYPE_t a, DTYPE_t b) nogil:
    return a if a >= b else b

cdef inline int int_max(int a, int b) nogil: return a if a >= b else b
cdef inline int int_min(int a, int b) nogil: return a if a <= b else b

# ------------------------- Cudarray-based routines ------------------------- #
# Please see Third Party License file for license information

@cython.boundscheck(False)
@cython.wraparound(False)
def maxpool_forward(DTYPE_t[:, :, :, ::1] inputs not None,
            tuple kernel not None,
            DTYPE_t[:, :, :, ::1] outputs not None,
            int padding,
            tuple strides not None,
            DTYPE_t[:, :, :, ::1] argmax not None):
    cdef int pool_h = kernel[0]
    cdef int pool_w = kernel[1]
    cdef int stride_x = strides[1]
    cdef int stride_y = strides[0]
    cdef int n_inputs = inputs.shape[0]
    cdef int n_channels = inputs.shape[3]
    cdef int in_h = inputs.shape[1]
    cdef int in_w = inputs.shape[2]
    cdef int out_h = outputs.shape[1]
    cdef int out_w = outputs.shape[2]
    cdef int i, c, y, x, y_out, x_out
    cdef int y_min, y_max, x_min, x_max
    cdef int in_y, in_x
    cdef int max_idx = -1
    cdef DTYPE_t value, new_value
    cdef DTYPE_t min_value

    # for output compatibility with cudnn, we must
    # use the minimum allowed value
    if DTYPE_t is np.float32_t:
        min_value = -FLT_MAX
    else:
        min_value = -DBL_MAX

    with nogil:
        for i in range(n_inputs):
            for c in range(n_channels):
                for y_out in range(out_h):
                    y = y_out * stride_y - padding
                    y_min = int_max(y, 0)
                    y_max = int_min(y + pool_h, in_h)
                    for x_out in range(out_w):
                        x = x_out * stride_x - padding
                        x_min = int_max(x, 0)
                        x_max = int_min(x + pool_w, in_w)
                        value = min_value
                        max_idx = -1
                        for in_y in range(y_min, y_max):
                            for in_x in range(x_min, x_max):
                                new_value = inputs[i, in_y, in_x, c]
                                if new_value > value:
                                    value = new_value
                                    max_idx = (in_y * in_w + in_x) * \
                                              n_channels + c
                        outputs[i, y_out, x_out, c] = value
                        argmax[i, y_out, x_out, c] = <DTYPE_t>max_idx
                        if max_idx == -1:
                            outputs[i, y_out, x_out, c] = 0

@cython.boundscheck(False)
@cython.wraparound(False)
def maxpool_backward(DTYPE_t[:, :, :, ::1] inputs not None,
                     tuple kernel not None,
                     DTYPE_t[:, :, :, ::1] outputs not None,
                     const int padding,
                     tuple strides not None,
                     DTYPE_t[:, :, :, ::1] argmax not None,
                     DTYPE_t[:, :, :, ::1] in_deltas not None,
                     DTYPE_t[:, :, :, ::1] out_deltas not None):
    cdef int pool_h = kernel[0]
    cdef int pool_w = kernel[1]
    cdef int stride_x = strides[1]
    cdef int stride_y = strides[0]
    cdef int n_inputs = inputs.shape[0]
    cdef int n_channels = inputs.shape[3]
    cdef int in_h = inputs.shape[1]
    cdef int in_w = inputs.shape[2]
    cdef int out_h = outputs.shape[1]
    cdef int out_w = outputs.shape[2]
    cdef int i, c, y, x, in_y, in_x, map_loc, max_idx
    with nogil:
        for i in range(n_inputs):
            for c in range(n_channels):
                for y in range(out_h):
                    for x in range(out_w):
                        max_idx = <int>(argmax[i, y, x, c])
                        if max_idx != -1:
                            map_loc = max_idx // n_channels
                            in_y = map_loc // in_w
                            in_x = map_loc % in_w
                            if in_y >= 0 and in_x >= 0:
                                in_deltas[i, in_y, in_x, c] += \
                                    out_deltas[i, y, x, c]


@cython.boundscheck(False)
@cython.wraparound(False)
def avgpool_forward(DTYPE_t[:, :, :, ::1] inputs not None,
            tuple kernel not None,
            DTYPE_t[:, :, :, ::1] outputs not None,
            int padding,
            tuple strides not None):
    # NOTE: Modified to count only non-padding pixels
    cdef int pool_h = kernel[0]
    cdef int pool_w = kernel[1]
    cdef int stride_x = strides[1]
    cdef int stride_y = strides[0]
    cdef int n_inputs = inputs.shape[0]
    cdef int n_channels = inputs.shape[3]
    cdef int in_h = inputs.shape[1]
    cdef int in_w = inputs.shape[2]
    cdef int out_h = outputs.shape[1]
    cdef int out_w = outputs.shape[2]
    cdef int i, c, y, x, y_out, x_out
    cdef int y_min, y_max, x_min, x_max
    cdef int in_y, in_x,
    cdef int in_y_max = 0
    cdef int in_x_max = 0
    cdef DTYPE_t value, new_value
    cdef int pool_size = 0
    with nogil:
        for i in range(n_inputs):
            for c in range(n_channels):
                for y_out in range(out_h):
                    y = y_out * stride_y - padding
                    y_min = int_max(y, 0)
                    y_max = int_min(y + pool_h, in_h)
                    for x_out in range(out_w):
                        x = x_out * stride_x - padding
                        x_min = int_max(x, 0)
                        x_max = int_min(x + pool_w, in_w)
                        value = 0
                        in_y_max = -1
                        in_x_max = -1
                        for in_y in range(y_min, y_max):
                            for in_x in range(x_min, x_max):
                                value += inputs[i, in_y, in_x, c]
                        pool_size = int_max((y_max - y_min) * (x_max - x_min),
                                            1)
                        outputs[i, y_out, x_out, c] = value / pool_size


@cython.boundscheck(False)
@cython.wraparound(False)
def avgpool_backward(DTYPE_t[:, :, :, ::1] inputs not None,
                     tuple kernel not None,
                     DTYPE_t[:, :, :, ::1] outputs not None,
                     const int padding,
                     tuple strides not None,
                     DTYPE_t[:, :, :, ::1] in_deltas not None,
                     DTYPE_t[:, :, :, ::1] out_deltas not None):
    # NOTE: No modification need to count only non-padding pixels
    cdef int pool_h = kernel[0]
    cdef int pool_w = kernel[1]
    cdef int stride_x = strides[1]
    cdef int stride_y = strides[0]
    cdef int n_inputs = inputs.shape[0]
    cdef int n_channels = inputs.shape[3]
    cdef int in_h = inputs.shape[1]
    cdef int in_w = inputs.shape[2]
    cdef int out_h = outputs.shape[1]
    cdef int out_w = outputs.shape[2]
    cdef int i, c, y, x, x_min, x_max, y_min, y_max, x_out, y_out
    cdef int pool_size = 0
    with nogil:
        for i in range(n_inputs):
            for c in range(n_channels):
                for y_out in range(out_h):
                    y = y_out * stride_y - padding
                    y_min = int_max(y, 0)
                    y_max = int_min(y + pool_h, in_h)
                    for x_out in range(out_w):
                        x = x_out * stride_x-padding
                        x_min = int_max(x, 0)
                        x_max = int_min(x + pool_w, in_w)
                        pool_size = (y_max - y_min) * (x_max - x_min)
                        for yy in range(y_min, y_max):
                            for xx in range(x_min, x_max):
                                 in_deltas[i, yy, xx, c] += \
                                     out_deltas[i, y_out, x_out, c] / pool_size


@cython.boundscheck(False)
@cython.wraparound(False)
def _crop_images(DTYPE_t[:, :, :, :, ::1] inputs not None,
                int height,
                int width,
                np.int_t[:] row_indices,
                np.int_t[:] col_indices,
                DTYPE_t[:, :, :, :, ::1] outputs not None):
    """
    Args:
        inputs (numpy.ndarray[ndim=5]):
            5 dimensional Numpy array
        height (int):
            height of output crop
        width (int):
            width of output crop
        row_indices (numpy.ndarray[ndim=1]):
            1D Numpy array containing location of top-left row positions
            with inputs.shape[1] elements (one for each item in batch)
        col_indices (numpy.ndarray[ndim=1]):
            1D Numpy array containing location of top-left column positions
            with inputs.shape[1] elements (one for each item in batch)
        outputs (numpy.ndarray[ndim=5]):
            5 dimensional Numpy array
    """
    cdef int batch_size = row_indices.shape[0]
    cdef int time_steps = inputs.shape[0]
    cdef int num_channels = inputs.shape[4]
    cdef int start_row, start_col, i, j, k, l, t
    with nogil:
        for i in range(batch_size):
            start_row = row_indices[i]
            start_col = col_indices[i]
            for t in range(0, time_steps):
                for k in range(0, height):
                    for l in range(0, width):
                        for c in range(0, num_channels):
                            outputs[t, i, k, l, c] = inputs[t, i,
                                                            k + start_row,
                                                            l + start_col, c]

# -------------------------- Caffe2-based routines -------------------------- #
# Please see Third Party License file for license information

@cython.boundscheck(False)
@cython.wraparound(False)
def im2col(DTYPE_t[::1] flat_in not None,
           const int height, const int width, const int channels,
           const int kernel_h, const int kernel_w,
           const int pad_t, const int pad_l, const int pad_b, const int pad_r,
           const int stride_h, const int stride_w,
           DTYPE_t[::1] flat_col not None):

    cdef int height_col = (height + pad_t + pad_b - kernel_h) // stride_h + 1
    cdef int width_col = (width + pad_l + pad_r - kernel_w) // stride_w + 1
    cdef int h_pad = -pad_t
    cdef int col_idx = 0
    cdef int h, w_pad, w, ih, iw
    with nogil:
        for h in range(height_col):
            w_pad = -pad_l
            for w in range(width_col):
                for ih in range(h_pad, h_pad + kernel_h):
                    for iw in range(w_pad, w_pad + kernel_w):
                        if 0 <= ih < height and 0 <= iw < width:
                            flat_col[col_idx: col_idx + channels] = \
                                flat_in[(ih * width + iw) * channels:
                                       (ih * width + iw) * channels + channels]
                        else:
                            flat_col[col_idx: col_idx + channels] = 0
                        col_idx += channels
                w_pad += stride_w
            h_pad += stride_h


@cython.boundscheck(False)
@cython.wraparound(False)
def col2im(DTYPE_t[::1] flat_col not None,
           const int height, const int width, const int channels,
           const int kernel_h, const int kernel_w,
           const int pad_t, const int pad_l, const int pad_b, const int pad_r,
           const int stride_h, const int stride_w,
           DTYPE_t[::1] flat_in not None):
    cdef int height_col = (height + pad_t + pad_b - kernel_h) // stride_h + 1
    cdef int width_col = (width + pad_l + pad_r - kernel_w) // stride_w + 1
    cdef int h_pad = -pad_t
    cdef int im_patch_idx = 0
    cdef int col_idx = 0
    cdef int h, w_pad, w, ih, iw, idx
    with nogil:
        for h in range(height_col):
            w_pad = -pad_l
            for w in range(width_col):
                im_patch_idx = (h_pad * width + w_pad) * channels
                for ih in range(h_pad, h_pad + kernel_h):
                    for iw in range(w_pad, w_pad + kernel_w):
                        if 0 <= ih < height and 0 <= iw < width:
                            for idx in range(channels):
                                flat_in[im_patch_idx + idx] += flat_col[col_idx
                                                                        +  idx]
                        im_patch_idx += channels
                        col_idx += channels
                    im_patch_idx += channels * (width - kernel_w)
                w_pad += stride_w
            h_pad += stride_h
