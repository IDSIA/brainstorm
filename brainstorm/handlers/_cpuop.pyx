# coding=utf-8
from __future__ import division, print_function
import cython
from cython.view cimport array as cvarray
import numpy as np
cimport numpy as np

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t


cdef inline DTYPE_t dtype_t_max(DTYPE_t a, DTYPE_t b) nogil:
    return a if a >= b else b

cdef inline int int_max(int a, int b) nogil: return a if a >= b else b
cdef inline int int_min(int a, int b) nogil: return a if a <= b else b



@cython.boundscheck(False)
@cython.wraparound(False)
def maxpool_forward(DTYPE_t[:, :, :, ::1] inputs not None,
            tuple window not None,
            DTYPE_t[:, :, :, ::1] outputs not None,
            int pad,
            tuple strides not None,
            np.int32_t[:, :, :, :, ::1] argmax not None):
    maxpool_forward_impl(inputs, window[0], window[1], outputs, pad,
                         strides[1], strides[0], inputs.shape[0],
                         inputs.shape[1], inputs.shape[2], inputs.shape[3],
                         outputs.shape[2], outputs.shape[3], argmax)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void maxpool_forward_impl(DTYPE_t[:, :, :, ::1] inputs,
                  const int pool_h, const int pool_w,
                  DTYPE_t[:, :, :, ::1] outputs,
                  const int pad,
                  const int stride_x, const int stride_y,
                  const int n_inputs, const int n_filters,
                  const int in_h, const int in_w,
                  const int out_h, const int out_w,
                  np.int32_t[:, :, :, :, ::1] argmax) nogil:
    cdef int i, c, y, x, y_out, x_out
    cdef int y_min, y_max, x_min, x_max
    cdef int in_y, in_x,
    cdef int in_y_max = 0
    cdef int in_x_max = 0
    cdef DTYPE_t value, new_value
    for i in range(n_inputs):
        for c in range(n_filters):
            for y_out in range(out_h):
                y = y_out*stride_y-pad
                y_min = int_max(y, 0)
                y_max = int_min(y+pool_h, in_h)
                for x_out in range(out_w):
                    x = x_out*stride_x-pad
                    x_min = int_max(x, 0)
                    x_max = int_min(x+pool_w, in_w)
                    value = -1e38
                    for in_y in range(y_min, y_max):
                        for in_x in range(x_min, x_max):
                            new_value = inputs[i, c, in_y, in_x,]
                            if new_value > value:
                                value = new_value
                                in_y_max = in_y
                                in_x_max = in_x
                    outputs[i, c, y_out, x_out] = value
                    argmax[i, c, y_out, x_out, 0] = in_y_max
                    argmax[i, c, y_out, x_out, 1] = in_x_max


@cython.boundscheck(False)
@cython.wraparound(False)
def maxpool_backward(DTYPE_t[:, :, :, ::1] inputs not None,
                     tuple window not None,
                     DTYPE_t[:, :, :, ::1] outputs not None,
                     int pad,
                     tuple strides not None,
                     np.int32_t[:, :, :, :, ::1] argmax not None,
                     DTYPE_t[:, :, :, ::1] in_deltas not None,
                     DTYPE_t[:, :, :, ::1] out_deltas not None):
    maxpool_backward_impl(inputs, window[0], window[1], outputs, pad,
                          strides[1], strides[0], inputs.shape[0],
                          inputs.shape[1],inputs.shape[2], inputs.shape[3],
                          outputs.shape[2], outputs.shape[3], argmax,
                          in_deltas, out_deltas)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void maxpool_backward_impl(DTYPE_t[:, :, :, ::1] inputs,
                                const int pool_h, const int pool_w,
                                DTYPE_t[:, :, :, ::1] outputs,
                                const int pad,
                                const int stride_x, const int stride_y,
                                const int n_inputs, const int n_filters,
                                const int in_h, const int in_w,
                                const int out_h, const int out_w,
                                np.int32_t[:, :, :, :, ::1] argmax,
                                DTYPE_t[:, :, :, ::1] in_deltas,
                                DTYPE_t[:, :, :, ::1] out_deltas) nogil:
    cdef int i, c, y, x, in_y, in_x
    for i in range(n_inputs):
        for c in range(n_filters):
            for y in range(out_h):
                for x in range(out_w):
                    in_y = argmax[i, c, y, x, 0]
                    in_x = argmax[i, c, y, x, 1]
                    in_deltas[i, c, in_y, in_x] += out_deltas[i, c, y, x]
